import math
import re
import uuid
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from hairy_graph import config
from hairy_graph.database import (
    BubbleRecord,
    ClusterRecord,
    archive_most_recent_active_state_bubble,
    count_active_bubbles,
    get_active_bubbles_by_cluster,
    get_active_state_bubbles_with_min_incoming_links,
    get_all_active_bubbles,
    get_all_clusters,
    get_bubble_by_id,
    get_context_cloud_by_bubble_id,
    get_recent_active_bubbles,
    get_recent_active_context_clouds,
    get_undirected_neighbors,
    increment_incoming_links,
    insert_bubble,
    insert_bubble_link,
    insert_cluster,
    insert_context_cloud,
    set_bubble_node_type,
    set_bubble_status,
    update_cluster_summary,
)


def estimate_tokens(text: str) -> int:
    """Calculate token estimate: ceil(word_count / 0.75)."""
    words = text.split()
    if not words:
        return 0
    return math.ceil(len(words) / config.TOKEN_WORD_RATIO)


def truncate_text_by_words(text: str, max_tokens: int) -> str:
    """Keep at most floor(max_tokens * 0.75) words so estimate never exceeds max_tokens."""
    max_words = math.floor(max_tokens * config.TOKEN_WORD_RATIO)
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def extract_extractive_summary(prompt: str, response: str) -> str:
    """Extractive summary: first 2 sentences of response (fallback to prompt if response is empty)."""
    source = response.strip() if response.strip() else prompt.strip()
    if not source:
        return ""
    # Split by sentence-ending punctuation followed by whitespace or end of string
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", source) if s.strip()]
    if not sentences:
        return source[:200]
    return " ".join(sentences[:2])


def extract_keywords(text: str, existing_corpus: List[str], count: int = config.KEYWORD_COUNT) -> List[str]:
    """
    Extract top keywords from prompt + response:
    Stop-word-filtered term frequency, breaking ties by IDF, then alphabetically.
    Deterministic even on empty corpus.
    """
    # Tokenize words >= 2 chars, lowercase alphanumeric + underscore
    tokens = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text)]
    filtered_tokens = [w for w in tokens if w not in ENGLISH_STOP_WORDS]
    if not filtered_tokens:
        return []

    tf = Counter(filtered_tokens)

    # Document frequency calculation across active corpus for tie-breaking
    n_docs = len(existing_corpus)
    df: Dict[str, int] = Counter()
    if n_docs > 0:
        for doc in existing_corpus:
            doc_words = set(re.findall(r"\b[a-zA-Z0-9_]{2,}\b", doc.lower()))
            for w in set(filtered_tokens):
                if w in doc_words:
                    df[w] += 1

    # Deterministic scoring: (-tf, -idf, word)
    candidate_items = []
    for word, freq in tf.items():
        if n_docs > 0:
            doc_freq = df.get(word, 0)
            idf = math.log((1.0 + n_docs) / (1.0 + doc_freq)) + 1.0
        else:
            idf = 1.0
        candidate_items.append((-freq, -idf, word))

    candidate_items.sort()
    return [item[2] for item in candidate_items[:count]]


def fit_shared_vectorizer(corpus: List[str]) -> Optional[TfidfVectorizer]:
    """
    Fit TfidfVectorizer(stop_words='english') once over the corpus.
    Gracefully returns None if corpus has empty vocabulary or only stop words.
    """
    if not corpus:
        return None
    try:
        vec = TfidfVectorizer(stop_words="english")
        vec.fit(corpus)
        if not hasattr(vec, "vocabulary_") or not vec.vocabulary_:
            return None
        return vec
    except ValueError:
        # Handles empty vocabulary or all-stop-words corpus
        return None


def compute_similarity(vec: Optional[TfidfVectorizer], text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts using a shared fitted vectorizer."""
    if vec is None:
        return 0.0
    try:
        va = vec.transform([text_a])
        vb = vec.transform([text_b])
        if va.nnz == 0 or vb.nnz == 0:
            return 0.0
        sim = cosine_similarity(va, vb)[0, 0]
        return float(sim)
    except Exception:
        return 0.0


def format_cloud_context(raw_prompt: str, raw_response: str) -> str:
    """Format single context cloud as chat history."""
    return f"User: {raw_prompt}\nAssistant: {raw_response}"


def get_bubble_search_text(bubble: BubbleRecord) -> str:
    return f"{bubble.summary} {' '.join(bubble.keywords)}".strip()


def get_cluster_search_text(cluster: ClusterRecord, active_member_bubbles: List[BubbleRecord]) -> str:
    member_texts = " ".join(get_bubble_search_text(b) for b in active_member_bubbles)
    return f"{cluster.name} {cluster.summary} {member_texts}".strip()


# Core Retrieval Engine
def retrieve_context(conn, query: str, max_tokens: int = config.DEFAULT_MAX_TOKENS) -> Dict[str, Any]:
    active_count = count_active_bubbles(conn)

    # Cold Start Mode (< 10 active bubbles)
    if active_count < config.COLD_START_THRESHOLD:
        clouds = get_recent_active_context_clouds(conn, limit=config.COLD_START_RECENT_LIMIT)
        if not clouds:
            return {
                "mode": "cold_start",
                "context": "",
                "tokens_used": 0,
                "debug_trace": [],
            }

        accumulated: List[str] = []
        for cloud in clouds:
            cloud_str = format_cloud_context(cloud.raw_prompt, cloud.raw_response)
            test_context = "\n\n".join(accumulated + [cloud_str])
            cost = estimate_tokens(test_context)

            if cost <= max_tokens:
                accumulated.append(cloud_str)
            elif not accumulated:
                # Even the newest cloud exceeds max_tokens -> truncate it by words
                truncated = truncate_text_by_words(cloud_str, max_tokens)
                accumulated.append(truncated)
                break
            else:
                break

        # Output kept clouds oldest -> newest
        accumulated.reverse()
        final_context = "\n\n".join(accumulated)
        tokens_used = estimate_tokens(final_context) if final_context else 0

        return {
            "mode": "cold_start",
            "context": final_context,
            "tokens_used": tokens_used,
            "debug_trace": [],
        }

    # Hierarchical Graph Retrieval Mode (active bubbles >= 10)
    active_bubbles = get_all_active_bubbles(conn)
    bubble_corpus = [get_bubble_search_text(b) for b in active_bubbles]

    # Fit vectorizer ONCE over active bubble texts
    vec = fit_shared_vectorizer(bubble_corpus)

    # 1. Score query against clusters and select top 2
    clusters = get_all_clusters(conn)
    cluster_scores: List[Tuple[float, ClusterRecord, List[BubbleRecord]]] = []
    for c in clusters:
        members = [b for b in active_bubbles if b.cluster_id == c.id]
        c_text = get_cluster_search_text(c, members)
        sim = compute_similarity(vec, query, c_text)
        cluster_scores.append((sim, c, members))

    # Sort clusters by similarity descending
    cluster_scores.sort(key=lambda x: x[0], reverse=True)
    top_clusters = cluster_scores[: config.GRAPH_TOP_CLUSTERS]

    debug_trace: List[Dict[str, Any]] = []
    for sim, c, _ in top_clusters:
        debug_trace.append(
            {
                "type": "cluster",
                "id": c.id,
                "cluster_id": c.id,
                "similarity": round(float(sim), 4),
                "node_type": None,
                "depth": 0,
                "included": True,
            }
        )

    # 2. Within those clusters, rank active bubbles by cosine similarity to query
    top_cluster_ids = {c.id for _, c, _ in top_clusters}
    candidate_bubbles = [b for b in active_bubbles if b.cluster_id in top_cluster_ids]

    bubble_scores: List[Tuple[float, BubbleRecord]] = []
    for b in candidate_bubbles:
        b_text = get_bubble_search_text(b)
        sim = compute_similarity(vec, query, b_text)
        bubble_scores.append((sim, b))

    bubble_scores.sort(key=lambda x: x[0], reverse=True)

    # Top 2 as seeds; fallback to newest active bubbles if all similarities are 0
    all_zero = not bubble_scores or all(s[0] == 0.0 for s in bubble_scores)
    if all_zero:
        seeds = get_recent_active_bubbles(conn, limit=config.GRAPH_SEEDS_PER_CLUSTER)
    else:
        seeds = [b for _, b in bubble_scores[: config.GRAPH_SEEDS_PER_CLUSTER]]

    # 3. Traverse bubble_links as undirected graph up to max_depth = 2, weight >= 0.35
    visited: Dict[str, Dict[str, Any]] = {}
    visit_order: List[str] = []
    queue: deque[Tuple[str, int]] = deque([(s.id, 0) for s in seeds])

    while queue:
        curr_id, depth = queue.popleft()
        if curr_id in visited:
            continue

        curr_bubble = get_bubble_by_id(conn, curr_id)
        if not curr_bubble or curr_bubble.status != "active":
            continue

        b_text = get_bubble_search_text(curr_bubble)
        sim = compute_similarity(vec, query, b_text)
        visited[curr_id] = {
            "bubble": curr_bubble,
            "depth": depth,
            "similarity": sim,
        }
        visit_order.append(curr_id)

        if depth < config.GRAPH_TRAVERSAL_MAX_DEPTH:
            neighbors = get_undirected_neighbors(conn, curr_id, min_weight=config.GRAPH_EDGE_MIN_WEIGHT)
            for nbr_id in neighbors:
                if nbr_id not in visited:
                    queue.append((nbr_id, depth + 1))

    # 4. Order visited bubbles: CORE first, then STATE; within group by similarity descending
    traversed_items = list(visited.values())
    traversed_items.sort(
        key=lambda x: (
            1 if x["bubble"].node_type == "CORE" else 0,
            x["similarity"],
        ),
        reverse=True,
    )

    # 5. Accumulate whole context clouds in that order
    accumulated_clouds: List[str] = []
    included_bubble_ids: set[str] = set()

    for item in traversed_items:
        b = item["bubble"]
        cloud = get_context_cloud_by_bubble_id(conn, b.id)
        if not cloud:
            continue

        cloud_str = format_cloud_context(cloud.raw_prompt, cloud.raw_response)
        test_context = "\n\n".join(accumulated_clouds + [cloud_str])
        cost = estimate_tokens(test_context)

        if cost <= max_tokens:
            accumulated_clouds.append(cloud_str)
            included_bubble_ids.add(b.id)
        elif not accumulated_clouds:
            # If even the first cloud exceeds budget, truncate it by words
            truncated = truncate_text_by_words(cloud_str, max_tokens)
            accumulated_clouds.append(truncated)
            included_bubble_ids.add(b.id)
            break
        else:
            break

    # 6. Add bubble records to debug_trace in visit order
    for b_id in visit_order:
        item = visited[b_id]
        b = item["bubble"]
        debug_trace.append(
            {
                "type": "bubble",
                "id": b.id,
                "cluster_id": b.cluster_id,
                "similarity": round(float(item["similarity"]), 4),
                "node_type": b.node_type,
                "depth": item["depth"],
                "included": b.id in included_bubble_ids,
            }
        )

    final_context = "\n\n".join(accumulated_clouds)
    tokens_used = estimate_tokens(final_context) if final_context else 0

    return {
        "mode": "graph",
        "context": final_context,
        "tokens_used": tokens_used,
        "debug_trace": debug_trace,
    }


# Ingestion, Promotion & Archiving Engine
def save_memory(conn, prompt: str, response: str, pin_as_core: bool = False) -> Dict[str, Any]:
    with conn:
        active_bubbles = get_all_active_bubbles(conn)
        active_corpus = [get_bubble_search_text(b) for b in active_bubbles]

        # 1. Deterministic top 8 keywords and extractive summary
        keywords = extract_keywords(f"{prompt} {response}", active_corpus, count=config.KEYWORD_COUNT)
        summary = extract_extractive_summary(prompt, response)
        new_bubble_text = f"{summary} {' '.join(keywords)}".strip()

        # 2. Fit shared vectorizer on active bubbles + new bubble
        vec = fit_shared_vectorizer(active_corpus + [new_bubble_text])

        # 3. Find matching cluster in shared TF-IDF space
        clusters = get_all_clusters(conn)
        best_cluster: Optional[ClusterRecord] = None
        best_sim = -1.0

        for c in clusters:
            members = [b for b in active_bubbles if b.cluster_id == c.id]
            c_text = get_cluster_search_text(c, members)
            sim = compute_similarity(vec, new_bubble_text, c_text)
            if sim > best_sim:
                best_sim = sim
                best_cluster = c

        if best_sim >= config.CLUSTER_ATTACH_THRESHOLD and best_cluster is not None:
            cluster_id = best_cluster.id
            recomputed_summary = ", ".join(keywords[:5]) if keywords else best_cluster.summary
            update_cluster_summary(conn, cluster_id, recomputed_summary)
        else:
            # Create new cluster named from top 2 keywords
            cluster_id = str(uuid.uuid4())
            name_words = keywords[: config.CLUSTER_NAME_KEYWORD_COUNT]
            cluster_name = " ".join(name_words).title() if name_words else "General Memory"
            cluster_summary = ", ".join(keywords[:5]) if keywords else "Cluster summary"
            insert_cluster(conn, cluster_id, cluster_name, cluster_summary)

        # 4. Insert Bubble and Context Cloud
        bubble_id = str(uuid.uuid4())
        cloud_id = str(uuid.uuid4())
        node_type = "CORE" if pin_as_core else "STATE"
        full_text = f"User: {prompt}\nAssistant: {response}"
        token_estimate = estimate_tokens(full_text)

        insert_bubble(
            conn,
            bubble_id=bubble_id,
            cluster_id=cluster_id,
            keywords=keywords,
            summary=summary,
            node_type=node_type,
            status="active",
            incoming_links=0,
        )

        insert_context_cloud(
            conn,
            cloud_id=cloud_id,
            bubble_id=bubble_id,
            raw_prompt=prompt,
            raw_response=response,
            full_text=full_text,
            token_estimate=token_estimate,
        )

        # 5. Compare new bubble to every other active bubble using single fit
        archived_node_ids: List[str] = []

        for older in active_bubbles:
            older_text = get_bubble_search_text(older)
            sim = compute_similarity(vec, new_bubble_text, older_text)

            if sim > config.SUPERSEDE_THRESHOLD:
                if older.node_type == "STATE":
                    # Supersede rule: set older STATE bubble to archived
                    set_bubble_status(conn, older.id, "archived")
                    archived_node_ids.append(older.id)
                else:
                    # Never archive CORE: create edge and increment incoming_links
                    insert_bubble_link(conn, bubble_id, older.id, sim)
                    increment_incoming_links(conn, older.id)
            elif sim >= config.LINK_SIMILARITY_MIN:
                # Link rule: insert edge (source = new, target = older) and increment incoming_links
                insert_bubble_link(conn, bubble_id, older.id, sim)
                increment_incoming_links(conn, older.id)

        # 6. Promotion rule: active STATE bubbles with incoming_links >= 5 become CORE
        promoted_node_ids: List[str] = []
        eligible_bubbles = get_active_state_bubbles_with_min_incoming_links(
            conn, min_links=config.PROMOTION_INCOMING_LINKS
        )
        for b in eligible_bubbles:
            set_bubble_node_type(conn, b.id, "CORE")
            promoted_node_ids.append(b.id)

        return {
            "bubble_id": bubble_id,
            "cluster_id": cluster_id,
            "node_type": node_type,
            "archived_node_ids": archived_node_ids,
            "promoted_node_ids": promoted_node_ids,
        }


def archive_last_memory(conn) -> Dict[str, Any]:
    with conn:
        archived_id = archive_most_recent_active_state_bubble(conn)
        if archived_id:
            return {
                "archived_bubble_id": archived_id,
                "message": "Successfully archived most recent active STATE bubble.",
            }
        return {
            "archived_bubble_id": None,
            "message": "No active STATE bubble found to archive.",
        }
