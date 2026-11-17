import math
import re
import threading
import uuid
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Tuple

from fastembed import TextEmbedding
import numpy as np

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

# Standard English stop words for lightweight human-readable keyword extraction
ENGLISH_STOP_WORDS = frozenset(
    [
        "a", "about", "above", "across", "after", "afterwards", "again", "against", "all",
        "almost", "alone", "along", "already", "also", "although", "always", "am", "among",
        "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone",
        "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be",
        "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand",
        "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both",
        "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con", "could",
        "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during",
        "each", "eg", "eight", "either", "eleven", "else", "elsewhere", "empty", "enough",
        "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few",
        "fifteen", "fifty", "fill", "find", "fire", "first", "five", "for", "former",
        "formerly", "forty", "found", "four", "from", "front", "full", "further", "get",
        "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here",
        "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
        "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed", "interest",
        "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least",
        "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine",
        "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name",
        "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none",
        "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
        "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours",
        "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put",
        "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious",
        "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so",
        "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still",
        "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves",
        "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon",
        "these", "they", "thick", "thin", "third", "this", "those", "though", "three",
        "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward",
        "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
        "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence",
        "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon",
        "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom",
        "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your",
        "yours", "yourself", "yourselves",
    ]
)

# Lazy singleton model management
_model: Optional[TextEmbedding] = None
_model_lock = threading.Lock()


def _get_model() -> TextEmbedding:
    """Lazy singleton accessor for TextEmbedding('BAAI/bge-small-en-v1.5').

    Thread-safe and initialized on first use to ensure fast, offline-safe module imports.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of strings into L2-normalized float32 vectors of shape (len(texts), 384).

    Returns shape (0, 384) for an empty list without invoking the model.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    model = _get_model()
    raw_embeddings = list(model.embed(texts))
    arr = np.array(raw_embeddings, dtype=np.float32)

    # L2-normalize each row with safe zero-norm handling
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, 1e-12)
    normalized = arr / safe_norms
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def compute_similarity(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts using normalized bge-small embeddings."""
    embs = embed_texts([text_a, text_b])
    return float(np.dot(embs[0], embs[1]))


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
    """Extract top keywords from prompt + response:

    Stop-word-filtered term frequency, breaking ties by IDF, then alphabetically.
    Used purely for human-readable cluster names and metadata.
    """
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
    query_emb = embed_texts([query])[0]

    # 1. Score query against clusters and select top 2
    clusters = get_all_clusters(conn)
    cluster_scores: List[Tuple[float, ClusterRecord, List[BubbleRecord]]] = []

    if clusters:
        cluster_data = [
            (c, [b for b in active_bubbles if b.cluster_id == c.id])
            for c in clusters
        ]
        cluster_texts = [get_cluster_search_text(c, members) for c, members in cluster_data]
        cluster_embs = embed_texts(cluster_texts)
        sims = cluster_embs @ query_emb

        for idx, (c, members) in enumerate(cluster_data):
            cluster_scores.append((float(sims[idx]), c, members))

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
    if candidate_bubbles:
        candidate_texts = [get_bubble_search_text(b) for b in candidate_bubbles]
        candidate_embs = embed_texts(candidate_texts)
        b_sims = candidate_embs @ query_emb

        for idx, b in enumerate(candidate_bubbles):
            bubble_scores.append((float(b_sims[idx]), b))

    bubble_scores.sort(key=lambda x: x[0], reverse=True)

    # Top 2 as seeds; fallback to newest active bubbles if all similarities are 0
    all_zero = not bubble_scores or all(s[0] == 0.0 for s in bubble_scores)
    if all_zero:
        seeds = get_recent_active_bubbles(conn, limit=config.GRAPH_SEEDS_PER_CLUSTER)
    else:
        seeds = [b for _, b in bubble_scores[: config.GRAPH_SEEDS_PER_CLUSTER]]

    # 3. Traverse bubble_links as undirected graph up to max_depth = 2, weight >= GRAPH_EDGE_MIN_WEIGHT
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

        visited[curr_id] = {
            "bubble": curr_bubble,
            "depth": depth,
            "similarity": 0.0,
        }
        visit_order.append(curr_id)

        if depth < config.GRAPH_TRAVERSAL_MAX_DEPTH:
            neighbors = get_undirected_neighbors(conn, curr_id, min_weight=config.GRAPH_EDGE_MIN_WEIGHT)
            for nbr_id in neighbors:
                if nbr_id not in visited:
                    queue.append((nbr_id, depth + 1))

    # Batch compute similarity for all traversed bubbles
    if visit_order:
        traversed_texts = [get_bubble_search_text(visited[bid]["bubble"]) for bid in visit_order]
        traversed_embs = embed_texts(traversed_texts)
        traversed_sims = traversed_embs @ query_emb
        for idx, bid in enumerate(visit_order):
            visited[bid]["similarity"] = float(traversed_sims[idx])

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

        # 2. Embed new bubble text once
        new_bubble_emb = embed_texts([new_bubble_text])[0]

        # 3. Find matching cluster in shared embedding space
        clusters = get_all_clusters(conn)
        best_cluster: Optional[ClusterRecord] = None
        best_sim = -1.0

        if clusters:
            cluster_texts = [
                get_cluster_search_text(c, [b for b in active_bubbles if b.cluster_id == c.id])
                for c in clusters
            ]
            cluster_embs = embed_texts(cluster_texts)
            cluster_sims = cluster_embs @ new_bubble_emb

            best_idx = int(np.argmax(cluster_sims))
            best_sim = float(cluster_sims[best_idx])
            best_cluster = clusters[best_idx]

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

        # 5. Compare new bubble to every other active bubble using a single batch embedding call
        archived_node_ids: List[str] = []

        if active_bubbles:
            older_texts = [get_bubble_search_text(b) for b in active_bubbles]
            older_embs = embed_texts(older_texts)
            sims = older_embs @ new_bubble_emb

            for idx, older in enumerate(active_bubbles):
                sim = float(sims[idx])

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
