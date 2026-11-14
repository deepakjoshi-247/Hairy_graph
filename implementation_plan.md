# HairyGraph Implementation Plan

## 1. Project Architecture & File Structure

All code and artifacts will reside strictly inside `hairy_graph/`.

```text
hairy_graph/
├── requirements.txt           # Dependencies: fastapi, uvicorn, scikit-learn, pydantic, pytest, httpx
├── config.py                  # Tunable thresholds, budget rules, dynamic DB path resolution
├── database.py                # SQLite connection manager (WAL mode, FKs), DDL migrations, transactions, CRUD
├── graph_engine.py            # TF-IDF, extractive summarization, keyword extraction, cold start & graph retrieval, linking & promotion
├── main.py                    # FastAPI application ("HairyGraph"), lifespan DB init, CORS, auth, endpoints
├── test_memory.py             # Deterministic pytest suite (8 core verification scenarios, isolated temp DBs)
├── README.md                  # System overview, quickstart, API documentation, Cloudflare tunnel guide & security advisory
└── implementation_plan.md     # Architectural plan and technical specification
```

---

## 2. SQLite DDL Schema & Indexes

SQLite will be configured with `PRAGMA foreign_keys = ON;` and `PRAGMA journal_mode = WAL;` on every connection.
All timestamp fields store ISO 8601 UTC strings with microsecond resolution (`YYYY-MM-DDTHH:MM:SS.ffffffZ`).
Tie-breaking for order ("latest/last/oldest") uses `created_at` combined with SQLite `rowid`.

```sql
-- 1. Clusters Table
CREATE TABLE IF NOT EXISTS clusters (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    summary TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- 2. Bubbles Table
CREATE TABLE IF NOT EXISTS bubbles (
    id TEXT PRIMARY KEY,
    cluster_id TEXT NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
    keywords TEXT NOT NULL,  -- JSON-encoded list of strings, e.g. ["fastapi", "sqlite"]
    summary TEXT NOT NULL,   -- Extractive, max 2 sentences
    node_type TEXT NOT NULL CHECK (node_type IN ('CORE', 'STATE')),
    status TEXT NOT NULL CHECK (status IN ('active', 'archived')),
    incoming_links INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

-- 3. Context Clouds Table (1:1 with Bubbles)
CREATE TABLE IF NOT EXISTS context_clouds (
    id TEXT PRIMARY KEY,
    bubble_id TEXT NOT NULL UNIQUE REFERENCES bubbles(id) ON DELETE CASCADE,
    raw_prompt TEXT NOT NULL,
    raw_response TEXT NOT NULL,
    full_text TEXT NOT NULL,
    token_estimate INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

-- 4. Bubble Links Table (Directed edges with weight 0.0 - 1.0)
CREATE TABLE IF NOT EXISTS bubble_links (
    source_bubble_id TEXT NOT NULL REFERENCES bubbles(id) ON DELETE CASCADE,
    target_bubble_id TEXT NOT NULL REFERENCES bubbles(id) ON DELETE CASCADE,
    weight REAL NOT NULL CHECK (weight >= 0.0 AND weight <= 1.0),
    PRIMARY KEY (source_bubble_id, target_bubble_id)
);

-- Indexes for performance and foreign keys
CREATE INDEX IF NOT EXISTS idx_bubbles_cluster_id ON bubbles(cluster_id);
CREATE INDEX IF NOT EXISTS idx_context_clouds_bubble_id ON context_clouds(bubble_id);
CREATE INDEX IF NOT EXISTS idx_bubble_links_source ON bubble_links(source_bubble_id);
CREATE INDEX IF NOT EXISTS idx_bubble_links_target ON bubble_links(target_bubble_id);
CREATE INDEX IF NOT EXISTS idx_bubbles_status_node_type ON bubbles(status, node_type);
CREATE INDEX IF NOT EXISTS idx_bubbles_status_created ON bubbles(status, created_at DESC, rowid DESC);
```

---

## 3. Configuration Constants (`config.py`)

All thresholds and limits are consolidated in `config.py` and can be overridden during tests.
The database path is resolved dynamically via a function (`get_db_path()`), reading `MEMORY_DB_PATH` from the environment on each connection creation.

```python
import os

DEFAULT_DB_PATH: str = "hairy_graph.db"

def get_db_path() -> str:
    """Dynamically read the DB path from environment or fall back to default."""
    return os.environ.get("MEMORY_DB_PATH", DEFAULT_DB_PATH)

# Cold Start
COLD_START_THRESHOLD: int = 10          # Switch from cold start to graph when active bubbles >= 10
COLD_START_RECENT_LIMIT: int = 5        # Number of recent context clouds to consider

# Graph Retrieval
GRAPH_TOP_CLUSTERS: int = 2             # Top clusters to select based on query similarity
GRAPH_SEEDS_PER_CLUSTER: int = 2        # Top active bubbles per cluster as seeds
GRAPH_TRAVERSAL_MAX_DEPTH: int = 2      # BFS traversal depth limit
GRAPH_EDGE_MIN_WEIGHT: float = 0.35     # Minimum edge weight to traverse

# Ingestion & Linking
CLUSTER_ATTACH_THRESHOLD: float = 0.25  # Min similarity to attach to an existing cluster
LINK_SIMILARITY_MIN: float = 0.35       # Min similarity to form an edge
LINK_SIMILARITY_MAX: float = 0.80       # Max similarity for standard edge linking
SUPERSEDE_THRESHOLD: float = 0.80       # Similarity threshold above which older STATE bubbles are superseded
PROMOTION_INCOMING_LINKS: int = 5       # Threshold of incoming_links for STATE -> CORE promotion

# Token & Text Limits
TOKEN_WORD_RATIO: float = 0.75          # ceil(words / 0.75) for token estimation; floor(max_tokens * 0.75) for word truncation
KEYWORD_COUNT: int = 8                  # Top keywords extracted from prompt + response
CLUSTER_NAME_KEYWORD_COUNT: int = 2     # Keywords used to name newly created clusters
DEFAULT_MAX_TOKENS: int = 1500
MIN_MAX_TOKENS: int = 1
MAX_MAX_TOKENS: int = 8000
MAX_TEXT_LENGTH: int = 20000

# Security
API_KEY_ENV_VAR: str = "MEMORY_API_KEY"
API_KEY_HEADER: str = "X-API-Key"
```

---

## 4. Graph Engine Algorithms (Pseudocode)

### 4.1. Shared Text & Vector Processing
```text
FUNCTION estimate_tokens(text: str) -> int:
    words = text.split()
    RETURN ceil(len(words) / 0.75)

FUNCTION truncate_to_tokens(text: str, max_tokens: int) -> str:
    max_words = floor(max_tokens * 0.75)
    words = text.split()
    IF len(words) <= max_words:
        RETURN text
    RETURN " ".join(words[:max_words])

FUNCTION extract_summary(prompt: str, response: str) -> str:
    // Extractive summary: first 2 sentences of response (fallback to prompt)
    source = response.strip() IF response.strip() ELSE prompt.strip()
    sentences = regex_split_sentences(source)
    RETURN " ".join(sentences[:2])

FUNCTION extract_keywords(text: str, existing_corpus: list[str]) -> list[str]:
    // Tokenize into alphanumeric words >= 2 chars, lowercase
    tokens = [t for t in tokenize(text.lower()) if t not in ENGLISH_STOP_WORDS]
    term_freq = count_frequencies(tokens)
    
    // Deterministic tie-breaking: calculate smooth IDF if corpus available, else 1.0
    scores = []
    FOR token, freq in term_freq.items():
        idf = compute_idf(token, existing_corpus)
        // Sort tuple: (-freq, -idf, token)
        scores.append((-freq, -idf, token))
    
    scores.sort()
    RETURN [item[2] for item in scores[:8]]

FUNCTION build_shared_vectorizer(corpus: list[str]):
    TRY:
        vec = TfidfVectorizer(stop_words='english')
        vec.fit(corpus)
        RETURN vec
    EXCEPT ValueError: // empty vocabulary or stop words only
        RETURN None

FUNCTION compute_similarity(vec, text_a: str, text_b: str) -> float:
    IF vec is None:
        RETURN 0.0
    TRY:
        v_a = vec.transform([text_a])
        v_b = vec.transform([text_b])
        sim = cosine_similarity(v_a, v_b)[0][0]
        RETURN float(sim)
    EXCEPT:
        RETURN 0.0
```

### 4.2. Cold Start Retrieval (`/get_context` with active bubbles < 10)
```text
ALGORITHM ColdStartRetrieval(query: str, max_tokens: int):
    active_clouds = DB.get_recent_active_context_clouds(limit=5)
    // Ordered newest first: created_at DESC, rowid DESC
    
    accumulated_clouds = []
    running_tokens = 0
    
    FOR cloud in active_clouds:
        cloud_text = "User: " + cloud.raw_prompt + "\nAssistant: " + cloud.raw_response
        
        // Estimate token cost if we add this cloud
        test_context = "\n\n".join(accumulated_clouds + [cloud_text])
        cost = estimate_tokens(test_context)
        
        IF cost <= max_tokens:
            accumulated_clouds.append(cloud_text)
        ELSE IF accumulated_clouds is EMPTY:
            // First (newest) cloud alone exceeds max_tokens -> truncate it by words
            truncated = truncate_to_tokens(cloud_text, max_tokens)
            accumulated_clouds.append(truncated)
            BREAK
        ELSE:
            // Cannot fit another cloud without exceeding budget
            BREAK
            
    // Output kept clouds oldest -> newest
    accumulated_clouds.reverse()
    final_context = "\n\n".join(accumulated_clouds)
    tokens_used = estimate_tokens(final_context)
    
    RETURN {
        "mode": "cold_start",
        "context": final_context,
        "tokens_used": tokens_used,
        "debug_trace": []
    }
```

### 4.3. Hierarchical Coarse-to-Fine Retrieval (`/get_context` with active bubbles ≥ 10)
```text
ALGORITHM HierarchicalGraphRetrieval(query: str, max_tokens: int):
    active_bubbles = DB.get_all_active_bubbles()
    clusters = DB.get_all_clusters()
    
    // Step 1: Fit vectorizer ONCE over all active bubble texts
    bubble_texts = [b.summary + " " + " ".join(b.keywords) for b in active_bubbles]
    vec = build_shared_vectorizer(bubble_texts)
    
    // Step 2: Score query against cluster texts
    cluster_scores = []
    FOR c in clusters:
        member_bubbles = [b for b in active_bubbles if b.cluster_id == c.id]
        c_text = c.name + " " + c.summary + " " + " ".join([b.summary for b in member_bubbles])
        sim = compute_similarity(vec, query, c_text)
        cluster_scores.append((sim, c))
    
    cluster_scores.sort(key=lambda x: x[0], reverse=True)
    top_clusters = cluster_scores[:2]
    
    debug_trace = []
    FOR sim, c in top_clusters:
        debug_trace.append({
            "type": "cluster",
            "id": c.id,
            "cluster_id": c.id,
            "similarity": round(sim, 4),
            "node_type": None,
            "depth": 0,
            "included": True
        })
        
    // Step 3: Select top 2 seeds from top clusters
    top_cluster_ids = {c.id for _, c in top_clusters}
    candidate_bubbles = [b for b in active_bubbles if b.cluster_id in top_cluster_ids]
    
    bubble_scores = []
    FOR b in candidate_bubbles:
        b_text = b.summary + " " + " ".join(b.keywords)
        sim = compute_similarity(vec, query, b_text)
        bubble_scores.append((sim, b))
    
    bubble_scores.sort(key=lambda x: x[0], reverse=True)
    
    // Seed selection (fall back to newest active if all similarities are 0)
    all_zero = len(bubble_scores) == 0 or all(s[0] == 0.0 for s in bubble_scores)
    IF all_zero:
        seeds = DB.get_recent_active_bubbles(limit=2)
    ELSE:
        seeds = [b for _, b in bubble_scores[:2]]
        
    // Step 4: Undirected BFS traversal up to max_depth = 2, weight >= 0.35
    visited = {}          // bubble_id -> {node, depth, sim}
    queue = [(seed.id, 0) for seed in seeds]
    
    WHILE queue is not EMPTY:
        curr_id, depth = queue.pop(0)
        IF curr_id in visited:
            CONTINUE
        
        curr_bubble = DB.get_bubble(curr_id)
        IF curr_bubble.status != 'active':
            CONTINUE
            
        b_text = curr_bubble.summary + " " + " ".join(curr_bubble.keywords)
        sim = compute_similarity(vec, query, b_text)
        visited[curr_id] = {"bubble": curr_bubble, "depth": depth, "sim": sim}
        
        IF depth < 2:
            neighbors = DB.get_undirected_neighbors(curr_id, min_weight=0.35)
            FOR neighbor_id in neighbors:
                IF neighbor_id not in visited:
                    queue.append((neighbor_id, depth + 1))
                    
    // Step 5: Order traversed bubbles: CORE first, then STATE; within each group, by similarity DESC
    traversed_nodes = list(visited.values())
    traversed_nodes.sort(key=lambda x: (1 if x["bubble"].node_type == 'CORE' else 0, x["sim"]), reverse=True)
    
    // Step 6: Accumulate context clouds
    accumulated_clouds = []
    included_ids = set()
    
    FOR item in traversed_nodes:
        b = item["bubble"]
        cloud = DB.get_context_cloud_by_bubble_id(b.id)
        cloud_text = "User: " + cloud.raw_prompt + "\nAssistant: " + cloud.raw_response
        
        test_context = "\n\n".join(accumulated_clouds + [cloud_text])
        cost = estimate_tokens(test_context)
        
        IF cost <= max_tokens:
            accumulated_clouds.append(cloud_text)
            included_ids.add(b.id)
        ELSE IF accumulated_clouds is EMPTY:
            // First cloud exceeds max_tokens -> truncate by words
            truncated = truncate_to_tokens(cloud_text, max_tokens)
            accumulated_clouds.append(truncated)
            included_ids.add(b.id)
            BREAK
        ELSE:
            BREAK
            
    // Append bubbles to debug_trace in visit order
    FOR b_id, item in visited.items():
        debug_trace.append({
            "type": "bubble",
            "id": b_id,
            "cluster_id": item["bubble"].cluster_id,
            "similarity": round(item["sim"], 4),
            "node_type": item["bubble"].node_type,
            "depth": item["depth"],
            "included": b_id in included_ids
        })
        
    final_context = "\n\n".join(accumulated_clouds)
    tokens_used = estimate_tokens(final_context)
    
    RETURN {
        "mode": "graph",
        "context": final_context,
        "tokens_used": tokens_used,
        "debug_trace": debug_trace
    }
```

### 4.4. Ingestion, Promotion & Archiving (`/save_memory`)
```text
ALGORITHM SaveMemory(prompt: str, response: str, pin_as_core: bool):
    BEGIN TRANSACTION
    
    active_bubbles = DB.get_all_active_bubbles()
    existing_corpus = [b.summary + " " + " ".join(b.keywords) for b in active_bubbles]
    
    // Step 1: Extract keywords and extractive summary
    keywords = extract_keywords(prompt + " " + response, existing_corpus)
    summary = extract_summary(prompt, response)
    new_bubble_text = summary + " " + " ".join(keywords)
    
    // Step 2: Fit vectorizer on active bubbles + new bubble
    vec = build_shared_vectorizer(existing_corpus + [new_bubble_text])
    
    // Step 3: Cluster attachment
    clusters = DB.get_all_clusters()
    best_cluster_id = None
    best_sim = -1.0
    
    FOR c in clusters:
        member_bubbles = [b for b in active_bubbles if b.cluster_id == c.id]
        c_text = c.name + " " + c.summary + " " + " ".join([b.summary for b in member_bubbles])
        sim = compute_similarity(vec, new_bubble_text, c_text)
        IF sim > best_sim:
            best_sim = sim
            best_cluster_id = c.id
            
    IF best_sim >= 0.25 and best_cluster_id is not None:
        cluster_id = best_cluster_id
        // Update cluster summary with recomputed keywords
        recomputed_summary = ", ".join(keywords[:5])
        DB.update_cluster_summary(cluster_id, recomputed_summary)
    ELSE:
        // Create new cluster named after top 2 keywords
        cluster_name = " ".join(keywords[:2]).title()
        cluster_id = generate_id()
        DB.create_cluster(id=cluster_id, name=cluster_name, summary=", ".join(keywords[:5]))
        
    // Step 4: Create Bubble and ContextCloud
    node_type = 'CORE' IF pin_as_core ELSE 'STATE'
    bubble_id = generate_id()
    cloud_id = generate_id()
    full_text = prompt + " " + response
    token_est = estimate_tokens(full_text)
    
    DB.insert_bubble(id=bubble_id, cluster_id=cluster_id, keywords=keywords,
                     summary=summary, node_type=node_type, status='active')
    DB.insert_context_cloud(id=cloud_id, bubble_id=bubble_id, raw_prompt=prompt,
                            raw_response=response, full_text=full_text, token_estimate=token_est)
                            
    // Step 5: Link, Supersede, and Promote against existing active bubbles
    archived_node_ids = []
    
    FOR older in active_bubbles:
        older_text = older.summary + " " + " ".join(older.keywords)
        sim = compute_similarity(vec, new_bubble_text, older_text)
        
        IF sim > 0.80:
            IF older.node_type == 'STATE':
                // Supersede rule: archive older STATE bubble
                DB.update_bubble_status(older.id, status='archived')
                archived_node_ids.append(older.id)
            ELSE:
                // CORE protection: never archive CORE; add edge and increment links
                DB.insert_link(source_id=bubble_id, target_id=older.id, weight=sim)
                DB.increment_incoming_links(older.id)
        ELSE IF sim >= 0.35:
            // Link rule: add edge and increment links
            DB.insert_link(source_id=bubble_id, target_id=older.id, weight=sim)
            DB.increment_incoming_links(older.id)
            
    // Step 6: Promotion rule
    promoted_node_ids = []
    eligible_for_promotion = DB.get_active_state_bubbles_with_min_incoming_links(5)
    FOR b in eligible_for_promotion:
        DB.update_bubble_node_type(b.id, node_type='CORE')
        promoted_node_ids.append(b.id)
        
    COMMIT TRANSACTION
    
    RETURN {
        "bubble_id": bubble_id,
        "cluster_id": cluster_id,
        "node_type": node_type,
        "archived_node_ids": archived_node_ids,
        "promoted_node_ids": promoted_node_ids
    }
```

### 4.5. Archive Last (`POST /archive_last`)
```text
ALGORITHM ArchiveLast():
    BEGIN TRANSACTION
    target = DB.query_one("""
        SELECT id FROM bubbles 
        WHERE status = 'active' AND node_type = 'STATE'
        ORDER BY created_at DESC, rowid DESC
        LIMIT 1
    """)
    
    IF target is None:
        COMMIT TRANSACTION
        RETURN {
            "archived_bubble_id": None,
            "message": "No active STATE bubble available to archive."
        }
        
    DB.update_bubble_status(target.id, status='archived')
    COMMIT TRANSACTION
    
    RETURN {
        "archived_bubble_id": target.id,
        "message": "Successfully archived most recent active STATE bubble."
    }
```

---

## 5. API Endpoints & Request/Response Models (`main.py`)

1. **`POST /get_context`**
   - Request: `{"query": str (1..20000 chars), "max_tokens": int (1..8000, default 1500)}`
   - Response: `{"mode": "cold_start" | "graph", "context": str, "tokens_used": int, "debug_trace": list[dict]}`
2. **`POST /save_memory`**
   - Request: `{"prompt": str (1..20000 chars), "response": str (0..20000 chars), "pin_as_core": bool (default False)}`
   - Response: `{"bubble_id": str, "cluster_id": str, "node_type": "CORE" | "STATE", "archived_node_ids": list[str], "promoted_node_ids": list[str]}`
3. **`POST /archive_last`**
   - Request: empty body
   - Response: `{"archived_bubble_id": str | None, "message": str}`
4. **`GET /health`**
   - Open even when auth is enabled. Response: `{"status": "ok"}`
5. **`GET /graph_stats`**
   - Response: `{"total_clusters": int, "active_core": int, "active_state": int, "archived_bubbles": int}`

### Security & Error Handling
- Default bind: `127.0.0.1:8000`.
- CORS: `allow_origins=["*"]`, `allow_credentials=False`.
- Optional Auth: If `MEMORY_API_KEY` is set in env, all routes except `/health` require `X-API-Key` matching `secrets.compare_digest`. Missing/invalid key returns `401 Unauthorized`.
- Validation errors return clean HTTP 422/400 JSON responses without leaking tracebacks.
- Prompts/responses are never logged in plaintext.

---

## 6. Test Matrix (`test_memory.py`)

All tests run against isolated SQLite temp databases initialized per test using pytest `tmp_path` and `monkeypatch.setenv("MEMORY_DB_PATH", ...)`, wrapped in `with TestClient(app) as client:`.

| # | Test Scenario | Description & Assertions |
|---|---------------|--------------------------|
| 1 | **Cold start mode (<10 active bubbles)** | Ingest 3-5 bubbles. Assert `mode == "cold_start"`, `debug_trace == []`, context ordered oldest -> newest. Assert `tokens_used <= max_tokens` on tiny budget (e.g., 5 tokens), verifying word truncation. |
| 2 | **Cold start to Graph transition** | Ingest 10 distinct memories -> Assert `mode == "graph"`. Use `/archive_last` to drop active count to 9 -> Assert query flips back to `mode == "cold_start"`. |
| 3 | **Graph routing & debug trace** | Seed two distinct topics (e.g. Astronomy vs Culinary Arts) with 10+ bubbles. Query with domain-specific terms. Assert query selects the astronomy cluster; verify `debug_trace` includes clusters, visited bubbles, similarity scores, depths, and node types. |
| 4 | **Cross-cluster bridging** | Ingest cluster A and cluster B nodes with an explicit bridging link ($0.35 \le weight \le 0.80$). Query cluster A -> Verify traversal follows edge to include the connected node from cluster B. |
| 5 | **Strict token cap enforcement** | Test `max_tokens` budgets: 1, 5, 20, 100, 1500 across cold start and graph modes. Verify `tokens_used <= max_tokens` across every single response without exception. |
| 6 | **Auto-promotion & CORE protection** | Ingest a STATE bubble S1. Ingest 5 subsequent bubbles linking to S1 ($0.35 \le sim \le 0.80$) -> verify S1 is promoted to `CORE` (`incoming_links >= 5`). Ingest a near-duplicate with $sim > 0.80$ -> verify S1 remains active `CORE`, is NOT archived, and edge is added. |
| 7 | **Supersede rule & archive undo** | Ingest STATE bubble S1. Ingest S2 with $sim > 0.80$ to S1 -> assert S1 is in `archived_node_ids`. Call `/archive_last` -> assert S2 is archived. Call `/archive_last` when no active STATE bubbles remain -> assert `archived_bubble_id is None`. |
| 8 | **Edge cases & Auth security** | Empty DB query returns cold start with empty context. Empty-vocabulary query (pure stop words) handles gracefully with 0.0 similarities. Bad inputs (empty string, token limits out of 1-8000) return 422. With `MEMORY_API_KEY` set: missing/bad key returns 401; `/health` remains accessible 200 OK. |

---

## 7. Assumptions & Open Questions

### Assumptions
1. **Extractive Summaries**: Pure regex splitting on punctuation (`[.!?]`) is robust and standard for local non-LLM sentence extraction without external dependencies.
2. **Transaction Isolation**: SQLite `BEGIN IMMEDIATE` or Python's `with conn:` provides complete all-or-nothing atomicity for the ingestion workflow.
3. **Link Directionality**: Edges are saved as `source = new_bubble_id, target = older_bubble_id`, but traversal in Graph Mode treats links as undirected to find semantic neighborhoods.
4. **Deterministic Keyword Sorting**: When word frequencies tie, ties are broken by IDF score descending, then alphabetically ascending, guaranteeing 100% deterministic outputs even on an empty corpus.

### Open Questions
- None. All requirements and edge cases are clearly defined and accounted for.
