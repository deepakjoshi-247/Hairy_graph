# HairyGraph

**HairyGraph** is a lightweight, local hierarchical graph memory service built with **FastAPI** and **SQLite**. It runs entirely locally at `http://127.0.0.1:8000` with **no external AI/LLM services**, API keys, or runtime network calls.

All clustering, semantic matching, and graph traversals are powered by pure Python and `scikit-learn` using TF-IDF and cosine similarity in a shared vector space. HairyGraph is a **retrieval-only** service: it stores, organizes, links, and returns text context without generating text.

---

## Architecture Overview

```text
               +--------------------------------------+
               |          Query (/get_context)        |
               +-------------------+------------------+
                                   |
                  Active Bubbles >= 10?
                 /                     \
             [YES]                     [NO]
               |                         |
       +-------v-------+         +-------v-------+
       | TF-IDF Scoring|         |   Cold Start  |
       |  Top Clusters |         | Sliding Window|
       +-------+-------+         |   (max 5)     |
               |                 +-------+-------+
       +-------v-------+                 |
       | Seed Selection|                 |
       |  Top Bubbles  |                 |
       +-------+-------+                 |
               |                         |
       +-------v-------+                 |
       |  Undirected   |                 |
       |  BFS Traversal|                 |
       |  (Depth <= 2) |                 |
       +-------+-------+                 |
               |                         |
       +-------v-------+                 |
       | CORE Priority |                 |
       | Token Budget  |                 |
       +-------+-------+                 |
               |                         |
               +-----------+-------------+
                           |
               +-----------v-----------+
               |  Context History Text |
               +-----------------------+
```

### Core Memory Hierarchy
1. **Clusters**: Top-level semantic buckets representing broad topics (e.g. Astronomy, Cooking, Kubernetes).
2. **Bubbles**: Nodes within clusters storing extracted keywords, an extractive 2-sentence summary, node type (`CORE` vs `STATE`), and active/archived status.
3. **Context Clouds**: 1:1 text storage for bubbles containing the exact `User:` prompt and `Assistant:` response.
4. **Bubble Links**: Weighted graph edges ($0.0 \le \text{weight} \le 1.0$) connecting bubbles across or within clusters.

### Key Retrieval Modes
- **Cold Start (Day 1 Mode)**: When the active bubble count is $< 10$, graph traversal is skipped. Context is formed from up to 5 most recent active clouds, budgeted newest-first and delivered oldest $\rightarrow$ newest.
  > **Note**: Archiving can drop the active bubble count below 10 and flip HairyGraph back to cold start. This behavior is intentional.
- **Hierarchical Graph Retrieval**: When active bubbles $\ge 10$, queries route through coarse cluster scoring, identify seed bubbles, traverse undirected links (up to depth 2, weight $\ge 0.35$), sort traversed nodes giving priority to `CORE` nodes, and pack context strictly within `max_tokens`.

### Ingestion & Graph Dynamics
- **Deterministic Keywords**: Extracted using stop-word-filtered term frequency, tie-broken by IDF and alphabetically.
- **Link Rule**: Cosine similarity in $[0.35, 0.80]$ creates an undirected edge and increments `incoming_links` on older bubbles.
- **Supersede Rule**: Cosine similarity $> 0.80$ against an older `STATE` bubble marks it as `archived`.
- **CORE Protection**: A `CORE` bubble is **never** archived by duplicate or high-similarity memories; edges are linked and knowledge remains intact.
- **Auto-Promotion**: Any active `STATE` bubble that accumulates $\ge 5$ incoming links is automatically promoted to `CORE`.

---

## Installation & Setup

### Prerequisites
- Python 3.10+

### Setup Virtual Environment
```bash
cd hairy_graph
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Server
```bash
uvicorn hairy_graph.main:app --host 127.0.0.1 --port 8000
```

The database defaults to `hairy_graph.db`. To change the database location:
```bash
export MEMORY_DB_PATH="/custom/path/memory.db"
# Windows PowerShell:
$env:MEMORY_DB_PATH = "C:\custom\path\memory.db"
```

---

## API Reference

### 1. `POST /get_context`
Retrieve relevant memory context formatted as clean chat history within a strict token budget.

**Request:**
```json
{
  "query": "quantum entanglement and qubits",
  "max_tokens": 1500
}
```

**Response:**
```json
{
  "mode": "graph",
  "context": "User: What is quantum entanglement?\nAssistant: Entanglement links two particles instantaneously regardless of distance.\n\nUser: Explain qubits.\nAssistant: Qubits exploit superposition to represent multiple states simultaneously.",
  "tokens_used": 34,
  "debug_trace": [
    {
      "type": "cluster",
      "id": "c71a34...",
      "cluster_id": "c71a34...",
      "similarity": 0.6521,
      "node_type": null,
      "depth": 0,
      "included": true
    },
    {
      "type": "bubble",
      "id": "b901fc...",
      "cluster_id": "c71a34...",
      "similarity": 0.7812,
      "node_type": "CORE",
      "depth": 0,
      "included": true
    }
  ]
}
```

### 2. `POST /save_memory`
Store a prompt and response pair into the hierarchical memory graph.

**Request:**
```json
{
  "prompt": "How do you calculate sourdough hydration?",
  "response": "Hydration is the total water weight divided by flour weight expressed as a percentage.",
  "pin_as_core": false
}
```

**Response:**
```json
{
  "bubble_id": "e4a2bc01-...",
  "cluster_id": "f5b3cd12-...",
  "node_type": "STATE",
  "archived_node_ids": [],
  "promoted_node_ids": []
}
```

### 3. `POST /archive_last`
Archive the single most recently created active `STATE` bubble (CORE bubbles are never touched).

**Response:**
```json
{
  "archived_bubble_id": "e4a2bc01-...",
  "message": "Successfully archived most recent active STATE bubble."
}
```

### 4. `GET /graph_stats`
Get current counts of clusters, active core nodes, active state nodes, and archived bubbles.

**Response:**
```json
{
  "total_clusters": 3,
  "active_core": 2,
  "active_state": 11,
  "archived_bubbles": 1
}
```

### 5. `GET /health`
Public health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

---

## Security & Cloudflare Quick Tunnel Guide

### Security Defaults
- The server binds strictly to `127.0.0.1` by default.
- Prompts and responses are never logged in plaintext.
- **Optional Authentication**: If the environment variable `MEMORY_API_KEY` is set, all endpoints except `/health` require the `X-API-Key` request header (verified with constant-time `secrets.compare_digest`).

### Cloudflare Quick Tunnel Setup & Security Warning

> [!WARNING]
> **CRITICAL SECURITY ADVISORY**
> Exposing port 8000 to the public internet using a Cloudflare Quick Tunnel (`cloudflared tunnel --url http://localhost:8000`) assigns an unauthenticated public HTTPS URL to your service.
> Because HairyGraph enables CORS (`allow_origins=["*"]`), **anyone with this URL can read and modify your memory store without restriction**.
>
> **ALWAYS set `MEMORY_API_KEY` before opening a public tunnel:**
> ```bash
> export MEMORY_API_KEY="your-strong-random-secret"
> # Windows PowerShell:
> $env:MEMORY_API_KEY = "your-strong-random-secret"
> ```

To expose HairyGraph securely via Cloudflare Quick Tunnel:
1. Set `MEMORY_API_KEY`:
   ```bash
   export MEMORY_API_KEY="my-secure-api-key"
   ```
2. Start HairyGraph:
   ```bash
   uvicorn hairy_graph.main:app --host 127.0.0.1 --port 8000
   ```
3. In a separate terminal, launch the tunnel:
   ```bash
   cloudflared tunnel --url http://127.0.0.1:8000
   ```
4. Configure clients with the `X-API-Key` header matching your secret key.

---

## Running Tests

Run the complete test suite:
```bash
pytest hairy_graph/test_memory.py -v
```
All tests use temporary SQLite databases isolated per test function without mutating real data.
