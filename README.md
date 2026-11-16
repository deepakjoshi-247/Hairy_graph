# HairyGraph

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
[![Tests](https://img.shields.io/github/actions/workflow/status/<owner>/<repo>/tests.yml?branch=main&label=tests)](https://github.com/<owner>/<repo>/actions/workflows/tests.yml)

HairyGraph is a lightweight, local-first hierarchical graph memory service built with FastAPI and SQLite. It provides persistent semantic retrieval for AI agents and LLM applications at `http://127.0.0.1:8000` with zero external AI/LLM API calls and zero cloud dependencies at runtime.

All clustering, lexical extraction, and graph traversals execute locally using Python and scikit-learn (TF-IDF + cosine similarity). HairyGraph is a retrieval-only service: it stores, indexes, associates, and returns context within strict token budgets without generating text.

For detailed technical specifications, internal pipelines, and trade-off analyses, refer to the [System Architecture Documentation](docs/ARCHITECTURE.md). For an integration walkthrough showing pre-inference context retrieval and post-inference ingestion, see the [Local Agent Client Example](examples/local_agent_demo.py). For setup instructions and contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/<owner>/<repo>.git && cd <repo>

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install
pip install -e .

# 4. Run
uvicorn hairy_graph.main:app --host 127.0.0.1 --port 8000
```

To configure a custom database path:
```bash
export MEMORY_DB_PATH="/path/to/custom_memory.db"
# Windows PowerShell:
$env:MEMORY_DB_PATH = "C:\path\to\custom_memory.db"
```

---

## Directory Structure

```text
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── workflows/
│       └── tests.yml
├── docs/
│   └── ARCHITECTURE.md
├── examples/
│   └── local_agent_demo.py
├── hairy_graph/
│   ├── __init__.py
│   ├── config.py
│   ├── database.py
│   ├── graph_engine.py
│   └── main.py
├── tests/
│   └── test_memory.py
├── .env.example
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements.txt
```

---

## Core Memory Model

1. **Clusters**: Top-level thematic groups representing broad knowledge domains.
2. **Bubbles**: Semantic nodes within clusters storing extracted keywords, an extractive 2-sentence summary, node type (`CORE` vs `STATE`), and status (`active` vs `archived`).
3. **Context Clouds**: Exact 1:1 text storage for bubbles containing the raw prompt and response pair.
4. **Bubble Links**: Weighted graph edges ($0.0 \le \text{weight} \le 1.0$) connecting semantically related bubbles across or within clusters.

### Operational Modes
- **Cold Start Mode (< 10 active bubbles)**: Graph traversal is skipped. Context is populated from the 5 most recent active context clouds, budgeted newest-first, and delivered in chronological order (oldest to newest).
- **Hierarchical Graph Mode (>= 10 active bubbles)**: Queries route through top 2 matching clusters, identify seed bubbles, traverse undirected links up to depth 2 (weight $\ge 0.35$), sort traversed nodes giving priority to `CORE` nodes, and pack context strictly within `max_tokens`.
- **Active Archiving**: Archiving nodes via superseding or `/archive_last` can reduce the active count below 10, returning HairyGraph to cold start mode.

### Lifecycle Rules
- **Link Rule**: Cosine similarity in $[0.35, 0.80]$ creates an undirected edge and increments `incoming_links` on older bubbles.
- **Supersede Rule**: Cosine similarity $> 0.80$ against an older `STATE` bubble marks it as `archived`.
- **CORE Protection**: A `CORE` bubble is never archived by duplicate or high-similarity memories; edges are linked and knowledge remains intact.
- **Auto-Promotion**: Any active `STATE` bubble accumulating $\ge 5$ incoming links transitions to `CORE`.

---

## API Reference

### `POST /get_context`
Retrieve relevant memory context formatted as clean chat history within a strict token budget.

**Request:**
```json
{
  "query": "quantum computing and qubit superposition",
  "max_tokens": 1500
}
```

**Response:**
```json
{
  "mode": "graph",
  "context": "User: Explain qubits.\nAssistant: Superposition allows qubits to represent multiple states simultaneously.",
  "tokens_used": 22,
  "debug_trace": [
    {
      "type": "cluster",
      "id": "c-101",
      "cluster_id": "c-101",
      "similarity": 0.6521,
      "node_type": null,
      "depth": 0,
      "included": true
    },
    {
      "type": "bubble",
      "id": "b-202",
      "cluster_id": "c-101",
      "similarity": 0.7812,
      "node_type": "CORE",
      "depth": 0,
      "included": true
    }
  ]
}
```

### `POST /save_memory`
Store a prompt and response pair into the hierarchical memory graph.

**Request:**
```json
{
  "prompt": "Our database must use SQLite with WAL mode enabled.",
  "response": "Understood. SQLite configured with WAL mode and foreign keys enabled.",
  "pin_as_core": true
}
```

**Response:**
```json
{
  "bubble_id": "8f1a4e2b-...",
  "cluster_id": "3c9d7e1a-...",
  "node_type": "CORE",
  "archived_node_ids": [],
  "promoted_node_ids": []
}
```

### `POST /archive_last`
Archive the single most recently created active `STATE` bubble (`CORE` bubbles are never modified).

**Response:**
```json
{
  "archived_bubble_id": "8f1a4e2b-...",
  "message": "Successfully archived most recent active STATE bubble."
}
```

### `GET /graph_stats`
Retrieve current counts of clusters, active core nodes, active state nodes, and archived bubbles.

**Response:**
```json
{
  "total_clusters": 4,
  "active_core": 3,
  "active_state": 12,
  "archived_bubbles": 2
}
```

### `GET /health`
Public health check endpoint. Always returns `{"status": "ok"}`.

---

## Security & Tunneling Notice

- **Host Binding**: Uvicorn binds to `127.0.0.1` by default.
- **Privacy**: Prompts and responses are never logged to console or stdout.
- **Authentication**: When `MEMORY_API_KEY` is configured, all endpoints except `/health` require an `X-API-Key` request header matching the configured key (evaluated via constant-time comparison).

### Cloudflare Quick Tunnel Warning

Exposing port 8000 using Cloudflare Quick Tunnel (`cloudflared tunnel --url http://127.0.0.1:8000`) creates an unauthenticated public HTTPS URL. Because CORS is permissive (`allow_origins=["*"]`), anyone who discovers the tunnel URL can read or modify your memory store. Always define `MEMORY_API_KEY` before opening a public tunnel.

---

## Verification & Testing

Execute the test suite from the repository root:

```bash
pytest -v
```

All tests execute against temporary SQLite instances and verify cold start behavior, graph transitions, semantic routing, cross-cluster bridging, strict token capping, supersede mechanics, auto-promotion, and API authentication.
