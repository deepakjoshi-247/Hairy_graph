import os

DEFAULT_DB_PATH: str = "hairy_graph.db"


def get_db_path() -> str:
    """Read the database path dynamically from environment at call/init time."""
    return os.environ.get("MEMORY_DB_PATH", DEFAULT_DB_PATH)


# Cold Start settings
COLD_START_THRESHOLD: int = 10
COLD_START_RECENT_LIMIT: int = 5

# Hierarchical Graph Retrieval settings (calibrated for BAAI/bge-small-en-v1.5 embeddings)
GRAPH_TOP_CLUSTERS: int = 2
GRAPH_SEEDS_PER_CLUSTER: int = 2
GRAPH_TRAVERSAL_MAX_DEPTH: int = 2
GRAPH_EDGE_MIN_WEIGHT: float = 0.65  # Minimum edge weight to traverse in graph mode

# Ingestion, Linking, Superseding & Promotion (calibrated for BAAI/bge-small-en-v1.5 embeddings)
CLUSTER_ATTACH_THRESHOLD: float = 0.60  # Minimum cosine similarity to attach to an existing cluster
LINK_SIMILARITY_MIN: float = 0.65  # Minimum cosine similarity to form an associative link edge
LINK_SIMILARITY_MAX: float = 0.88  # Upper bound of associative linking before supersede evaluation
SUPERSEDE_THRESHOLD: float = 0.88  # Cosine similarity above which older STATE bubbles are superseded
PROMOTION_INCOMING_LINKS: int = 5

# Token & Text constraints
TOKEN_WORD_RATIO: float = 0.75  # ceil(words / 0.75) for tokens, floor(max_tokens * 0.75) for words
KEYWORD_COUNT: int = 8
CLUSTER_NAME_KEYWORD_COUNT: int = 2
DEFAULT_MAX_TOKENS: int = 1500
MIN_MAX_TOKENS: int = 1
MAX_MAX_TOKENS: int = 8000
MAX_TEXT_LENGTH: int = 20000

# Security
API_KEY_ENV_VAR: str = "MEMORY_API_KEY"
