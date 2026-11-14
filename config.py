import os

DEFAULT_DB_PATH: str = "hairy_graph.db"


def get_db_path() -> str:
    """Read the database path dynamically from environment at call/init time."""
    return os.environ.get("MEMORY_DB_PATH", DEFAULT_DB_PATH)


# Cold Start settings
COLD_START_THRESHOLD: int = 10
COLD_START_RECENT_LIMIT: int = 5

# Hierarchical Graph Retrieval settings
GRAPH_TOP_CLUSTERS: int = 2
GRAPH_SEEDS_PER_CLUSTER: int = 2
GRAPH_TRAVERSAL_MAX_DEPTH: int = 2
GRAPH_EDGE_MIN_WEIGHT: float = 0.35

# Ingestion, Linking, Superseding & Promotion
CLUSTER_ATTACH_THRESHOLD: float = 0.25
LINK_SIMILARITY_MIN: float = 0.35
LINK_SIMILARITY_MAX: float = 0.80
SUPERSEDE_THRESHOLD: float = 0.80
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
