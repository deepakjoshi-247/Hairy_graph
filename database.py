import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Generator, List, Optional

from hairy_graph.config import get_db_path


def get_utc_now_iso() -> str:
    """Return ISO 8601 UTC timestamp with microsecond precision."""
    return datetime.now(timezone.utc).isoformat()


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Create and configure a SQLite connection with WAL mode and foreign keys enabled."""
    path = db_path or get_db_path()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn


@contextmanager
def get_db_context(db_path: Optional[str] = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for managing connection per request."""
    conn = get_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize database schema and indexes."""
    with get_connection(db_path) as conn:
        with conn:
            conn.executescript(
                """
                -- Clusters Table
                CREATE TABLE IF NOT EXISTS clusters (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL
                );

                -- Bubbles Table
                CREATE TABLE IF NOT EXISTS bubbles (
                    id TEXT PRIMARY KEY,
                    cluster_id TEXT NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
                    keywords TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    node_type TEXT NOT NULL CHECK (node_type IN ('CORE', 'STATE')),
                    status TEXT NOT NULL CHECK (status IN ('active', 'archived')),
                    incoming_links INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMP NOT NULL
                );

                -- Context Clouds Table
                CREATE TABLE IF NOT EXISTS context_clouds (
                    id TEXT PRIMARY KEY,
                    bubble_id TEXT NOT NULL UNIQUE REFERENCES bubbles(id) ON DELETE CASCADE,
                    raw_prompt TEXT NOT NULL,
                    raw_response TEXT NOT NULL,
                    full_text TEXT NOT NULL,
                    token_estimate INTEGER NOT NULL,
                    created_at TIMESTAMP NOT NULL
                );

                -- Bubble Links Table
                CREATE TABLE IF NOT EXISTS bubble_links (
                    source_bubble_id TEXT NOT NULL REFERENCES bubbles(id) ON DELETE CASCADE,
                    target_bubble_id TEXT NOT NULL REFERENCES bubbles(id) ON DELETE CASCADE,
                    weight REAL NOT NULL CHECK (weight >= 0.0 AND weight <= 1.0),
                    PRIMARY KEY (source_bubble_id, target_bubble_id)
                );

                -- Indexes on foreign keys and queries
                CREATE INDEX IF NOT EXISTS idx_bubbles_cluster_id ON bubbles(cluster_id);
                CREATE INDEX IF NOT EXISTS idx_context_clouds_bubble_id ON context_clouds(bubble_id);
                CREATE INDEX IF NOT EXISTS idx_bubble_links_source ON bubble_links(source_bubble_id);
                CREATE INDEX IF NOT EXISTS idx_bubble_links_target ON bubble_links(target_bubble_id);
                CREATE INDEX IF NOT EXISTS idx_bubbles_status_node_type ON bubbles(status, node_type);
                CREATE INDEX IF NOT EXISTS idx_bubbles_status_created ON bubbles(status, created_at DESC);
                """
            )


@dataclass
class ClusterRecord:
    id: str
    name: str
    summary: str
    created_at: str


@dataclass
class BubbleRecord:
    id: str
    cluster_id: str
    keywords: List[str]
    summary: str
    node_type: str
    status: str
    incoming_links: int
    created_at: str


@dataclass
class ContextCloudRecord:
    id: str
    bubble_id: str
    raw_prompt: str
    raw_response: str
    full_text: str
    token_estimate: int
    created_at: str


# Helper row mappers
def _row_to_cluster(row: sqlite3.Row) -> ClusterRecord:
    return ClusterRecord(
        id=row["id"],
        name=row["name"],
        summary=row["summary"],
        created_at=row["created_at"],
    )


def _row_to_bubble(row: sqlite3.Row) -> BubbleRecord:
    keywords = json.loads(row["keywords"]) if isinstance(row["keywords"], str) else row["keywords"]
    return BubbleRecord(
        id=row["id"],
        cluster_id=row["cluster_id"],
        keywords=keywords,
        summary=row["summary"],
        node_type=row["node_type"],
        status=row["status"],
        incoming_links=row["incoming_links"],
        created_at=row["created_at"],
    )


def _row_to_context_cloud(row: sqlite3.Row) -> ContextCloudRecord:
    return ContextCloudRecord(
        id=row["id"],
        bubble_id=row["bubble_id"],
        raw_prompt=row["raw_prompt"],
        raw_response=row["raw_response"],
        full_text=row["full_text"],
        token_estimate=row["token_estimate"],
        created_at=row["created_at"],
    )


# Database CRUD Operations
def count_active_bubbles(conn: sqlite3.Connection) -> int:
    cursor = conn.execute("SELECT COUNT(*) AS count FROM bubbles WHERE status = 'active'")
    row = cursor.fetchone()
    return row["count"] if row else 0


def get_recent_active_context_clouds(conn: sqlite3.Connection, limit: int = 5) -> List[ContextCloudRecord]:
    """Retrieve candidate context clouds for active bubbles ordered newest-first (created_at DESC, rowid DESC)."""
    cursor = conn.execute(
        """
        SELECT c.id, c.bubble_id, c.raw_prompt, c.raw_response, c.full_text, c.token_estimate, c.created_at
        FROM context_clouds c
        JOIN bubbles b ON c.bubble_id = b.id
        WHERE b.status = 'active'
        ORDER BY b.created_at DESC, b.rowid DESC
        LIMIT ?
        """,
        (limit,),
    )
    return [_row_to_context_cloud(row) for row in cursor.fetchall()]


def get_all_active_bubbles(conn: sqlite3.Connection) -> List[BubbleRecord]:
    cursor = conn.execute(
        """
        SELECT id, cluster_id, keywords, summary, node_type, status, incoming_links, created_at
        FROM bubbles
        WHERE status = 'active'
        ORDER BY created_at DESC, rowid DESC
        """
    )
    return [_row_to_bubble(row) for row in cursor.fetchall()]


def get_recent_active_bubbles(conn: sqlite3.Connection, limit: int = 2) -> List[BubbleRecord]:
    cursor = conn.execute(
        """
        SELECT id, cluster_id, keywords, summary, node_type, status, incoming_links, created_at
        FROM bubbles
        WHERE status = 'active'
        ORDER BY created_at DESC, rowid DESC
        LIMIT ?
        """,
        (limit,),
    )
    return [_row_to_bubble(row) for row in cursor.fetchall()]


def get_bubble_by_id(conn: sqlite3.Connection, bubble_id: str) -> Optional[BubbleRecord]:
    cursor = conn.execute(
        """
        SELECT id, cluster_id, keywords, summary, node_type, status, incoming_links, created_at
        FROM bubbles
        WHERE id = ?
        """,
        (bubble_id,),
    )
    row = cursor.fetchone()
    return _row_to_bubble(row) if row else None


def get_context_cloud_by_bubble_id(conn: sqlite3.Connection, bubble_id: str) -> Optional[ContextCloudRecord]:
    cursor = conn.execute(
        """
        SELECT id, bubble_id, raw_prompt, raw_response, full_text, token_estimate, created_at
        FROM context_clouds
        WHERE bubble_id = ?
        """,
        (bubble_id,),
    )
    row = cursor.fetchone()
    return _row_to_context_cloud(row) if row else None


def get_all_clusters(conn: sqlite3.Connection) -> List[ClusterRecord]:
    cursor = conn.execute(
        """
        SELECT id, name, summary, created_at
        FROM clusters
        ORDER BY created_at ASC, rowid ASC
        """
    )
    return [_row_to_cluster(row) for row in cursor.fetchall()]


def get_active_bubbles_by_cluster(conn: sqlite3.Connection, cluster_id: str) -> List[BubbleRecord]:
    cursor = conn.execute(
        """
        SELECT id, cluster_id, keywords, summary, node_type, status, incoming_links, created_at
        FROM bubbles
        WHERE cluster_id = ? AND status = 'active'
        ORDER BY created_at DESC, rowid DESC
        """,
        (cluster_id,),
    )
    return [_row_to_bubble(row) for row in cursor.fetchall()]


def get_undirected_neighbors(conn: sqlite3.Connection, bubble_id: str, min_weight: float = 0.35) -> List[str]:
    """Return IDs of neighboring active bubbles connected by an edge with weight >= min_weight."""
    cursor = conn.execute(
        """
        SELECT DISTINCT CASE 
            WHEN l.source_bubble_id = ? THEN l.target_bubble_id
            ELSE l.source_bubble_id
        END AS neighbor_id
        FROM bubble_links l
        JOIN bubbles b ON (
            CASE WHEN l.source_bubble_id = ? THEN l.target_bubble_id ELSE l.source_bubble_id END = b.id
        )
        WHERE (l.source_bubble_id = ? OR l.target_bubble_id = ?)
          AND l.weight >= ?
          AND b.status = 'active'
        """,
        (bubble_id, bubble_id, bubble_id, bubble_id, min_weight),
    )
    return [row["neighbor_id"] for row in cursor.fetchall()]


def insert_cluster(conn: sqlite3.Connection, cluster_id: str, name: str, summary: str, created_at: Optional[str] = None) -> None:
    ts = created_at or get_utc_now_iso()
    conn.execute(
        "INSERT INTO clusters (id, name, summary, created_at) VALUES (?, ?, ?, ?)",
        (cluster_id, name, summary, ts),
    )


def update_cluster_summary(conn: sqlite3.Connection, cluster_id: str, summary: str) -> None:
    conn.execute(
        "UPDATE clusters SET summary = ? WHERE id = ?",
        (summary, cluster_id),
    )


def insert_bubble(
    conn: sqlite3.Connection,
    bubble_id: str,
    cluster_id: str,
    keywords: List[str],
    summary: str,
    node_type: str,
    status: str = "active",
    incoming_links: int = 0,
    created_at: Optional[str] = None,
) -> None:
    ts = created_at or get_utc_now_iso()
    conn.execute(
        """
        INSERT INTO bubbles (id, cluster_id, keywords, summary, node_type, status, incoming_links, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (bubble_id, cluster_id, json.dumps(keywords), summary, node_type, status, incoming_links, ts),
    )


def insert_context_cloud(
    conn: sqlite3.Connection,
    cloud_id: str,
    bubble_id: str,
    raw_prompt: str,
    raw_response: str,
    full_text: str,
    token_estimate: int,
    created_at: Optional[str] = None,
) -> None:
    ts = created_at or get_utc_now_iso()
    conn.execute(
        """
        INSERT INTO context_clouds (id, bubble_id, raw_prompt, raw_response, full_text, token_estimate, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (cloud_id, bubble_id, raw_prompt, raw_response, full_text, token_estimate, ts),
    )


def insert_bubble_link(conn: sqlite3.Connection, source_id: str, target_id: str, weight: float) -> None:
    conn.execute(
        """
        INSERT INTO bubble_links (source_bubble_id, target_bubble_id, weight)
        VALUES (?, ?, ?)
        ON CONFLICT(source_bubble_id, target_bubble_id) DO UPDATE SET weight = excluded.weight
        """,
        (source_id, target_id, weight),
    )


def increment_incoming_links(conn: sqlite3.Connection, bubble_id: str) -> None:
    conn.execute(
        "UPDATE bubbles SET incoming_links = incoming_links + 1 WHERE id = ?",
        (bubble_id,),
    )


def set_bubble_status(conn: sqlite3.Connection, bubble_id: str, status: str) -> None:
    conn.execute(
        "UPDATE bubbles SET status = ? WHERE id = ?",
        (status, bubble_id),
    )


def set_bubble_node_type(conn: sqlite3.Connection, bubble_id: str, node_type: str) -> None:
    conn.execute(
        "UPDATE bubbles SET node_type = ? WHERE id = ?",
        (node_type, bubble_id),
    )


def get_active_state_bubbles_with_min_incoming_links(conn: sqlite3.Connection, min_links: int = 5) -> List[BubbleRecord]:
    cursor = conn.execute(
        """
        SELECT id, cluster_id, keywords, summary, node_type, status, incoming_links, created_at
        FROM bubbles
        WHERE status = 'active' AND node_type = 'STATE' AND incoming_links >= ?
        """,
        (min_links,),
    )
    return [_row_to_bubble(row) for row in cursor.fetchall()]


def archive_most_recent_active_state_bubble(conn: sqlite3.Connection) -> Optional[str]:
    """Find and archive the single most recently created active STATE bubble (tie-broken by rowid)."""
    cursor = conn.execute(
        """
        SELECT id
        FROM bubbles
        WHERE status = 'active' AND node_type = 'STATE'
        ORDER BY created_at DESC, rowid DESC
        LIMIT 1
        """
    )
    row = cursor.fetchone()
    if not row:
        return None
    bubble_id = row["id"]
    conn.execute("UPDATE bubbles SET status = 'archived' WHERE id = ?", (bubble_id,))
    return bubble_id


def get_graph_stats(conn: sqlite3.Connection) -> dict:
    total_clusters = conn.execute("SELECT COUNT(*) AS count FROM clusters").fetchone()["count"]
    active_core = conn.execute(
        "SELECT COUNT(*) AS count FROM bubbles WHERE status = 'active' AND node_type = 'CORE'"
    ).fetchone()["count"]
    active_state = conn.execute(
        "SELECT COUNT(*) AS count FROM bubbles WHERE status = 'active' AND node_type = 'STATE'"
    ).fetchone()["count"]
    archived_bubbles = conn.execute(
        "SELECT COUNT(*) AS count FROM bubbles WHERE status = 'archived'"
    ).fetchone()["count"]

    return {
        "total_clusters": total_clusters,
        "active_core": active_core,
        "active_state": active_state,
        "archived_bubbles": archived_bubbles,
    }
