"""SQLite storage for document metadata and the explanation cache.

SQLite is a database stored in a single file (data/nexus.db) and is built
into Python, so no extra server is needed. Two tables:
  - documents:    one row per upload (status, progress, mind map JSON)
  - explanations: cached LLM explanations, so a second click is instant
"""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    path        TEXT NOT NULL,
    status      TEXT NOT NULL,
    progress    INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    mindmap     TEXT,          -- JSON string of models.MindMap
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS explanations (
    doc_id   TEXT NOT NULL,
    node_id  TEXT NOT NULL,
    level    TEXT NOT NULL,
    payload  TEXT NOT NULL,     -- JSON string of models.ExplainResponse
    PRIMARY KEY (doc_id, node_id, level)
);
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Open a connection, commit on success, and always close it.

    A fresh connection per call is simplest and safe across threads
    (background tasks run in other threads).
    """
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create the data folders and tables if they don't exist yet."""
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA)


def create_document(doc_id: str, filename: str, path: str) -> None:
    """Insert a new document row with status 'queued'."""
    now = datetime.now(timezone.utc).isoformat()
    with connect() as conn:
        conn.execute(
            "INSERT INTO documents (doc_id, filename, path, status, progress, created_at)"
            " VALUES (?, ?, ?, 'queued', 0, ?)",
            (doc_id, filename, path, now),
        )


def update_status(doc_id: str, status: str, progress: int, error: str | None = None) -> None:
    """Save the current pipeline stage so the frontend can show it."""
    with connect() as conn:
        conn.execute(
            "UPDATE documents SET status = ?, progress = ?, error = ? WHERE doc_id = ?",
            (status, progress, error, doc_id),
        )


def save_mindmap(doc_id: str, mindmap: dict[str, Any]) -> None:
    """Store the finished mind map as JSON text."""
    with connect() as conn:
        conn.execute(
            "UPDATE documents SET mindmap = ? WHERE doc_id = ?",
            (json.dumps(mindmap), doc_id),
        )


def get_document(doc_id: str) -> dict[str, Any] | None:
    """Return one document as a dict (mind map parsed back from JSON), or None."""
    with connect() as conn:
        row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    if row is None:
        return None
    doc = dict(row)
    doc["mindmap"] = json.loads(doc["mindmap"]) if doc["mindmap"] else None
    return doc


def list_documents(limit: int = 20) -> list[dict[str, Any]]:
    """Return the most recent documents, newest first."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT doc_id, filename, status, progress, created_at FROM documents"
            " ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_document(doc_id: str) -> None:
    """Remove a document row and all its cached explanations."""
    with connect() as conn:
        conn.execute("DELETE FROM explanations WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))


def get_cached_explanation(doc_id: str, node_id: str, level: str) -> dict[str, Any] | None:
    """Look up a previously generated explanation (None if not cached)."""
    with connect() as conn:
        row = conn.execute(
            "SELECT payload FROM explanations WHERE doc_id = ? AND node_id = ? AND level = ?",
            (doc_id, node_id, level),
        ).fetchone()
    return json.loads(row["payload"]) if row else None


def cache_explanation(doc_id: str, node_id: str, level: str, payload: dict[str, Any]) -> None:
    """Save an explanation; INSERT OR REPLACE overwrites an older one."""
    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO explanations (doc_id, node_id, level, payload)"
            " VALUES (?, ?, ?, ?)",
            (doc_id, node_id, level, json.dumps(payload)),
        )
