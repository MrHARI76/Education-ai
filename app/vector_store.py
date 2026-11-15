"""Qdrant vector database: store chunk vectors and find similar ones.

All documents share one collection. Each point (= one chunk) carries a
payload with its doc_id and source info, and every search filters by
doc_id so results never mix between documents.
"""

import logging
import time
import uuid
from typing import Any

from qdrant_client import QdrantClient, models

from app.config import settings
from app.embeddings import EMBED_DIM

logger = logging.getLogger(__name__)

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """Create the Qdrant client once. ':memory:' runs an in-process Qdrant (for tests)."""
    global _client
    if _client is None:
        if settings.qdrant_url == ":memory:":
            _client = QdrantClient(location=":memory:")
        else:
            _client = QdrantClient(url=settings.qdrant_url, timeout=30)
    return _client


def ensure_collection(max_wait_seconds: float = 30.0) -> None:
    """Create the collection and doc_id index if missing.

    Retries for up to ~30 s because in Docker Compose Qdrant may still be
    starting when the app boots.
    """
    client = get_client()
    deadline = time.monotonic() + max_wait_seconds
    while True:
        try:
            if not client.collection_exists(settings.qdrant_collection):
                client.create_collection(
                    settings.qdrant_collection,
                    vectors_config=models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE),
                )
                # An index on doc_id makes the per-document filter fast.
                client.create_payload_index(
                    settings.qdrant_collection,
                    field_name="doc_id",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                logger.info("Created Qdrant collection %s", settings.qdrant_collection)
            return
        except Exception as exc:
            if time.monotonic() > deadline:
                raise
            logger.warning("Qdrant not ready yet (%s); retrying in 2 s", exc)
            time.sleep(2)


def is_healthy() -> bool:
    """True if Qdrant answers a simple request."""
    try:
        get_client().get_collections()
        return True
    except Exception:
        return False


def _doc_filter(doc_id: str) -> models.Filter:
    """Filter that matches only points belonging to one document."""
    return models.Filter(
        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
    )


def upsert_chunks(doc_id: str, chunks: list[dict[str, Any]], vectors: list[list[float]]) -> None:
    """Store each chunk with its vector. 'Upsert' = insert, or update if it exists."""
    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "doc_id": doc_id,
                "text": chunk["text"],
                "source": chunk["source"],
                "file_type": chunk["file_type"],
                "location": chunk["location"],
                "heading": chunk["heading"],
                "chunk_index": chunk["chunk_index"],
            },
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    # Send in batches so a huge document doesn't create one giant request.
    for start in range(0, len(points), 256):
        get_client().upsert(settings.qdrant_collection, points=points[start : start + 256])


def search(doc_id: str, vector: list[float], limit: int = 20) -> list[dict[str, Any]]:
    """Return the payloads of the `limit` chunks most similar to `vector`."""
    result = get_client().query_points(
        settings.qdrant_collection,
        query=vector,
        query_filter=_doc_filter(doc_id),
        limit=limit,
        with_payload=True,
    )
    return [point.payload or {} for point in result.points]


def delete_document(doc_id: str) -> None:
    """Remove every point that belongs to a document."""
    get_client().delete(
        settings.qdrant_collection,
        points_selector=models.FilterSelector(filter=_doc_filter(doc_id)),
    )
