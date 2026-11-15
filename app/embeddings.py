"""Embedding and reranking models (both run locally on the CPU).

- The embedder turns text into a list of 384 numbers (a "vector"). Texts
  with similar meaning get vectors that point in similar directions.
- The cross-encoder (reranker) reads a question and a passage *together*
  and scores how relevant the passage is. It is slower but more accurate
  than comparing vectors, so we use it only on the top 20 search hits.

Models are loaded once at startup (`load_models`) because loading takes
a few seconds; afterwards every request reuses them.
"""

import logging

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)

EMBED_DIM = 384  # size of all-MiniLM-L6-v2 vectors; Qdrant needs to know this

_embedder: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None


def load_models() -> None:
    """Load both models into memory (no-op if already loaded)."""
    global _embedder, _reranker
    if _embedder is None:
        logger.info("Loading embedding model %s", settings.embed_model)
        _embedder = SentenceTransformer(settings.embed_model, device="cpu")
    if _reranker is None:
        logger.info("Loading reranker %s", settings.rerank_model)
        _reranker = CrossEncoder(settings.rerank_model, device="cpu")


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return one normalised vector per text (shape: len(texts) x 384).

    normalize_embeddings=True makes every vector length 1, so the dot
    product of two vectors equals their cosine similarity.
    """
    load_models()
    assert _embedder is not None
    return _embedder.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)


def rerank(query: str, passages: list[str], top_k: int) -> list[int]:
    """Return the indexes of the `top_k` passages most relevant to `query`, best first."""
    if not passages:
        return []
    load_models()
    assert _reranker is not None
    scores = _reranker.predict([(query, p) for p in passages], show_progress_bar=False)
    order = np.argsort(scores)[::-1]  # argsort sorts ascending, [::-1] flips to best-first
    return [int(i) for i in order[:top_k]]
