"""The background job that turns an uploaded file into a mind map.

Stages (status → progress %):
  queued 0 → parsing 10 → embedding 35 → keywords 60 → building_map 75 → ready 100
or `failed` with a human-readable error. The status is saved to SQLite
after each stage, and the frontend polls it to animate the stepper.

Heavy CPU work (parsing, embedding, YAKE) runs in a worker thread via
asyncio.to_thread so the web server keeps answering other requests.
"""

import asyncio
import logging
from pathlib import Path

from app import db, vector_store
from app.embeddings import embed_texts
from app.keywords import extract_keywords
from app.llm import LLMError
from app.mindmap import build_mindmap, fallback_mindmap
from app.parser import ParseError, chunk_sections, parse_document

logger = logging.getLogger(__name__)


def _embed_and_store(doc_id: str, chunks: list[dict]) -> None:
    """Embed every chunk and save the vectors in Qdrant (blocking; run in a thread)."""
    vector_store.delete_document(doc_id)  # clear old points if this is a retry
    vectors = embed_texts([c["text"] for c in chunks])
    vector_store.upsert_chunks(doc_id, chunks, vectors.tolist())


async def process_document(doc_id: str, path: Path, filename: str) -> None:
    """Run the whole pipeline for one document, recording progress as it goes.

    Never raises: any error is stored as status 'failed' so the UI can show it.
    """
    try:
        db.update_status(doc_id, "parsing", 10)
        sections, headings = await asyncio.to_thread(parse_document, path, filename)
        chunks = chunk_sections(sections)
        logger.info("%s: %d sections, %d chunks", filename, len(sections), len(chunks))

        db.update_status(doc_id, "embedding", 35)
        await asyncio.to_thread(_embed_and_store, doc_id, chunks)

        db.update_status(doc_id, "keywords", 60)
        full_text = "\n".join(s.text for s in sections)
        keywords = await asyncio.to_thread(extract_keywords, full_text)

        db.update_status(doc_id, "building_map", 75)
        try:
            mindmap = await build_mindmap(filename, headings, keywords, chunks)
        except LLMError as exc:
            # The LLM failed twice (or is down): build a simpler map so the demo still works.
            logger.warning("LLM mind map failed (%s); using fallback", exc)
            mindmap = fallback_mindmap(filename, headings, keywords, chunks)

        db.save_mindmap(doc_id, mindmap)
        db.update_status(doc_id, "ready", 100)
        logger.info("%s ready (%d nodes, %s)", filename, len(mindmap["nodes"]), mindmap["generated_by"])
    except ParseError as exc:
        db.update_status(doc_id, "failed", 100, str(exc))
    except Exception as exc:
        logger.exception("Processing failed for %s", doc_id)
        db.update_status(doc_id, "failed", 100, f"Something went wrong while processing: {exc}")
