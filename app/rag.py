"""Explain a mind-map concept using Retrieval-Augmented Generation (RAG).

RAG = first *retrieve* the most relevant passages from the document, then
let the LLM *generate* an answer using only those passages. This keeps
answers grounded in the uploaded file instead of the model's memory.

Flow for one click: build query → embed → Qdrant top 20 → cross-encoder
rerank → top 4 → LLM → add sources → cache in SQLite.
"""

import asyncio
from typing import Any

from app import db, prompts, vector_store
from app.embeddings import embed_texts, rerank
from app.llm import chat_json
from app.models import AskResponse, ExplainResponse, LLMAnswer, LLMExplanation, Level, Source

SEARCH_LIMIT = 20
TOP_K = 4


class NotFound(Exception):
    """The document or node doesn't exist (the API turns this into HTTP 404)."""


class NotReady(Exception):
    """The document is still processing or failed (HTTP 409)."""


def _retrieve_sync(doc_id: str, query: str) -> list[dict[str, Any]]:
    """Embed the query, fetch 20 similar chunks, rerank, and keep the best 4."""
    vector = embed_texts([query])[0].tolist()
    candidates = vector_store.search(doc_id, vector, limit=SEARCH_LIMIT)
    best = rerank(query, [c["text"] for c in candidates], top_k=TOP_K)
    return [candidates[i] for i in best]


async def retrieve(doc_id: str, query: str) -> list[dict[str, Any]]:
    """Run retrieval in a worker thread so the server stays responsive (it's CPU work)."""
    return await asyncio.to_thread(_retrieve_sync, doc_id, query)


def format_context(chunks: list[dict[str, Any]]) -> str:
    """Join chunks into one block, each labelled with where it came from."""
    return "\n\n".join(f"[{c['location']}] {c['text']}" for c in chunks) or "(no matching text found)"


def collect_sources(chunks: list[dict[str, Any]]) -> list[Source]:
    """Build source chips from chunk metadata, without duplicates.

    Sources come from our own data, never from the LLM, so they can't be made up.
    """
    sources: list[Source] = []
    for c in chunks:
        src = Source(source=c["source"], location=c["location"], heading=c.get("heading", ""))
        if src not in sources:
            sources.append(src)
    return sources


def load_ready_document(doc_id: str) -> dict[str, Any]:
    """Fetch a document and make sure its mind map is finished."""
    doc = db.get_document(doc_id)
    if doc is None:
        raise NotFound("Document not found.")
    if doc["status"] != "ready" or not doc["mindmap"]:
        raise NotReady("This document is not ready yet.")
    return doc


async def explain_node(doc_id: str, node_id: str, level: Level) -> ExplainResponse:
    """Return an explanation, quiz and sources for one mind-map node (cached)."""
    doc = load_ready_document(doc_id)
    cached = db.get_cached_explanation(doc_id, node_id, level)
    if cached:
        return ExplainResponse(**{**cached, "cached": True})

    nodes = {n["id"]: n for n in doc["mindmap"]["nodes"]}
    node = nodes.get(node_id)
    if node is None:
        raise NotFound("Concept not found in this mind map.")
    topic = doc["mindmap"]["central_topic"]
    parent = nodes.get(node["parent_id"] or "", {}).get("label", topic)

    query = f"{node['label']}. {parent}. {topic}"
    chunks = await retrieve(doc_id, query)

    user = prompts.EXPLAIN_USER.format(
        label=node["label"],
        parent=parent,
        topic=topic,
        level_instruction=prompts.LEVEL_INSTRUCTIONS[level],
        context=format_context(chunks),
    )
    temperature = 0.4 if level == "simple" else 0.2
    explanation = await chat_json(prompts.EXPLAIN_SYSTEM, user, LLMExplanation, temperature=temperature)

    response = ExplainResponse(
        **explanation.model_dump(),
        node_id=node_id,
        label=node["label"],
        level=level,
        sources=collect_sources(chunks),
    )
    db.cache_explanation(doc_id, node_id, level, response.model_dump())
    return response


async def answer_question(doc_id: str, question: str) -> AskResponse:
    """Answer a free-form question about a document (optional /api/ask)."""
    load_ready_document(doc_id)
    chunks = await retrieve(doc_id, question)
    user = prompts.ASK_USER.format(question=question, context=format_context(chunks))
    answer = await chat_json(prompts.ASK_SYSTEM, user, LLMAnswer)
    return AskResponse(answer=answer.answer, sources=collect_sources(chunks))
