"""FastAPI application: HTTP routes and the static frontend.

This is the entry point (`uvicorn app.main:app`). It wires the other
modules together:
  upload  → save file → db row → background pipeline
  explain → rag.explain_node
and serves the HTML/JS frontend at "/".
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import db, rag, vector_store
from app.config import settings
from app.embeddings import load_models
from app.llm import LLMError, LLMUnavailable, health as llm_health
from app.models import (
    AskRequest,
    AskResponse,
    DocumentDetail,
    DocumentSummary,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    UploadResponse,
)
from app.parser import ParseError, check_extension
from app.pipeline import process_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("nexused")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
READ_BLOCK = 1024 * 1024
ACTIVE_STATUSES = ("queued", "parsing", "embedding", "keywords", "building_map")


def mark_interrupted_documents() -> None:
    """Documents left 'processing' by a crash/restart would spin forever; mark them failed."""
    for doc in db.list_documents(limit=1000):
        if doc["status"] in ACTIVE_STATUSES:
            db.update_status(doc["doc_id"], "failed", 100, "Processing was interrupted. Press Retry.")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Startup: create DB, load ML models once, connect to Qdrant."""
    db.init_db()
    mark_interrupted_documents()
    await asyncio.to_thread(load_models)
    try:
        await asyncio.to_thread(vector_store.ensure_collection)
    except Exception:
        logger.exception("Could not connect to Qdrant at %s", settings.qdrant_url)
    logger.info("NexusEd AI ready (model: %s)", settings.llm_model)
    yield


app = FastAPI(title="NexusEd AI", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_document(doc_id: str) -> dict:
    """Fetch a document or raise HTTP 404."""
    doc = db.get_document(doc_id)
    if doc is None:
        raise HTTPException(404, "Document not found.")
    return doc


def llm_http_error(exc: LLMError) -> HTTPException:
    """Map LLM problems to HTTP codes: 503 = Ollama down, 502 = bad model output."""
    return HTTPException(503 if isinstance(exc, LLMUnavailable) else 502, str(exc))


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Report whether Ollama, the model and Qdrant are available (drives the status chips)."""
    ollama_ok, model_ok = await llm_health()
    qdrant_ok = await asyncio.to_thread(vector_store.is_healthy)
    return HealthResponse(ollama=ollama_ok, model_available=model_ok, qdrant=qdrant_ok, model=settings.llm_model)


async def save_upload(file: UploadFile, dest: Path) -> None:
    """Stream the upload to disk, stopping with HTTP 413 if it exceeds the size limit."""
    max_bytes = settings.max_upload_mb * 1024 * 1024
    written = 0
    with dest.open("wb") as out:
        while block := await file.read(READ_BLOCK):
            written += len(block)
            if written > max_bytes:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File is too large. The limit is {settings.max_upload_mb} MB.")
            out.write(block)
    if written == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "The file is empty.")


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile, background: BackgroundTasks) -> UploadResponse:
    """Accept one file, save it, and start processing in the background.

    Returns the doc_id immediately; the browser then polls for progress.
    """
    filename = Path(file.filename or "").name
    try:
        ext = check_extension(filename)
    except ParseError as exc:
        raise HTTPException(400, str(exc)) from exc

    doc_id = uuid.uuid4().hex
    dest = settings.upload_dir / f"{doc_id}{ext}"
    await save_upload(file, dest)
    db.create_document(doc_id, filename, str(dest))
    background.add_task(process_document, doc_id, dest, filename)
    logger.info("Uploaded %s as %s", filename, doc_id)
    return UploadResponse(doc_id=doc_id)


@app.get("/api/documents", response_model=list[DocumentSummary])
def list_documents() -> list[dict]:
    """Recent documents for the upload screen."""
    return db.list_documents()


@app.get("/api/documents/{doc_id}", response_model=DocumentDetail)
def get_document(doc_id: str) -> dict:
    """Status, progress and (when ready) the mind map of one document."""
    return require_document(doc_id)


@app.post("/api/documents/{doc_id}/retry", response_model=UploadResponse)
def retry_document(doc_id: str, background: BackgroundTasks) -> UploadResponse:
    """Re-run the pipeline on an already uploaded file (used by the Retry button)."""
    doc = require_document(doc_id)
    if doc["status"] in ACTIVE_STATUSES:
        raise HTTPException(409, "This document is already being processed.")
    db.update_status(doc_id, "queued", 0)
    background.add_task(process_document, doc_id, Path(doc["path"]), doc["filename"])
    return UploadResponse(doc_id=doc_id)


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    """Delete the file, its vectors in Qdrant, its cache and its DB row."""
    doc = require_document(doc_id)
    Path(doc["path"]).unlink(missing_ok=True)
    try:
        vector_store.delete_document(doc_id)
    except Exception:
        logger.warning("Could not delete Qdrant points for %s", doc_id)
    db.delete_document(doc_id)
    return {"deleted": doc_id}


@app.post("/api/explain", response_model=ExplainResponse)
async def explain(body: ExplainRequest) -> ExplainResponse:
    """Explain one mind-map concept with retrieval + rerank + LLM."""
    try:
        return await rag.explain_node(body.doc_id, body.node_id, body.level)
    except rag.NotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except rag.NotReady as exc:
        raise HTTPException(409, str(exc)) from exc
    except LLMError as exc:
        raise llm_http_error(exc) from exc


@app.post("/api/ask", response_model=AskResponse)
async def ask(body: AskRequest) -> AskResponse:
    """Answer a free-form question about a document."""
    try:
        return await rag.answer_question(body.doc_id, body.question)
    except rag.NotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except rag.NotReady as exc:
        raise HTTPException(409, str(exc)) from exc
    except LLMError as exc:
        raise llm_http_error(exc) from exc


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
