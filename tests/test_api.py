"""End-to-end API tests with FastAPI's TestClient.

They run without Docker or Ollama: Qdrant runs in memory (see conftest.py)
and Ollama points at a closed port, so the pipeline must fall back to the
no-LLM mind map. For explain/ask we swap the LLM call for a fake one.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import rag
from app.config import settings
from app.llm import parse_reply
from app.main import app
from app.mindmap import flatten
from app.models import LLMAnswer, LLMBranch, LLMChildNode, LLMExplanation, LLMMindMap, Quiz

DOCX_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """One client for the module; `with` runs the app's startup (loads models once)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def ready_doc(client: TestClient, sample_files: dict[str, Path]) -> str:
    """Upload the sample DOCX and return its doc_id (background task finishes before we continue)."""
    with sample_files["docx"].open("rb") as f:
        response = client.post("/api/upload", files={"file": ("bio.docx", f, DOCX_TYPE)})
    assert response.status_code == 200
    return response.json()["doc_id"]


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the real Ollama call in rag.py with an instant, predictable answer."""

    async def fake_chat_json(system: str, user: str, schema_model: type, temperature: float = 0.2):
        if schema_model is LLMAnswer:
            return LLMAnswer(answer="Plants make glucose.")
        return LLMExplanation(
            simple_explanation="Plants use light to make food.",
            real_life_example="Like a solar panel charging a phone.",
            key_points=["Needs light", "Makes glucose", "Releases oxygen"],
            quiz=Quiz(question="What gas is released?", options=["Oxygen", "Helium", "Neon", "Argon"], correct_index=0, why="Water is split."),
        )

    monkeypatch.setattr(rag, "chat_json", fake_chat_json)


def test_health_reports_each_service(client: TestClient) -> None:
    """Qdrant (in memory) is up; Ollama is deliberately unreachable in tests."""
    body = client.get("/api/health").json()
    assert body["qdrant"] is True
    assert body["ollama"] is False


def test_rejects_unsupported_file(client: TestClient) -> None:
    response = client.post("/api/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]


def test_rejects_legacy_doc_with_hint(client: TestClient) -> None:
    response = client.post("/api/upload", files={"file": ("old.doc", b"x", "application/msword")})
    assert response.status_code == 400
    assert ".docx" in response.json()["detail"]


def test_rejects_too_large_file(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_upload_mb", 0)
    response = client.post("/api/upload", files={"file": ("big.pdf", b"%PDF-1.4 data", "application/pdf")})
    assert response.status_code == 413


def test_unreadable_file_marks_document_failed(client: TestClient) -> None:
    """A corrupt PDF is accepted, then the pipeline records a friendly error."""
    doc_id = client.post("/api/upload", files={"file": ("bad.pdf", b"not a pdf", "application/pdf")}).json()["doc_id"]
    doc = client.get(f"/api/documents/{doc_id}").json()
    assert doc["status"] == "failed"
    assert doc["error"]


def test_upload_builds_fallback_mindmap_when_llm_down(client: TestClient, ready_doc: str) -> None:
    """With Ollama down the demo still gets a usable map."""
    doc = client.get(f"/api/documents/{ready_doc}").json()
    assert doc["status"] == "ready" and doc["progress"] == 100
    mindmap = doc["mindmap"]
    assert mindmap["generated_by"] == "fallback"
    assert mindmap["nodes"][0] == {**mindmap["nodes"][0], "id": "n0", "level": 0, "parent_id": None}
    assert any(n["level"] == 1 for n in mindmap["nodes"])
    ids = {n["id"] for n in mindmap["nodes"]}
    assert all(n["parent_id"] in ids for n in mindmap["nodes"][1:])


def test_document_list_contains_upload(client: TestClient, ready_doc: str) -> None:
    assert ready_doc in [d["doc_id"] for d in client.get("/api/documents").json()]


def test_unknown_document_is_404(client: TestClient) -> None:
    assert client.get("/api/documents/nope").status_code == 404


def test_flatten_enforces_limits_and_ids() -> None:
    """The server trims oversized LLM output and assigns n0, n1, … in order."""
    branches = [
        LLMBranch(
            label=f"Branch {b}",
            summary="s",
            children=[LLMChildNode(label=f"Child {b}-{c}", summary="s") for c in range(6)],
        )
        for b in range(8)
    ]
    tree = LLMMindMap(central_topic="Topic", branches=branches)
    result = flatten(tree, generated_by="llm")
    nodes = result["nodes"]
    assert sum(n["level"] == 1 for n in nodes) == 7
    assert sum(n["parent_id"] == "n1" for n in nodes) == 5
    assert [n["id"] for n in nodes] == [f"n{i}" for i in range(len(nodes))]


def test_explain_returns_sources_and_caches(client: TestClient, ready_doc: str, fake_llm: None) -> None:
    body = {"doc_id": ready_doc, "node_id": "n1", "level": "standard"}
    first = client.post("/api/explain", json=body).json()
    assert first["cached"] is False
    assert len(first["key_points"]) == 3
    assert first["sources"] and first["sources"][0]["source"] == "bio.docx"
    second = client.post("/api/explain", json=body).json()
    assert second["cached"] is True


def test_explain_without_ollama_is_503(client: TestClient, ready_doc: str) -> None:
    response = client.post("/api/explain", json={"doc_id": ready_doc, "node_id": "n1", "level": "simple"})
    assert response.status_code == 503
    assert "Ollama" in response.json()["detail"]


def test_explain_unknown_node_is_404(client: TestClient, ready_doc: str, fake_llm: None) -> None:
    response = client.post("/api/explain", json={"doc_id": ready_doc, "node_id": "n999"})
    assert response.status_code == 404


def test_explain_rejects_bad_level(client: TestClient, ready_doc: str) -> None:
    response = client.post("/api/explain", json={"doc_id": ready_doc, "node_id": "n1", "level": "expert"})
    assert response.status_code == 422


def test_ask_returns_answer_and_sources(client: TestClient, ready_doc: str, fake_llm: None) -> None:
    body = client.post("/api/ask", json={"doc_id": ready_doc, "question": "What do plants make?"}).json()
    assert body["answer"] == "Plants make glucose."
    assert body["sources"]


def test_delete_removes_document(client: TestClient, ready_doc: str) -> None:
    assert client.delete(f"/api/documents/{ready_doc}").status_code == 200
    assert client.get(f"/api/documents/{ready_doc}").status_code == 404


def test_parse_reply_strips_code_fences() -> None:
    """Models sometimes wrap JSON in ```json fences; we remove them before parsing."""
    assert parse_reply('```json\n{"answer": "hi"}\n```', LLMAnswer).answer == "hi"


def test_frontend_is_served(client: TestClient) -> None:
    """The static site is mounted at / without hiding the API."""
    response = client.get("/")
    assert response.status_code == 200 and "NexusEd AI" in response.text
