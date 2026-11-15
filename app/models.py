"""Data shapes (schemas) used across the app.

Pydantic models describe exactly what JSON looks like. We use them for:
  1. API requests and responses (FastAPI validates them automatically).
  2. LLM outputs: the schema is sent to Ollama so the model must answer in
     that shape ("structured output"), and we validate the reply with it.
"""

from typing import Literal

from pydantic import BaseModel, Field

Level = Literal["simple", "standard"]


class LLMChildNode(BaseModel):
    """A sub-concept under a branch of the mind map."""

    label: str = Field(description="Concept name, at most 5 words")
    summary: str = Field(description="One sentence, at most 20 words")


class LLMBranch(BaseModel):
    """A main branch coming out of the central topic."""

    label: str = Field(description="Branch name, at most 5 words")
    summary: str = Field(description="One sentence, at most 20 words")
    children: list[LLMChildNode] = Field(min_length=1, max_length=6)


class LLMMindMap(BaseModel):
    """The whole mind map as the LLM produces it (nested tree)."""

    central_topic: str
    branches: list[LLMBranch] = Field(min_length=2, max_length=8)


class Quiz(BaseModel):
    """A single multiple-choice question."""

    question: str
    options: list[str] = Field(min_length=4, max_length=4)
    correct_index: int = Field(ge=0, le=3)
    why: str


class LLMExplanation(BaseModel):
    """What the LLM returns when a learner clicks a concept."""

    simple_explanation: str
    real_life_example: str
    key_points: list[str] = Field(min_length=3, max_length=3)
    quiz: Quiz


class LLMAnswer(BaseModel):
    """What the LLM returns for a free-form question (/api/ask)."""

    answer: str


class MindMapNode(BaseModel):
    """One node of the graph. `parent_id` is None only for the centre."""

    id: str
    label: str
    summary: str
    parent_id: str | None
    level: int


class MindMap(BaseModel):
    """The flattened mind map stored in SQLite and sent to the browser."""

    central_topic: str
    nodes: list[MindMapNode]
    generated_by: Literal["llm", "fallback"] = "llm"


class Source(BaseModel):
    """Where a piece of information came from, e.g. 'notes.pdf, page 3'."""

    source: str
    location: str
    heading: str = ""


class ExplainRequest(BaseModel):
    """Body of POST /api/explain."""

    doc_id: str
    node_id: str
    level: Level = "standard"


class ExplainResponse(LLMExplanation):
    """Explanation plus server-added fields (sources are never made up by the LLM)."""

    node_id: str
    label: str
    level: Level
    sources: list[Source]
    cached: bool = False


class AskRequest(BaseModel):
    """Body of POST /api/ask."""

    doc_id: str
    question: str = Field(min_length=3, max_length=500)


class AskResponse(BaseModel):
    """Answer to a free-form question with its sources."""

    answer: str
    sources: list[Source]


class UploadResponse(BaseModel):
    """Returned immediately after upload; processing continues in background."""

    doc_id: str


class DocumentSummary(BaseModel):
    """One row in the 'Recent documents' list."""

    doc_id: str
    filename: str
    status: str
    progress: int
    created_at: str


class DocumentDetail(BaseModel):
    """Full status of one document, including the mind map once ready."""

    doc_id: str
    filename: str
    status: str
    progress: int
    error: str | None = None
    mindmap: MindMap | None = None


class HealthResponse(BaseModel):
    """Result of GET /api/health: is each dependency reachable?"""

    ollama: bool
    model_available: bool
    qdrant: bool
    model: str
