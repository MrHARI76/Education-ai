"""Talk to the local LLM through Ollama's HTTP API.

We call `/api/chat` directly with httpx (no LangChain). The key trick is
"structured output": we send a JSON Schema in the `format` field and
Ollama constrains the model so it can only produce JSON of that shape.
We still validate the reply with Pydantic, and retry once if it's bad.
"""

import json
import logging
import re
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)  # "any Pydantic model" — lets chat_json return the right type

RETRY_REMINDER = "\n\nReturn only valid JSON matching the schema. No extra text."


class LLMError(Exception):
    """The LLM answered, but not with usable JSON (after one retry)."""


class LLMUnavailable(LLMError):
    """Ollama could not be reached, timed out, or the model isn't pulled."""


def strip_fences(text: str) -> str:
    """Remove accidental Markdown code fences like ```json ... ```."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


async def _post_chat(system: str, user: str, schema: dict, temperature: float) -> str:
    """Send one chat request and return the raw text the model produced."""
    body = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "format": schema,  # Ollama structured outputs: reply must match this JSON Schema
        "options": {"num_ctx": settings.llm_num_ctx, "temperature": temperature},
        "stream": False,  # one complete reply instead of a token-by-token stream
    }
    try:
        async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
            response = await client.post(f"{settings.ollama_url}/api/chat", json=body)
    except httpx.TimeoutException as exc:
        raise LLMUnavailable("The local AI model took too long to answer. Try again.") from exc
    except httpx.HTTPError as exc:
        raise LLMUnavailable("Ollama is not reachable. Is it running?") from exc
    if response.status_code == 404:  # Ollama returns 404 when the model isn't pulled
        raise LLMUnavailable(f"Model '{settings.llm_model}' not found. Run: ollama pull {settings.llm_model}")
    if response.status_code != 200:
        raise LLMError(f"Ollama error {response.status_code}: {response.text[:200]}")
    return response.json()["message"]["content"]


def parse_reply(raw: str, schema_model: type[T]) -> T:
    """Turn the model's text into a validated Pydantic object (raises on failure)."""
    return schema_model.model_validate(json.loads(strip_fences(raw)))


async def chat_json(system: str, user: str, schema_model: type[T], temperature: float = 0.2) -> T:
    """Ask the LLM for JSON matching `schema_model`; retry once on bad output.

    Low temperature (0.2) makes answers more focused and repeatable.
    Raises LLMUnavailable if Ollama is down, LLMError if output stays invalid.
    """
    schema = schema_model.model_json_schema()
    for attempt in (1, 2):
        prompt = user if attempt == 1 else user + RETRY_REMINDER
        raw = await _post_chat(system, prompt, schema, temperature)
        try:
            return parse_reply(raw, schema_model)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("LLM returned invalid JSON (attempt %d): %s", attempt, str(exc)[:300])
    raise LLMError("The AI model returned an answer in the wrong format. Please try again.")


async def health() -> tuple[bool, bool]:
    """Return (ollama_reachable, model_is_pulled) using GET /api/tags."""
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            response = await client.get(f"{settings.ollama_url}/api/tags")
        response.raise_for_status()
    except httpx.HTTPError:
        return False, False
    names = {m.get("name", "") for m in response.json().get("models", [])}
    wanted = settings.llm_model
    # "llama3.1" with no tag means "llama3.1:latest" to Ollama.
    available = wanted in names or (":" not in wanted and f"{wanted}:latest" in names)
    return True, available
