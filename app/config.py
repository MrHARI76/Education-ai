"""Application settings.

Every tunable value (URLs, model names, limits) lives here. Values are read
from environment variables or a `.env` file, so the same code runs inside
Docker and on a laptop without edits. Other modules import `settings`.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration, with safe defaults for the Docker setup.

    pydantic-settings matches each field to an environment variable of the
    same name (case-insensitive), e.g. `LLM_MODEL=llama3.2:3b` overrides
    `llm_model`.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Ollama (the local LLM server running on the host machine) ---
    ollama_url: str = "http://host.docker.internal:11434"
    llm_model: str = "llama3.1:8b"
    llm_num_ctx: int = 8192  # how many tokens the model can "see" at once
    llm_timeout: float = 180.0  # seconds; local models can be slow on CPU

    # --- Qdrant (the vector database) ---
    # Use ":memory:" to run Qdrant inside Python without a server (used by tests).
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "nexus_chunks"

    # --- Machine-learning models (downloaded once, then used offline) ---
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Storage and limits ---
    data_dir: Path = Path("./data")
    max_upload_mb: int = 25

    @property
    def upload_dir(self) -> Path:
        """Folder where uploaded files are saved."""
        return self.data_dir / "uploads"

    @property
    def db_path(self) -> Path:
        """Path of the SQLite database file."""
        return self.data_dir / "nexus.db"


@lru_cache
def get_settings() -> Settings:
    """Build the settings once and reuse them (lru_cache remembers the result)."""
    return Settings()


settings = get_settings()
