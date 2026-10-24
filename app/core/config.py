from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application configuration loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Application ---
    app_name: str = "Prax RAG"
    app_env: str = "development"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"

    # --- Vertex AI ---
    vertex_project: str = "pid-gousenad-enen-gpaig-01"
    vertex_location: str = "global"
    llm_model: str = "gemini-3.8-flash"
    embedding_model: str = "text-embedding-005"
    embedding_dim: int = 768

    # --- Postgres / pgvector ---
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_db: str = "praxrag"
    postgres_user: str = "praxrag"
    postgres_password: str = "praxrag"

    # --- RAG parameters ---
    chunk_size: int = Field(800, description="Max characters per chunk.")
    chunk_overlap: int = Field(
        120, description="Character overlap carried between adjacent chunks."
    )
    retrieval_top_k: int = Field(
        5, description="Number of nearest chunks retrieved per query."
    )

    # --- Knowledge base / ingestion quality ---
    chunk_strategy: str = Field(
        "semantic",
        description="Chunking strategy: 'semantic' (embedding breakpoints) or 'recursive'.",
    )
    semantic_breakpoint_percentile: float = Field(
        90.0,
        description="Sentence-distance percentile at which semantic chunking starts a new chunk.",
    )
    enable_ocr: bool = Field(
        True,
        description="Run RapidOCR on scanned pages and images embedded in text PDFs.",
    )
    enable_parent_child: bool = Field(
        True,
        description="Retrieve small child chunks but feed their larger parent block to the LLM.",
    )
    parent_chunk_size: int = Field(
        2000,
        description="Max characters per parent block (broader context for the LLM).",
    )
    enable_deduplication: bool = Field(
        True,
        description="Skip ingesting a document whose content is identical to an existing one.",
    )
    enable_versioning: bool = Field(
        True,
        description="Version documents by source and archive older versions so retrieval ignores them.",
    )
    enable_incremental: bool = Field(
        True,
        description="Reuse embeddings for unchanged chunks when re-indexing a modified document.",
    )
    enable_adaptive_chunking: bool = Field(
        True,
        description="Pick smaller chunks for dense/legal docs and larger chunks for narrative docs.",
    )

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
