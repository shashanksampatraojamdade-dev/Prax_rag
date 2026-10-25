"""Configuration for the LangChain-based Prax retrieval engine (``prax_re_lc``).

Every knob is a plain dataclass field so a host can set it however it likes
(constructor args, ``from_env()``, a config file, defaults). Nothing is required
up front: build ``LCREConfig()`` for all defaults, override only what you need,
or call ``LCREConfig.from_env()`` to pull DB credentials and model names from
environment variables.

The config also knows how to assemble a Postgres connection string from discrete
``POSTGRES_*`` parts *or* accept a single ``DATABASE_URL`` — so the SDK can open
its own DB connection when you use the batteries-included factory, while staying
fully pluggable (inject your own retrievers/LLM and the DB settings are ignored).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        return float(raw) if raw is not None else default
    except ValueError:
        return default


@dataclass
class LCREConfig:
    # --- Hybrid retrieval ------------------------------------------------- #
    enable_semantic: bool = True
    enable_keyword: bool = True
    candidate_k: int = 10
    top_k: int = 5
    rrf_k: int = 60

    # --- Contextual compression ------------------------------------------ #
    enable_compression: bool = True
    compression_max_sentences: int = 4
    compression_min_overlap: float = 0.1

    # --- "I don't know" threshold ---------------------------------------- #
    idk_threshold: float = 0.5
    idk_message: str = (
        "I don't have enough information in the knowledge base to answer that "
        "confidently."
    )

    # --- Answer generation ----------------------------------------------- #
    temperature: float = 0.2
    system_prompt: str = (
        "You are a helpful assistant. Answer the question using ONLY the context "
        "below. Cite sources as [1], [2], … matching the numbered context blocks. "
        "If the context does not contain the answer, say you don't know."
    )

    # --- Vertex AI (defaults mirror the Prax RAG app) -------------------- #
    vertex_project: str = "pid-gousenad-enen-gpaig-01"
    vertex_location: str = "global"
    llm_model: str = "gemini-3.8-flash-lite"
    embedding_model: str = "text-embedding-005"

    # --- Postgres / pgvector --------------------------------------------- #
    # Either provide a full database_url, or the discrete parts below.
    database_url: str | None = None
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_db: str = "praxrag"
    postgres_user: str = "praxrag"
    postgres_password: str = "praxrag"
    pgvector_collection: str = "prax_documents"

    # --- Escape hatch ----------------------------------------------------- #
    # Arbitrary extra kwargs forwarded to the default LLM / embeddings / store
    # builders, so advanced users can tune anything without a config change.
    llm_kwargs: dict = field(default_factory=dict)
    embedding_kwargs: dict = field(default_factory=dict)
    vectorstore_kwargs: dict = field(default_factory=dict)

    @property
    def connection_string(self) -> str:
        """SQLAlchemy/psycopg connection string for the pgvector database."""
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @classmethod
    def from_env(cls, **overrides) -> "LCREConfig":
        """Build config from environment variables, then apply any overrides.

        Recognised env vars (all optional — defaults used when unset):
        retrieval ``ENABLE_SEMANTIC``, ``ENABLE_KEYWORD``,
        ``RETRIEVAL_CANDIDATE_K``, ``RETRIEVAL_TOP_K``, ``RRF_K``;
        compression ``ENABLE_COMPRESSION``, ``COMPRESSION_MAX_SENTENCES``,
        ``COMPRESSION_MIN_OVERLAP``; gating ``IDK_THRESHOLD``;
        models ``VERTEX_PROJECT``, ``VERTEX_LOCATION``, ``LLM_MODEL``,
        ``EMBEDDING_MODEL``, ``LLM_TEMPERATURE``;
        database ``DATABASE_URL`` or ``POSTGRES_HOST`` / ``POSTGRES_PORT`` /
        ``POSTGRES_DB`` / ``POSTGRES_USER`` / ``POSTGRES_PASSWORD`` /
        ``PGVECTOR_COLLECTION``.
        """
        base = cls(
            enable_semantic=_env_bool("ENABLE_SEMANTIC", True),
            enable_keyword=_env_bool("ENABLE_KEYWORD", True),
            candidate_k=_env_int("RETRIEVAL_CANDIDATE_K", 10),
            top_k=_env_int("RETRIEVAL_TOP_K", 5),
            rrf_k=_env_int("RRF_K", 60),
            enable_compression=_env_bool("ENABLE_COMPRESSION", True),
            compression_max_sentences=_env_int("COMPRESSION_MAX_SENTENCES", 4),
            compression_min_overlap=_env_float("COMPRESSION_MIN_OVERLAP", 0.1),
            idk_threshold=_env_float("IDK_THRESHOLD", 0.5),
            temperature=_env_float("LLM_TEMPERATURE", 0.2),
            vertex_project=os.getenv("VERTEX_PROJECT", cls.vertex_project),
            vertex_location=os.getenv("VERTEX_LOCATION", cls.vertex_location),
            llm_model=os.getenv("LLM_MODEL", cls.llm_model),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            database_url=os.getenv("DATABASE_URL"),
            postgres_host=os.getenv("POSTGRES_HOST", cls.postgres_host),
            postgres_port=_env_int("POSTGRES_PORT", cls.postgres_port),
            postgres_db=os.getenv("POSTGRES_DB", cls.postgres_db),
            postgres_user=os.getenv("POSTGRES_USER", cls.postgres_user),
            postgres_password=os.getenv("POSTGRES_PASSWORD", cls.postgres_password),
            pgvector_collection=os.getenv("PGVECTOR_COLLECTION", cls.pgvector_collection),
        )
        for key, value in overrides.items():
            setattr(base, key, value)
        return base
