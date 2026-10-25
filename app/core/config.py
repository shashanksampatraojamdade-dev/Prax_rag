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
    llm_model: str = "gemini-3.8-flash-lite"
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

    # --- Hybrid retrieval + answer gating (prax_re SDK) ---
    enable_hybrid_retrieval: bool = Field(
        True,
        description="Fuse semantic (vector) + keyword (lexical) retrieval via RRF.",
    )
    enable_keyword_retrieval: bool = Field(
        True,
        description="Include Postgres full-text keyword search in hybrid retrieval.",
    )
    retrieval_candidate_k: int = Field(
        10,
        description="Candidates each retriever returns before fusion.",
    )
    rrf_k: int = Field(
        60,
        description="Reciprocal Rank Fusion constant for combining hybrid results.",
    )
    idk_threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum answer confidence (0-1) to answer; below it the engine "
            "declines ('I don't know') instead of hallucinating. 0 = always answer."
        ),
    )
    enable_compression: bool = Field(
        True,
        description=(
            "Contextual compression: trim retrieved chunks to only the "
            "query-relevant sentences before answering (saves tokens, boosts "
            "precision)."
        ),
    )
    compression_max_sentences: int = Field(
        4,
        ge=1,
        description="Max sentences kept per chunk during contextual compression.",
    )
    compression_min_overlap: float = Field(
        0.1,
        ge=0.0,
        le=1.0,
        description=(
            "Min fraction of query terms a sentence must share to be kept during "
            "contextual compression."
        ),
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

    # --- Query understanding & transformation (prax_qt SDK) ---
    enable_query_transformation: bool = Field(
        True,
        description="Master switch: transform each query before retrieval using the prax_qt SDK.",
    )
    enable_query_rewriting: bool = Field(
        True,
        description="Rewrite vague/conversational questions into clear, search-friendly queries.",
    )
    enable_query_expansion: bool = Field(
        True,
        description="Expand queries with synonyms, expanded acronyms and related terms.",
    )
    query_expansion_terms: int = Field(
        6,
        description="Maximum number of related terms query expansion adds to a query.",
    )
    enable_multi_query: bool = Field(
        True,
        description="Generate alternative phrasings of the query and merge their retrieval hits.",
    )
    multi_query_count: int = Field(
        3,
        description="Number of alternative query phrasings generated for multi-query retrieval.",
    )
    enable_hyde: bool = Field(
        True,
        description="HyDE: draft a hypothetical answer and retrieve real documents similar to it.",
    )
    enable_query_classification: bool = Field(
        True,
        description="Classify the query (exact_lookup/semantic/analytical/multi_hop) to adapt retrieval.",
    )
    enable_query_routing: bool = Field(
        True,
        description="Route the query to the best retrieval target among QUERY_ROUTES.",
    )
    query_routes: str = Field(
        "knowledge_base",
        description="Comma-separated candidate retrieval routes for adaptive routing.",
    )
    enable_intent_detection: bool = Field(
        True,
        description="Detect the answer format the user wants (fact/summary/explanation/comparison/list).",
    )
    enable_multi_hop: bool = Field(
        True,
        description="Plan multi-hop retrieval: decompose chained questions into ordered sub-queries.",
    )
    max_hops: int = Field(
        3,
        description="Maximum number of sub-queries in a multi-hop retrieval plan.",
    )
    query_single_call: bool = Field(
        True,
        description="Run all query-understanding techniques in one LLM call (~8x faster).",
    )
    query_max_workers: int = Field(
        8,
        description=(
            "Thread-pool size for the per-technique path: independent "
            "techniques run concurrently in two waves. Set to 1 for sequential."
        ),
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
