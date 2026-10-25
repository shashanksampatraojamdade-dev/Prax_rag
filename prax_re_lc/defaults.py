"""Batteries-included defaults: build the whole engine from env with one call.

Everything here is **optional and pluggable**. Inject your own LangChain
retrievers/LLM into ``RetrievalEngine`` and you never touch this module. But if
you just want "point it at my pgvector DB + Vertex and go", call
``build_engine_from_env()`` — it reads DB credentials and model names from
environment variables (see ``LCREConfig.from_env``) and wires:

- a **semantic** retriever over ``langchain_postgres.PGVector`` (scored), 
- a **keyword** retriever doing Postgres full-text search over the same table,
- a **Vertex AI** chat model + embeddings,
- the ``RetrievalEngine`` (hybrid RRF + confidence gate + compression + citations).

Heavy third-party imports (``langchain_postgres``, ``langchain_google_vertexai``)
are done lazily inside the builders, so importing ``prax_re_lc`` stays cheap and
dependency-light. Install them with the ``langchain`` extra:

    pip install "prax-rag[langchain]"
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from prax_re_lc.config import LCREConfig
from prax_re_lc.engine import RetrievalEngine

logger = logging.getLogger(__name__)


def _missing(pkg: str, extra: str) -> "ImportError":
    return ImportError(
        f"'{pkg}' is required for this default builder. Install it with "
        f"`pip install \"prax-rag[langchain]\"` (or `pip install {extra}`), or "
        f"inject your own LangChain components into RetrievalEngine instead."
    )


# --- Vertex AI models ----------------------------------------------------- #
def build_vertex_embeddings(config: LCREConfig | None = None):
    """Vertex AI text-embeddings model (``langchain_google_vertexai``)."""
    config = config or LCREConfig()
    try:
        from langchain_google_vertexai import VertexAIEmbeddings
    except ImportError as exc:  # pragma: no cover - env dependent
        raise _missing("langchain_google_vertexai", "langchain-google-vertexai") from exc
    return VertexAIEmbeddings(
        model_name=config.embedding_model,
        project=config.vertex_project,
        location=config.vertex_location,
        **config.embedding_kwargs,
    )


def build_vertex_llm(config: LCREConfig | None = None):
    """Vertex AI Gemini chat model (``langchain_google_vertexai``)."""
    config = config or LCREConfig()
    try:
        from langchain_google_vertexai import ChatVertexAI
    except ImportError as exc:  # pragma: no cover - env dependent
        raise _missing("langchain_google_vertexai", "langchain-google-vertexai") from exc
    return ChatVertexAI(
        model=config.llm_model,
        project=config.vertex_project,
        location=config.vertex_location,
        temperature=config.temperature,
        **config.llm_kwargs,
    )


# --- Vector store / semantic retriever ------------------------------------ #
def build_vectorstore(config: LCREConfig | None = None, embeddings: Any = None):
    """A ``langchain_postgres.PGVector`` store over the configured database."""
    config = config or LCREConfig()
    try:
        from langchain_postgres import PGVector
    except ImportError as exc:  # pragma: no cover - env dependent
        raise _missing("langchain_postgres", "langchain-postgres") from exc
    embeddings = embeddings or build_vertex_embeddings(config)
    return PGVector(
        embeddings=embeddings,
        collection_name=config.pgvector_collection,
        connection=config.connection_string,
        use_jsonb=True,
        **config.vectorstore_kwargs,
    )


class _ScoredVectorStoreRetriever(BaseRetriever):
    """Semantic retriever that returns scored ``Document``s from a vector store.

    Unlike ``vectorstore.as_retriever()`` (which drops scores), this writes the
    relevance score into ``metadata['score']`` so the engine's confidence gate
    works end to end.
    """

    vectorstore: Any
    k: int = 10

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:  # noqa: D401
        try:
            pairs = self.vectorstore.similarity_search_with_relevance_scores(query, k=self.k)
        except Exception:
            # Fall back to plain similarity search without scores.
            docs = self.vectorstore.similarity_search(query, k=self.k)
            pairs = [(d, None) for d in docs]
        out: list[Document] = []
        for rank, (doc, score) in enumerate(pairs):
            meta = dict(doc.metadata or {})
            if score is not None:
                meta["score"] = max(0.0, min(1.0, float(score)))
            meta.setdefault("source_type", "semantic")
            out.append(Document(page_content=doc.page_content, metadata=meta))
        return out


def build_semantic_retriever(config: LCREConfig | None = None, vectorstore: Any = None):
    """Scored pgvector semantic retriever."""
    config = config or LCREConfig()
    vectorstore = vectorstore if vectorstore is not None else build_vectorstore(config)
    return _ScoredVectorStoreRetriever(vectorstore=vectorstore, k=config.candidate_k)


# --- Keyword retriever (Postgres full-text) ------------------------------- #
class _PGKeywordRetriever(BaseRetriever):
    """Lexical retriever doing Postgres FTS over the langchain_postgres table.

    Searches ``langchain_pg_embedding.document`` for the configured collection
    using ``websearch_to_tsquery`` + ``ts_rank_cd`` (ILIKE fallback for single
    tokens / IDs). Scores are normalised to 0-1 and stored in
    ``metadata['score']`` so hybrid fusion + confidence work the same as the
    semantic side.
    """

    connection_string: str
    collection_name: str
    k: int = 10

    model_config = {"arbitrary_types_allowed": True}

    def _engine(self):
        from sqlalchemy import create_engine

        url = self.connection_string.replace("postgresql+psycopg", "postgresql+psycopg")
        return create_engine(url)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:  # noqa: D401
        from sqlalchemy import text

        sql = text(
            """
            SELECT e.document AS content,
                   e.cmetadata AS meta,
                   ts_rank_cd(to_tsvector('english', e.document),
                              websearch_to_tsquery('english', :q)) AS rank
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON c.uuid = e.collection_id
            WHERE c.name = :coll
              AND to_tsvector('english', e.document) @@ websearch_to_tsquery('english', :q)
            ORDER BY rank DESC
            LIMIT :k
            """
        )
        fallback = text(
            """
            SELECT e.document AS content, e.cmetadata AS meta, 0.0 AS rank
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON c.uuid = e.collection_id
            WHERE c.name = :coll AND e.document ILIKE :like
            LIMIT :k
            """
        )
        try:
            engine = self._engine()
            with engine.connect() as conn:
                rows = conn.execute(sql, {"q": query, "coll": self.collection_name, "k": self.k}).fetchall()
                if not rows:
                    rows = conn.execute(
                        fallback,
                        {"coll": self.collection_name, "like": f"%{query}%", "k": self.k},
                    ).fetchall()
        except Exception:
            logger.exception("Keyword retrieval failed; returning no keyword hits")
            return []

        max_rank = max((float(r.rank) for r in rows), default=0.0) or 1.0
        out: list[Document] = []
        for r in rows:
            meta = dict(r.meta or {})
            meta["score"] = max(0.0, min(1.0, float(r.rank) / max_rank)) if r.rank else 0.5
            meta["source_type"] = "keyword"
            out.append(Document(page_content=r.content or "", metadata=meta))
        return out


def build_keyword_retriever(config: LCREConfig | None = None):
    """Postgres full-text keyword retriever over the pgvector collection."""
    config = config or LCREConfig()
    return _PGKeywordRetriever(
        connection_string=config.connection_string,
        collection_name=config.pgvector_collection,
        k=config.candidate_k,
    )


# --- One-call factory ----------------------------------------------------- #
def build_engine_from_env(config: LCREConfig | None = None, **overrides) -> RetrievalEngine:
    """Assemble a ready-to-use ``RetrievalEngine`` from env vars + defaults.

    Any keyword override is applied on top of the env-derived config, e.g.::

        engine = build_engine_from_env(top_k=8, idk_threshold=0.4)

    Pass ``config=`` to supply a fully custom ``LCREConfig`` instead.
    """
    cfg = config or LCREConfig.from_env(**overrides)
    embeddings = build_vertex_embeddings(cfg)
    vectorstore = build_vectorstore(cfg, embeddings)
    semantic = build_semantic_retriever(cfg, vectorstore) if cfg.enable_semantic else None
    keyword = build_keyword_retriever(cfg) if cfg.enable_keyword else None
    llm = build_vertex_llm(cfg)
    logger.info(
        "Built prax_re_lc engine (semantic=%s keyword=%s model=%s collection=%s)",
        semantic is not None,
        keyword is not None,
        cfg.llm_model,
        cfg.pgvector_collection,
    )
    return RetrievalEngine(answer_model=llm, semantic=semantic, keyword=keyword, config=cfg)
