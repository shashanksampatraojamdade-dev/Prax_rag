"""Adapters that plug this app's pgvector + Vertex backends into ``prax_re``.

``prax_re`` is backend-agnostic — it only knows the protocols in
:mod:`prax_re.interfaces`. These thin adapters implement those protocols using
the app's :class:`VectorStore`, :class:`EmbeddingService` and
:class:`LLMService`, and expose :func:`build_engine` for standalone reuse of the
retrieval engine outside the request pipeline.
"""

from __future__ import annotations

from app.core.config import Settings
from app.repositories.vector_store import RetrievedChunk, VectorStore
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from prax_re import REConfig, RetrievalEngine, Retrieved


def chunk_to_retrieved(chunk: RetrievedChunk, source_type: str) -> Retrieved:
    """Map an app ``RetrievedChunk`` onto the SDK's ``Retrieved`` type.

    ``score`` is similarity (``1 - distance``); all citation metadata (source
    path, title, upload date, version, parent context) is carried through the
    free-form ``metadata`` dict so the SDK stays ignorant of app specifics.
    """
    return Retrieved(
        id=chunk.chunk_id,
        content=chunk.parent_content or chunk.content,
        score=max(0.0, 1.0 - float(chunk.distance)),
        source_type=source_type,
        metadata={
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "source": chunk.source,
            "title": chunk.title,
            "content_type": chunk.content_type,
            "uploaded_at": (
                chunk.uploaded_at.isoformat()
                if getattr(chunk, "uploaded_at", None) is not None
                else None
            ),
            "version": getattr(chunk, "version", 1),
        },
    )


class SemanticRetriever:
    """``prax_re.Retriever`` backed by pgvector cosine similarity."""

    def __init__(self, embeddings: EmbeddingService, store: VectorStore) -> None:
        self._embeddings = embeddings
        self._store = store

    def search(self, query: str, k: int) -> list[Retrieved]:
        embedding = self._embeddings.embed_query(query)
        hits = self._store.similarity_search(embedding, top_k=k)
        return [chunk_to_retrieved(h, "semantic") for h in hits]


class KeywordRetriever:
    """``prax_re.Retriever`` backed by Postgres full-text search."""

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    def search(self, query: str, k: int) -> list[Retrieved]:
        hits = self._store.keyword_search(query, top_k=k)
        return [chunk_to_retrieved(h, "keyword") for h in hits]


def settings_to_re_config(settings: Settings) -> REConfig:
    return REConfig(
        enable_semantic=True,
        enable_keyword=settings.enable_keyword_retrieval,
        candidate_k=settings.retrieval_candidate_k,
        top_k=settings.retrieval_top_k,
        rrf_k=settings.rrf_k,
        idk_threshold=settings.idk_threshold,
    )


def build_engine(
    settings: Settings,
    embeddings: EmbeddingService,
    store: VectorStore,
    llm: LLMService,
) -> RetrievalEngine:
    """Assemble a standalone ``prax_re`` engine from this app's backends."""
    keyword = (
        KeywordRetriever(store)
        if settings.enable_hybrid_retrieval and settings.enable_keyword_retrieval
        else None
    )
    return RetrievalEngine(
        answer_model=llm,
        semantic=SemanticRetriever(embeddings, store),
        keyword=keyword,
        config=settings_to_re_config(settings),
    )
