from collections.abc import Iterator
from time import perf_counter

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.repositories.vector_store import RetrievedChunk, VectorStore
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from prax_kb import build_context

logger = get_logger(__name__)


def _ms(start: float) -> float:
    return round((perf_counter() - start) * 1000, 1)


class RAGService:
    """Orchestrates retrieval (pgvector) + generation (Vertex LLM)."""

    def __init__(
        self,
        session: Session,
        embeddings: EmbeddingService,
        llm: LLMService,
    ) -> None:
        self._settings = get_settings()
        self._store = VectorStore(session)
        self._embeddings = embeddings
        self._llm = llm

    def _retrieve(self, question: str, top_k: int | None):
        top_k = top_k or self._settings.retrieval_top_k
        query_embedding = self._embeddings.embed_query(question)
        return self._store.similarity_search(query_embedding, top_k=top_k)

    def answer(self, question: str, top_k: int | None = None) -> dict:
        t0 = perf_counter()
        retrieved = self._retrieve(question, top_k)
        retrieval_ms = _ms(t0)

        if not retrieved:
            return {
                "answer": "I don't have any indexed knowledge to answer that yet.",
                "sources": [],
                "timings": {
                    "retrieval_ms": retrieval_ms,
                    "generation_ms": 0.0,
                    "total_ms": retrieval_ms,
                },
            }

        context = self._build_context(retrieved)
        t1 = perf_counter()
        answer = self._llm.generate(question=question, context=context)
        generation_ms = _ms(t1)
        logger.info(
            "Query timing: retrieval=%.1fms generation=%.1fms",
            retrieval_ms,
            generation_ms,
        )
        return {
            "answer": answer,
            "sources": self._serialize_sources(retrieved),
            "timings": {
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": round(retrieval_ms + generation_ms, 1),
            },
        }

    def answer_stream(
        self, question: str, top_k: int | None = None
    ) -> Iterator[dict]:
        """Yield streaming events: one 'sources', many 'token', then 'done'."""
        t0 = perf_counter()
        retrieved = self._retrieve(question, top_k)
        retrieval_ms = _ms(t0)

        if not retrieved:
            yield {"type": "sources", "sources": []}
            yield {
                "type": "token",
                "text": "I don't have any indexed knowledge to answer that yet.",
            }
            yield {
                "type": "done",
                "timings": {
                    "retrieval_ms": retrieval_ms,
                    "generation_ms": 0.0,
                    "total_ms": retrieval_ms,
                },
            }
            return

        yield {"type": "sources", "sources": self._serialize_sources(retrieved)}
        context = self._build_context(retrieved)
        t1 = perf_counter()
        for token in self._llm.generate_stream(question=question, context=context):
            yield {"type": "token", "text": token}
        generation_ms = _ms(t1)
        yield {
            "type": "done",
            "timings": {
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": round(retrieval_ms + generation_ms, 1),
            },
        }

    @staticmethod
    def _serialize_sources(retrieved: list[RetrievedChunk]) -> list[dict]:
        return [
            {
                "rank": i + 1,
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "source": r.source,
                "title": r.title,
                "content_type": r.content_type,
                "score": round(1.0 - r.distance, 4),
                "preview": r.content[:200],
            }
            for i, r in enumerate(retrieved)
        ]

    @staticmethod
    def _build_context(retrieved: list[RetrievedChunk]) -> str:
        # vector_store.RetrievedChunk is structurally compatible with the SDK's
        # RetrievedChunk, so the backend-agnostic builder can consume it directly.
        return build_context(retrieved)
