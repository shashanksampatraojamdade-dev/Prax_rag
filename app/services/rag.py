from collections.abc import Iterator
from time import perf_counter

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.repositories.vector_store import RetrievedChunk, VectorStore
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.services.qt_adapters import build_query_transformer
from app.services.re_adapters import chunk_to_retrieved
from prax_kb import build_context
from prax_re import compress_content, reciprocal_rank_fusion, score_confidence

logger = get_logger(__name__)

IDK_MESSAGE = (
    "I don't have enough information in the knowledge base to answer that "
    "confidently."
)


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
        # Optional plug-in query understanding (prax_qt SDK). None when disabled.
        self._transformer = build_query_transformer(self._settings, llm)

    def _transform(self, question: str):
        """Apply the query-transformation SDK, returning (probes, info).

        ``probes`` is the list of query strings to embed, search and merge —
        a single-element ``[question]`` when transformation is disabled.
        """
        if self._transformer is None:
            return [question], None
        result = self._transformer.transform(question)
        probes = result.search_queries or [result.search_query or question]
        return probes, result

    def _retrieve(self, probes: list[str], top_k: int | None):
        """Hybrid retrieval: semantic (vector) + keyword (lexical), fused by RRF.

        Returns ``(chunks, fused, mode)`` where ``chunks`` are the ordered
        ``RetrievedChunk`` objects (for context + citations), ``fused`` are the
        SDK ``Retrieved`` hits (carrying scores, for confidence), and ``mode`` is
        the effective retrieval strategy ("hybrid" / "semantic" / "keyword").
        """
        top_k = top_k or self._settings.retrieval_top_k
        cand_k = self._settings.retrieval_candidate_k
        use_keyword = (
            self._settings.enable_hybrid_retrieval
            and self._settings.enable_keyword_retrieval
        )

        chunk_map: dict[int, RetrievedChunk] = {}
        result_lists: list[list] = []

        # Semantic: embed all probes in one batched call, search each.
        if len(probes) == 1:
            embeddings = [self._embeddings.embed_query(probes[0])]
        else:
            embeddings = self._embeddings.embed_queries(probes)
        for embedding in embeddings:
            hits = self._store.similarity_search(embedding, top_k=cand_k)
            sem_list = []
            for h in hits:
                chunk_map.setdefault(h.chunk_id, h)
                sem_list.append(chunk_to_retrieved(h, "semantic"))
            result_lists.append(sem_list)

        # Keyword: lexical search per probe (exact terms / IDs vectors miss).
        if use_keyword:
            for probe in probes:
                hits = self._store.keyword_search(probe, top_k=cand_k)
                kw_list = []
                for h in hits:
                    chunk_map.setdefault(h.chunk_id, h)
                    kw_list.append(chunk_to_retrieved(h, "keyword"))
                result_lists.append(kw_list)

        # Fuse every list (across probes + retrievers) with Reciprocal Rank Fusion.
        if len([lst for lst in result_lists if lst]) > 1:
            fused = reciprocal_rank_fusion(
                result_lists, rrf_k=self._settings.rrf_k, top_k=top_k
            )
        else:
            flat = [r for lst in result_lists for r in lst]
            fused = sorted(flat, key=lambda r: r.score, reverse=True)[:top_k]

        types = {r.source_type for r in fused}
        if "hybrid" in types or len(types) > 1:
            mode = "hybrid"
        elif types:
            mode = next(iter(types))
        else:
            mode = "none"

        chunks = [chunk_map[r.id] for r in fused if r.id in chunk_map]
        logger.info(
            "Retrieval mode=%s | %d probe(s) -> %d fused chunk(s) (keyword=%s)",
            mode,
            len(probes),
            len(chunks),
            use_keyword,
        )
        return chunks, fused, mode

    def _gate(self, fused):
        """Confidence + "I don't know" decision via the prax_re SDK."""
        return score_confidence(fused, idk_threshold=self._settings.idk_threshold)

    def answer(self, question: str, top_k: int | None = None) -> dict:
        tq = perf_counter()
        probes, transform = self._transform(question)
        transform_ms = _ms(tq)

        t0 = perf_counter()
        retrieved, fused, mode = self._retrieve(probes, top_k)
        retrieval_ms = _ms(t0)

        if not retrieved:
            return {
                "answer": "I don't have any indexed knowledge to answer that yet.",
                "confidence": 0.0,
                "confidence_label": "low",
                "answerable": False,
                "retrieval_mode": mode,
                "sources": [],
                "query_transformation": transform.as_dict() if transform else None,
                "timings": {
                    "query_transform_ms": transform_ms,
                    "retrieval_ms": retrieval_ms,
                    "generation_ms": 0.0,
                    "total_ms": round(transform_ms + retrieval_ms, 1),
                },
            }

        conf = self._gate(fused)
        # "I don't know" threshold: decline instead of hallucinating when the
        # retrieved evidence is too weak to support a grounded answer.
        if not conf.answerable:
            logger.info(
                "Declining answer: confidence %.4f < idk_threshold %.2f",
                conf.score,
                self._settings.idk_threshold,
            )
            return {
                "answer": IDK_MESSAGE,
                "confidence": conf.score,
                "confidence_label": conf.label,
                "answerable": False,
                "retrieval_mode": mode,
                "sources": self._serialize_sources(retrieved),
                "query_transformation": transform.as_dict() if transform else None,
                "timings": {
                    "query_transform_ms": transform_ms,
                    "retrieval_ms": retrieval_ms,
                    "generation_ms": 0.0,
                    "total_ms": round(transform_ms + retrieval_ms, 1),
                },
            }

        context = self._build_context(retrieved, question)
        t1 = perf_counter()
        answer = self._llm.generate(question=question, context=context)
        generation_ms = _ms(t1)
        logger.info(
            "Query timing: transform=%.1fms retrieval=%.1fms generation=%.1fms "
            "| mode=%s confidence=%.4f (%s)",
            transform_ms,
            retrieval_ms,
            generation_ms,
            mode,
            conf.score,
            conf.label,
        )
        return {
            "answer": answer,
            "confidence": conf.score,
            "confidence_label": conf.label,
            "answerable": True,
            "retrieval_mode": mode,
            "sources": self._serialize_sources(retrieved),
            "query_transformation": transform.as_dict() if transform else None,
            "timings": {
                "query_transform_ms": transform_ms,
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": round(transform_ms + retrieval_ms + generation_ms, 1),
            },
        }

    def answer_stream(
        self, question: str, top_k: int | None = None
    ) -> Iterator[dict]:
        """Yield streaming events: one 'sources', many 'token', then 'done'."""
        tq = perf_counter()
        probes, transform = self._transform(question)
        transform_ms = _ms(tq)

        t0 = perf_counter()
        retrieved, fused, mode = self._retrieve(probes, top_k)
        retrieval_ms = _ms(t0)

        if not retrieved:
            yield {
                "type": "sources",
                "sources": [],
                "confidence": 0.0,
                "confidence_label": "low",
                "answerable": False,
                "retrieval_mode": mode,
                "query_transformation": transform.as_dict() if transform else None,
            }
            yield {
                "type": "token",
                "text": "I don't have any indexed knowledge to answer that yet.",
            }
            yield {
                "type": "done",
                "confidence": 0.0,
                "confidence_label": "low",
                "answerable": False,
                "retrieval_mode": mode,
                "timings": {
                    "query_transform_ms": transform_ms,
                    "retrieval_ms": retrieval_ms,
                    "generation_ms": 0.0,
                    "total_ms": round(transform_ms + retrieval_ms, 1),
                },
            }
            return

        conf = self._gate(fused)
        yield {
            "type": "sources",
            "sources": self._serialize_sources(retrieved),
            "confidence": conf.score,
            "confidence_label": conf.label,
            "answerable": conf.answerable,
            "retrieval_mode": mode,
            "query_transformation": transform.as_dict() if transform else None,
        }

        # "I don't know" threshold: decline (stream the safe message) instead of
        # generating when the retrieved evidence is too weak.
        if not conf.answerable:
            logger.info(
                "Declining answer (stream): confidence %.4f < idk_threshold %.2f",
                conf.score,
                self._settings.idk_threshold,
            )
            yield {"type": "token", "text": IDK_MESSAGE}
            yield {
                "type": "done",
                "confidence": conf.score,
                "confidence_label": conf.label,
                "answerable": False,
                "retrieval_mode": mode,
                "timings": {
                    "query_transform_ms": transform_ms,
                    "retrieval_ms": retrieval_ms,
                    "generation_ms": 0.0,
                    "total_ms": round(transform_ms + retrieval_ms, 1),
                },
            }
            return

        context = self._build_context(retrieved, question)
        t1 = perf_counter()
        for token in self._llm.generate_stream(question=question, context=context):
            yield {"type": "token", "text": token}
        generation_ms = _ms(t1)
        yield {
            "type": "done",
            "confidence": conf.score,
            "confidence_label": conf.label,
            "answerable": True,
            "retrieval_mode": mode,
            "timings": {
                "query_transform_ms": transform_ms,
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": round(transform_ms + retrieval_ms + generation_ms, 1),
            },
        }

    @staticmethod
    def _serialize_sources(retrieved: list[RetrievedChunk]) -> list[dict]:
        """Serialize sources defensively.

        The optional metadata (uploaded_at, version, title, content_type) is not
        guaranteed — this retrieval layer may run over a KB that never recorded
        it. Missing or malformed fields are skipped/defaulted rather than raised,
        so a leaner backend still gets a valid answer with whatever it does have.
        """
        sources: list[dict] = []
        for i, r in enumerate(retrieved):
            item = {
                "rank": i + 1,
                "chunk_id": getattr(r, "chunk_id", None),
                "document_id": getattr(r, "document_id", None),
                "source": getattr(r, "source", None),
                "title": getattr(r, "title", None),
                "content_type": getattr(r, "content_type", "text"),
                "score": round(1.0 - r.distance, 4),
                "preview": (getattr(r, "content", "") or "")[:200],
            }
            # Optional metadata — include only when present and well-formed.
            uploaded_at = getattr(r, "uploaded_at", None)
            if uploaded_at is not None:
                item["uploaded_at"] = (
                    uploaded_at.isoformat()
                    if hasattr(uploaded_at, "isoformat")
                    else str(uploaded_at)
                )
            version = getattr(r, "version", None)
            if version is not None:
                item["version"] = version
            sources.append(item)
        return sources

    def _build_context(
        self, retrieved: list[RetrievedChunk], question: str | None = None
    ) -> str:
        # vector_store.RetrievedChunk is structurally compatible with the SDK's
        # RetrievedChunk, so the backend-agnostic builder can consume it directly.
        if question and getattr(self._settings, "enable_compression", False):
            retrieved = self._compress_chunks(retrieved, question)
        return build_context(retrieved)

    def _compress_chunks(
        self, retrieved: list[RetrievedChunk], question: str
    ) -> list[RetrievedChunk]:
        """Trim each chunk's text to only the question-relevant sentences.

        Contextual compression: shrinks the (often large, parent-expanded) blocks
        fed to the LLM to the sentences that matter for this question, saving
        tokens and sharpening precision. Never raises — falls back to the full
        chunk on any error.
        """
        from dataclasses import replace

        max_sents = getattr(self._settings, "compression_max_sentences", 4)
        min_overlap = getattr(self._settings, "compression_min_overlap", 0.1)
        compressed: list[RetrievedChunk] = []
        saved = 0
        for r in retrieved:
            try:
                field = "parent_content" if r.parent_content else "content"
                original = getattr(r, field) or ""
                trimmed = compress_content(
                    question, original, max_sentences=max_sents, min_overlap=min_overlap
                )
                saved += max(0, len(original) - len(trimmed))
                compressed.append(replace(r, **{field: trimmed}))
            except Exception:
                logger.exception("Compression failed for chunk; using full content")
                compressed.append(r)
        logger.info("Contextual compression saved ~%d char(s) of context", saved)
        return compressed
