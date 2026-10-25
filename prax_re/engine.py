"""The Prax retrieval engine — hybrid retrieval + grounded answering.

``RetrievalEngine`` is the plug-and-play entry point. Compose it with your own
retrievers and answer model (any objects satisfying the protocols in
:mod:`prax_re.interfaces`)::

    from prax_re import RetrievalEngine, REConfig

    engine = RetrievalEngine(
        semantic=my_vector_retriever,   # .search(query, k) -> list[Retrieved]
        keyword=my_keyword_retriever,   # .search(query, k) -> list[Retrieved]
        answer_model=my_llm,            # .generate(question, context) -> str
        config=REConfig(),
    )
    result = engine.answer("How many vacation days do I get?")
    print(result.answer, result.confidence, result.citations)

The engine owns the reusable retrieval/answer logic — hybrid fusion, confidence
scoring, the "I don't know" threshold and citation assembly — while delegating
all backend specifics (DB, embedder, LLM) to the injected objects.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
import logging

from prax_re.compression import compress_results
from prax_re.confidence import Confidence, score_confidence
from prax_re.config import REConfig
from prax_re.fusion import reciprocal_rank_fusion
from prax_re.interfaces import AnswerModel, Retrieved, Retriever

logger = logging.getLogger(__name__)


@dataclass
class AnswerResult:
    answer: str
    confidence: float
    confidence_label: str
    answerable: bool
    citations: list[dict] = field(default_factory=list)
    results: list[Retrieved] = field(default_factory=list)
    retrieval_mode: str = "hybrid"


class RetrievalEngine:
    """Reusable hybrid retrieval + grounded answering engine."""

    def __init__(
        self,
        answer_model: AnswerModel,
        semantic: Retriever | None = None,
        keyword: Retriever | None = None,
        config: REConfig | None = None,
        compressor: "Callable[[str, str], str] | None" = None,
    ) -> None:
        self._answer_model = answer_model
        self._semantic = semantic
        self._keyword = keyword
        self._config = config or REConfig()
        # Optional custom compressor (e.g. an LLM-based one). When ``None`` the
        # built-in dependency-free extractive compressor is used.
        self._compressor = compressor

    # --- Retrieval -------------------------------------------------------- #
    def retrieve(self, query: str) -> list[Retrieved]:
        """Run the enabled retrievers and fuse their hits (hybrid by default)."""
        cfg = self._config
        lists: list[list[Retrieved]] = []
        modes: list[str] = []
        if cfg.enable_semantic and self._semantic is not None:
            hits = self._semantic.search(query, cfg.candidate_k)
            logger.info("Semantic retrieval returned %d hit(s)", len(hits))
            lists.append(hits)
            modes.append("semantic")
        if cfg.enable_keyword and self._keyword is not None:
            hits = self._keyword.search(query, cfg.candidate_k)
            logger.info("Keyword retrieval returned %d hit(s)", len(hits))
            lists.append(hits)
            modes.append("keyword")

        if not lists:
            logger.warning("No retrievers enabled/available; returning no hits")
            return []
        if len(lists) == 1:
            logger.info("Single-retriever mode (%s); skipping fusion", modes[0])
            return lists[0][: cfg.top_k]
        return reciprocal_rank_fusion(lists, rrf_k=cfg.rrf_k, top_k=cfg.top_k)

    # --- Answering -------------------------------------------------------- #
    def answer(self, query: str) -> AnswerResult:
        """Retrieve, gate on confidence, then generate a grounded answer.

        Declines with ``config.idk_message`` when confidence is below the
        configured threshold, so the engine never hallucinates past its evidence.
        """
        results = self.retrieve(query)
        mode = self._mode(results)
        conf = score_confidence(results, idk_threshold=self._config.idk_threshold)
        citations = self.build_citations(results)

        if not conf.answerable:
            logger.info("Engine declining to answer (confidence below threshold)")
            return AnswerResult(
                answer=self._config.idk_message,
                confidence=conf.score,
                confidence_label=conf.label,
                answerable=False,
                citations=citations,
                results=results,
                retrieval_mode=mode,
            )

        context = self.build_context(self.compress(query, results))
        answer_text = self._answer_model.generate(question=query, context=context)
        return AnswerResult(
            answer=answer_text,
            confidence=conf.score,
            confidence_label=conf.label,
            answerable=True,
            citations=citations,
            results=results,
            retrieval_mode=mode,
        )

    def retrieve_and_gate(self, query: str) -> tuple[list[Retrieved], Confidence, str]:
        """Retrieve + score confidence without generating (for streaming hosts).

        Returns ``(results, confidence, retrieval_mode)`` so a host can emit
        citations/confidence first, then stream the answer itself.
        """
        results = self.retrieve(query)
        conf = score_confidence(results, idk_threshold=self._config.idk_threshold)
        return results, conf, self._mode(results)

    # --- Helpers ---------------------------------------------------------- #
    def compress(self, query: str, results: list[Retrieved]) -> list[Retrieved]:
        """Trim each hit to only its query-relevant sentences (contextual
        compression). Returns ``results`` unchanged when compression is disabled.

        Uses an injected ``compressor`` callable if one was provided, otherwise
        the built-in dependency-free extractive compressor.
        """
        cfg = self._config
        if not cfg.enable_compression or not results:
            return results
        if self._compressor is None:
            return compress_results(query, results, cfg)

        compressed: list[Retrieved] = []
        for r in results:
            try:
                new_content = self._compressor(query, r.content)
            except Exception:
                logger.exception("Custom compressor failed for %s; using full", r.id)
                compressed.append(r)
                continue
            meta = dict(r.metadata or {})
            meta.setdefault("uncompressed_content", r.content)
            compressed.append(
                Retrieved(
                    id=r.id,
                    content=new_content,
                    score=r.score,
                    source_type=r.source_type,
                    metadata=meta,
                )
            )
        return compressed

    @staticmethod
    def _mode(results: list[Retrieved]) -> str:
        types = {r.source_type for r in results}
        if not types:
            return "none"
        if types == {"hybrid"} or len(types) > 1:
            return "hybrid"
        return next(iter(types))

    @staticmethod
    def build_context(results: list[Retrieved]) -> str:
        """Assemble a numbered context block the model can cite as [1], [2]...."""
        blocks = []
        for i, r in enumerate(results, start=1):
            label = r.metadata.get("title") or r.metadata.get("source") or f"chunk {r.id}"
            blocks.append(f"[{i}] ({label})\n{r.content}")
        return "\n\n".join(blocks)

    @staticmethod
    def build_citations(results: list[Retrieved]) -> list[dict]:
        """Flatten each hit into a citation dict (never raises on missing keys)."""
        citations: list[dict] = []
        for i, r in enumerate(results, start=1):
            citation = {
                "rank": i,
                "id": r.id,
                "score": round(float(r.score), 4),
                "source_type": r.source_type,
                "preview": (r.content or "")[:200],
            }
            # Pass through whatever metadata the host attached (source path,
            # upload date, version, ...) without requiring any of it.
            for key, value in (r.metadata or {}).items():
                citation.setdefault(key, value)
            citations.append(citation)
        return citations
