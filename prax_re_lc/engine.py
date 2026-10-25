"""The LangChain-based Prax retrieval engine.

``RetrievalEngine`` is the plug-and-play entry point — a functional clone of
``prax_re.RetrievalEngine`` built on LangChain primitives. Compose it with any
LangChain ``BaseRetriever`` (semantic and/or keyword) and any LangChain chat
model / ``Runnable`` (or a plain ``callable(question, context) -> str``)::

    from prax_re_lc import RetrievalEngine, LCREConfig

    engine = RetrievalEngine(
        answer_model=my_chat_model,     # a LangChain chat model / Runnable
        semantic=my_vector_retriever,   # a LangChain BaseRetriever
        keyword=my_keyword_retriever,   # optional second BaseRetriever
        config=LCREConfig(),
    )
    result = engine.answer("How many vacation days do I get?")

For a zero-wiring start, see ``prax_re_lc.build_engine_from_env`` which builds the
pgvector retrievers + Vertex models for you from environment variables.

The engine owns the reusable logic — hybrid fusion (RRF), confidence scoring, the
"I don't know" threshold, contextual compression and citation/context assembly —
and delegates all backend specifics to the injected LangChain objects.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from prax_re_lc.compression import compress_documents
from prax_re_lc.confidence import Confidence, score_confidence
from prax_re_lc.config import LCREConfig
from prax_re_lc.documents import doc_id, doc_score, set_meta
from prax_re_lc.fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


@dataclass
class AnswerResult:
    answer: str
    confidence: float
    confidence_label: str
    answerable: bool
    citations: list[dict] = field(default_factory=list)
    results: list[Document] = field(default_factory=list)
    retrieval_mode: str = "hybrid"


class RetrievalEngine:
    """Reusable hybrid retrieval + grounded answering engine (LangChain)."""

    def __init__(
        self,
        answer_model: Runnable | Callable[[str, str], str],
        semantic: "object | None" = None,
        keyword: "object | None" = None,
        config: LCREConfig | None = None,
        compressor: "object | None" = None,
    ) -> None:
        self._answer_model = answer_model
        self._semantic = semantic
        self._keyword = keyword
        self._config = config or LCREConfig()
        # Optional LangChain BaseDocumentCompressor; None -> built-in extractive.
        self._compressor = compressor

    # --- Retrieval -------------------------------------------------------- #
    def _invoke_retriever(self, retriever, query: str, source_type: str) -> list[Document]:
        docs: list[Document] = list(retriever.invoke(query))[: self._config.candidate_k]
        for rank, d in enumerate(docs):
            d.metadata.setdefault("id", doc_id(d))
            d.metadata.setdefault("score", doc_score(d, rank))
            set_meta(d, source_type=d.metadata.get("source_type", source_type))
        logger.info("%s retrieval returned %d hit(s)", source_type.title(), len(docs))
        return docs

    def retrieve(self, query: str) -> list[Document]:
        """Run the enabled LangChain retrievers and fuse their hits (hybrid)."""
        cfg = self._config
        lists: list[list[Document]] = []
        modes: list[str] = []
        if cfg.enable_semantic and self._semantic is not None:
            lists.append(self._invoke_retriever(self._semantic, query, "semantic"))
            modes.append("semantic")
        if cfg.enable_keyword and self._keyword is not None:
            lists.append(self._invoke_retriever(self._keyword, query, "keyword"))
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
        """Retrieve, gate on confidence, compress, then generate a grounded answer."""
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
        answer_text = self._generate(query, context)
        return AnswerResult(
            answer=answer_text,
            confidence=conf.score,
            confidence_label=conf.label,
            answerable=True,
            citations=citations,
            results=results,
            retrieval_mode=mode,
        )

    def answer_stream(self, query: str) -> Iterator[str]:
        """Yield answer tokens (or the IDK message) — convenience for streaming."""
        results, conf, _ = self.retrieve_and_gate(query)
        if not conf.answerable:
            yield self._config.idk_message
            return
        context = self.build_context(self.compress(query, results))
        yield from self._generate_stream(query, context)

    def retrieve_and_gate(self, query: str) -> tuple[list[Document], Confidence, str]:
        """Retrieve + score confidence without generating (for streaming hosts)."""
        results = self.retrieve(query)
        conf = score_confidence(results, idk_threshold=self._config.idk_threshold)
        return results, conf, self._mode(results)

    # --- Generation ------------------------------------------------------- #
    def _messages(self, question: str, context: str):
        return [
            SystemMessage(content=self._config.system_prompt),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
        ]

    def _generate(self, question: str, context: str) -> str:
        model = self._answer_model
        if isinstance(model, Runnable):
            result = model.invoke(self._messages(question, context))
            return getattr(result, "content", result)
        if callable(model):
            return model(question, context)
        raise TypeError("answer_model must be a LangChain Runnable or a callable")

    def _generate_stream(self, question: str, context: str) -> Iterator[str]:
        model = self._answer_model
        if isinstance(model, Runnable):
            for chunk in model.stream(self._messages(question, context)):
                text = getattr(chunk, "content", chunk)
                if text:
                    yield text
            return
        gen_stream = getattr(model, "generate_stream", None)
        if callable(gen_stream):
            yield from gen_stream(question, context)
            return
        # Fall back to a single non-streamed generation.
        yield self._generate(question, context)

    # --- Compression ------------------------------------------------------ #
    def compress(self, query: str, results: list[Document]) -> list[Document]:
        """Trim each hit to only its query-relevant sentences (contextual
        compression). Returns ``results`` unchanged when compression is disabled.
        """
        cfg = self._config
        if not cfg.enable_compression or not results:
            return results
        if self._compressor is not None:
            try:
                return list(
                    self._compressor.compress_documents(results, query)  # LC API
                )
            except Exception:
                logger.exception("Injected compressor failed; using built-in")
        return compress_documents(
            query,
            results,
            max_sentences=cfg.compression_max_sentences,
            min_overlap=cfg.compression_min_overlap,
        )

    # --- Helpers ---------------------------------------------------------- #
    @staticmethod
    def _mode(results: list[Document]) -> str:
        types = {r.metadata.get("source_type", "semantic") for r in results}
        if not types:
            return "none"
        if len(types) > 1:
            return "hybrid"
        return next(iter(types))

    @staticmethod
    def build_context(results: list[Document]) -> str:
        """Assemble a numbered context block the model can cite as [1], [2]...."""
        blocks = []
        for i, r in enumerate(results, start=1):
            meta = r.metadata or {}
            label = meta.get("title") or meta.get("source") or f"chunk {doc_id(r)}"
            blocks.append(f"[{i}] ({label})\n{r.page_content}")
        return "\n\n".join(blocks)

    @staticmethod
    def build_citations(results: list[Document]) -> list[dict]:
        """Flatten each hit into a citation dict (never raises on missing keys)."""
        citations: list[dict] = []
        for i, r in enumerate(results, start=1):
            meta = r.metadata or {}
            citation = {
                "rank": i,
                "id": doc_id(r),
                "score": round(float(doc_score(r, i - 1)), 4),
                "source_type": meta.get("source_type", "semantic"),
                "preview": (r.page_content or "")[:200],
            }
            for key, value in meta.items():
                if key in {"uncompressed_content"}:
                    continue
                citation.setdefault(key, value)
            citations.append(citation)
        return citations
