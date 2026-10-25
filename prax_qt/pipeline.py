"""The Prax query-transformation pipeline — the plug-and-play entry point.

Compose it with any :class:`~prax_qt.interfaces.LanguageModel`::

    from prax_qt import QueryTransformer, QTConfig

    qt = QueryTransformer(model=my_llm, config=QTConfig())
    result = qt.transform("How do I fix that login error?")
    retrieved = my_retriever(result.search_query)   # embed/search this
    answer = my_llm_answer(question=result.original, context=retrieved)

The pipeline owns every query-understanding technique (rewriting, expansion)
and calls back into the injected model. Enable the SDK and every enabled method
is applied, in order, to each incoming query. The *original* question is always
preserved so the host can still show the user their words and ground the final
answer on their real intent.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
from time import perf_counter

from prax_qt.classification import classify_query
from prax_qt.config import QTConfig
from prax_qt.expansion import expand_query
from prax_qt.hyde import generate_hypothetical_document
from prax_qt.intent import detect_intent
from prax_qt.interfaces import LanguageModel
from prax_qt.multi_hop import plan_hops
from prax_qt.multi_query import generate_query_variations
from prax_qt.planner import plan_query
from prax_qt.routing import route_query
from prax_qt.rewriting import rewrite_query

logger = logging.getLogger(__name__)


# --- Accuracy strategy: class-driven technique selection -------------------- #
#
# Not every technique helps every query. Running the wrong ones *lowers*
# accuracy (e.g. query expansion / HyDE inject synonyms that pull irrelevant
# passages for an exact-ID lookup) and wastes latency. We therefore let the
# query *class* decide which retrieval-shaping techniques fire. Rewriting,
# routing and intent detection are always useful and are not gated here.
#
#   exact_lookup → none: keep the precise rewritten query, add no synonym noise.
#   semantic     → expansion + multi-query + HyDE: broaden to find similar text.
#   analytical   → expansion + multi-query + HyDE: gather many supporting sources.
#   multi_hop    → multi-query + multi-hop planning: chain the sub-questions.
_CLASS_TECHNIQUES: dict[str, set[str]] = {
    "exact_lookup": set(),
    "semantic": {"expansion", "multi_query", "hyde"},
    "analytical": {"expansion", "multi_query", "hyde"},
    "multi_hop": {"expansion", "multi_query", "multi_hop"},
}
_ALL_TECHNIQUES = {"expansion", "multi_query", "hyde", "multi_hop"}


def _relevant_techniques(cfg: QTConfig, query_class: str | None) -> set[str]:
    """Which retrieval-shaping techniques are worth running for this class.

    When classification is off or inconclusive we cannot gate, so every enabled
    technique is allowed (the previous behaviour).
    """
    if query_class is None or not cfg.enable_classification:
        return set(_ALL_TECHNIQUES)
    return set(_CLASS_TECHNIQUES.get(query_class, _ALL_TECHNIQUES))


def _ms(start: float) -> float:
    return round((perf_counter() - start) * 1000, 1)


@dataclass
class TransformedQuery:
    """The result of applying the enabled techniques to one query.

    - ``search_query`` is the single primary query (rewritten + expansion terms).
    - ``search_queries`` is the full set of probes the host should embed, search
      and merge — the primary query plus multi-query variations, the HyDE
      document and multi-hop sub-queries when those techniques run.
    - ``query_class``, ``route`` and ``intent`` are advisory metadata the host
      can use to adapt retrieval/generation; ``hops`` is the multi-hop plan.
    """

    original: str
    rewritten: str
    expansions: list[str] = field(default_factory=list)
    search_query: str = ""
    variations: list[str] = field(default_factory=list)
    hypothetical_document: str | None = None
    hops: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    query_class: str | None = None
    route: str | None = None
    intent: str | None = None
    applied: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "original": self.original,
            "rewritten": self.rewritten,
            "expansions": self.expansions,
            "search_query": self.search_query,
            "variations": self.variations,
            "hypothetical_document": self.hypothetical_document,
            "hops": self.hops,
            "search_queries": self.search_queries,
            "query_class": self.query_class,
            "route": self.route,
            "intent": self.intent,
            "applied": self.applied,
        }


class QueryTransformer:
    """Backend-agnostic query understanding & transformation pipeline."""

    def __init__(
        self,
        model: LanguageModel,
        config: QTConfig | None = None,
    ) -> None:
        self._model = model
        self._config = config or QTConfig()

    def transform(self, query: str) -> TransformedQuery:
        """Transform a query, preferring the fast single-call path.

        When ``single_call`` is enabled, all understanding techniques run in one
        structured LLM call; if that fails to parse we fall back to the
        per-technique pipeline (which runs independent techniques concurrently)
        so behaviour is never lost.
        """
        if self._config.single_call:
            result = self._transform_single_call(query)
            if result is not None:
                return result
            logger.info("QT falling back to per-technique pipeline")
        return self._transform_per_technique(query)

    def _transform_single_call(self, query: str) -> TransformedQuery | None:
        cfg = self._config
        original = query or ""
        if not original.strip():
            return None
        t_start = perf_counter()
        logger.info("QT pipeline start (single-call) | query=%r", original)
        plan = plan_query(self._model, original, cfg)
        if plan is None:
            return None

        rewritten = plan.get("rewritten") or original.strip()
        expansions = plan.get("expansions", [])
        variations = plan.get("variations", [])
        hypothetical_document = plan.get("hypothetical_document")
        hops = plan.get("hops", [])
        query_class = plan.get("classification")
        intent = plan.get("intent")
        route = plan.get("route")

        # Accuracy gate: drop retrieval-shaping outputs that do not help this
        # query class (e.g. synonyms/HyDE on an exact-ID lookup add only noise).
        relevant = _relevant_techniques(cfg, query_class)
        gated: list[str] = []
        if "expansion" not in relevant and expansions:
            expansions = []
            gated.append("expansion")
        if "multi_query" not in relevant and variations:
            variations = []
            gated.append("multi_query")
        if "hyde" not in relevant and hypothetical_document:
            hypothetical_document = None
            gated.append("hyde")
        if "multi_hop" not in relevant and hops:
            hops = []
            gated.append("multi_hop")
        if gated:
            logger.info(
                "QT class=%s gated out %s (not useful for this query class)",
                query_class,
                gated,
            )

        applied: list[str] = []
        if query_class is not None:
            applied.append("classification")
        if intent is not None:
            applied.append("intent_detection")
        if route is not None:
            applied.append("routing")
        if cfg.enable_rewriting and rewritten and rewritten != original.strip():
            applied.append("rewriting")
        if expansions:
            applied.append("expansion")
        if variations:
            applied.append("multi_query")
        if hypothetical_document:
            applied.append("hyde")
        if hops:
            applied.append("multi_hop")

        search_query = self._build_search_query(rewritten, expansions)
        search_queries = self._build_search_queries(
            search_query, variations, hypothetical_document, hops
        )
        logger.info(
            "QT pipeline done (single-call) in %.1fms | applied=%s | class=%s "
            "intent=%s route=%s | probes=%d | search_query=%r",
            _ms(t_start),
            applied or ["none"],
            query_class,
            intent,
            route,
            len(search_queries),
            search_query,
        )
        return TransformedQuery(
            original=original,
            rewritten=rewritten,
            expansions=expansions,
            search_query=search_query,
            variations=variations,
            hypothetical_document=hypothetical_document,
            hops=hops,
            search_queries=search_queries,
            query_class=query_class,
            route=route,
            intent=intent,
            applied=applied,
        )

    def _transform_per_technique(self, query: str) -> TransformedQuery:
        """Per-technique path with independent techniques run concurrently.

        The techniques form a small dependency graph, so we execute them in two
        concurrent "waves" via a thread pool (the LLM calls are I/O-bound):

        * **Wave 1** depends only on the original query — classification,
          intent detection and rewriting run in parallel.
        * **Wave 2** depends on wave 1 — routing (needs the class), expansion,
          multi-query, HyDE and multi-hop planning (needs the rewritten query
          and the class gate) all run in parallel.

        This collapses ~8 sequential round-trips into two, while preserving the
        exact same results and ``applied`` ordering as the sequential version.
        Set ``QTConfig.max_workers = 1`` to force strictly-sequential execution.
        """
        cfg = self._config
        original = query or ""
        base_query = original.strip()
        t_start = perf_counter()
        logger.info(
            "QT pipeline start (parallel, max_workers=%d) | query=%r | "
            "classification=%s intent=%s routing=%s rewriting=%s expansion=%s "
            "multi_query=%s hyde=%s multi_hop=%s",
            cfg.max_workers,
            original,
            cfg.enable_classification,
            cfg.enable_intent_detection,
            cfg.enable_routing,
            cfg.enable_rewriting,
            cfg.enable_expansion,
            cfg.enable_multi_query,
            cfg.enable_hyde,
            cfg.enable_multi_hop,
        )

        with ThreadPoolExecutor(max_workers=max(1, cfg.max_workers)) as pool:
            # --- Wave 1: techniques that depend only on the original query --- #
            w1: dict[str, object] = {}
            if cfg.enable_classification and base_query:
                logger.info("QT applying technique 21: query classification")
                w1["classification"] = pool.submit(
                    self._timed, "classification", classify_query,
                    self._model, base_query,
                )
            else:
                logger.info(
                    "QT skipping technique 21: query classification (disabled)"
                )
            if cfg.enable_intent_detection and base_query:
                logger.info("QT applying technique 23: query intent detection")
                w1["intent"] = pool.submit(
                    self._timed, "intent_detection", detect_intent,
                    self._model, base_query,
                )
            else:
                logger.info(
                    "QT skipping technique 23: query intent detection (disabled)"
                )
            if cfg.enable_rewriting and base_query:
                logger.info("QT applying technique 13: query rewriting")
                w1["rewriting"] = pool.submit(
                    self._timed, "rewriting", rewrite_query,
                    self._model, base_query,
                )
            else:
                logger.info("QT skipping technique 13: query rewriting (disabled)")

            query_class = w1["classification"].result() if "classification" in w1 else None
            intent = w1["intent"].result() if "intent" in w1 else None
            rewritten = base_query
            rewriting_changed = False
            if "rewriting" in w1:
                new = w1["rewriting"].result()
                if new and new != base_query:
                    rewritten = new
                    rewriting_changed = True

            # --- Wave 2: techniques that depend on wave 1 results ----------- #
            # Classification now decides which retrieval-shaping techniques are
            # worth running: skipping the useless ones both raises accuracy and
            # cuts latency (fewer LLM calls for simple queries).
            relevant = _relevant_techniques(cfg, query_class)
            w2: dict[str, object] = {}
            if cfg.enable_routing and rewritten:
                logger.info("QT applying technique 22: adaptive query routing")
                w2["routing"] = pool.submit(
                    self._timed, "routing", route_query,
                    self._model, rewritten, query_class, cfg.routes,
                )
            else:
                logger.info(
                    "QT skipping technique 22: adaptive query routing (disabled)"
                )
            if cfg.enable_expansion and rewritten and "expansion" in relevant:
                logger.info("QT applying technique 14: query expansion")
                base = rewritten if cfg.expand_rewritten else original
                w2["expansion"] = pool.submit(
                    self._timed, "expansion", expand_query,
                    self._model, base, cfg.max_expansion_terms,
                )
            elif cfg.enable_expansion and rewritten:
                logger.info(
                    "QT skipping technique 14: query expansion (class=%s)", query_class
                )
            else:
                logger.info("QT skipping technique 14: query expansion (disabled)")
            if cfg.enable_multi_query and rewritten and "multi_query" in relevant:
                logger.info("QT applying technique 15: multi-query retrieval")
                w2["multi_query"] = pool.submit(
                    self._timed, "multi_query", generate_query_variations,
                    self._model, rewritten, cfg.multi_query_count,
                )
            elif cfg.enable_multi_query and rewritten:
                logger.info(
                    "QT skipping technique 15: multi-query retrieval (class=%s)",
                    query_class,
                )
            else:
                logger.info(
                    "QT skipping technique 15: multi-query retrieval (disabled)"
                )
            if cfg.enable_hyde and rewritten and "hyde" in relevant:
                logger.info("QT applying technique 16: HyDE")
                w2["hyde"] = pool.submit(
                    self._timed, "hyde", generate_hypothetical_document,
                    self._model, rewritten,
                )
            elif cfg.enable_hyde and rewritten:
                logger.info("QT skipping technique 16: HyDE (class=%s)", query_class)
            else:
                logger.info("QT skipping technique 16: HyDE (disabled)")
            # Classification gates multi-hop: only a multi_hop query needs a plan.
            if cfg.enable_multi_hop and rewritten and "multi_hop" in relevant:
                logger.info("QT applying technique 24: multi-hop retrieval planning")
                w2["multi_hop"] = pool.submit(
                    self._timed, "multi_hop", plan_hops,
                    self._model, rewritten, cfg.max_hops,
                )
            elif cfg.enable_multi_hop and rewritten:
                logger.info(
                    "QT skipping technique 24: multi-hop (class=%s is not multi_hop)",
                    query_class,
                )
            else:
                logger.info(
                    "QT skipping technique 24: multi-hop retrieval planning (disabled)"
                )

            route = w2["routing"].result() if "routing" in w2 else None
            expansions = w2["expansion"].result() if "expansion" in w2 else []
            variations = w2["multi_query"].result() if "multi_query" in w2 else []
            hypothetical_document = w2["hyde"].result() if "hyde" in w2 else None
            hops = w2["multi_hop"].result() if "multi_hop" in w2 else []

        # Assemble results in the canonical technique order so ``applied`` reads
        # the same as the sequential pipeline regardless of completion timing.
        applied: list[str] = []
        if query_class is not None:
            applied.append("classification")
        if intent is not None:
            applied.append("intent_detection")
        if route:
            applied.append("routing")
        if rewriting_changed:
            applied.append("rewriting")
        if expansions:
            applied.append("expansion")
        if variations:
            applied.append("multi_query")
        if hypothetical_document:
            applied.append("hyde")
        if hops:
            applied.append("multi_hop")

        search_query = self._build_search_query(rewritten, expansions)
        search_queries = self._build_search_queries(
            search_query, variations, hypothetical_document, hops
        )
        logger.info(
            "QT pipeline done (parallel) in %.1fms | applied=%s | class=%s "
            "intent=%s route=%s | probes=%d | search_query=%r",
            _ms(t_start),
            applied or ["none"],
            query_class,
            intent,
            route,
            len(search_queries),
            search_query,
        )
        return TransformedQuery(
            original=original,
            rewritten=rewritten,
            expansions=expansions,
            search_query=search_query,
            variations=variations,
            hypothetical_document=hypothetical_document,
            hops=hops,
            search_queries=search_queries,
            query_class=query_class,
            route=route,
            intent=intent,
            applied=applied,
        )

    @staticmethod
    def _timed(label: str, fn, *args):
        """Run ``fn(*args)`` and log how long the technique took."""
        t0 = perf_counter()
        result = fn(*args)
        logger.info("QT technique %s done in %.1fms", label, _ms(t0))
        return result

    @staticmethod
    def _build_search_query(rewritten: str, expansions: list[str]) -> str:
        parts = [rewritten] if rewritten else []
        parts.extend(expansions)
        return " ".join(p for p in parts if p).strip()

    @staticmethod
    def _build_search_queries(
        search_query: str,
        variations: list[str],
        hypothetical_document: str | None,
        hops: list[str],
    ) -> list[str]:
        """Assemble the de-duplicated list of probes the host should retrieve on."""
        probes: list[str] = []
        seen: set[str] = set()
        for probe in [search_query, *variations, hypothetical_document, *hops]:
            if not probe:
                continue
            norm = probe.strip().lower()
            if norm in seen:
                continue
            seen.add(norm)
            probes.append(probe)
        return probes
