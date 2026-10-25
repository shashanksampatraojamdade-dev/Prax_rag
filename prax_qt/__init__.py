"""Prax QT — a plug-and-play query understanding & transformation SDK for RAG.

Bring your own language model; the SDK reshapes each incoming query before
retrieval using eight techniques:

- **Query rewriting** — clarify vague, conversational questions.
- **Query expansion** — add synonyms, expanded acronyms and related terms.
- **Multi-query retrieval** — generate alternative phrasings to search in parallel.
- **HyDE** — draft a hypothetical answer and search for documents like it.
- **Query classification** — exact-lookup / semantic / analytical / multi-hop.
- **Adaptive routing** — pick the best retrieval target for the query.
- **Intent detection** — the answer format wanted (fact / summary / comparison / ...).
- **Multi-hop planning** — decompose chained questions into ordered sub-queries.

Techniques that widen retrieval (multi-query, HyDE, multi-hop) emit extra
*probes* in ``TransformedQuery.search_queries``; the host embeds each, searches,
and merges the hits. Classification, routing and intent are advisory metadata.

Quick start::

    from prax_qt import QueryTransformer, QTConfig

    qt = QueryTransformer(model=my_llm, config=QTConfig())
    result = qt.transform("How do I fix that login error?")
    hits = my_retriever(result.search_query)   # embed & search this
    answer = my_llm_answer(question=result.original, context=hits)

The SDK is backend-agnostic: it depends only on the :class:`LanguageModel`
protocol, never on a specific model client.
"""

from prax_qt.config import QTConfig
from prax_qt.interfaces import LanguageModel
from prax_qt.pipeline import QueryTransformer, TransformedQuery

__all__ = [
    "QueryTransformer",
    "TransformedQuery",
    "QTConfig",
    "LanguageModel",
]

__version__ = "0.4.0"
