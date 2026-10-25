"""prax_re_lc — the LangChain-based Prax retrieval-engine SDK.

A functional clone of :mod:`prax_re`, re-implemented on LangChain primitives
(``Document``, ``BaseRetriever``, chat ``Runnable``s). Same guarantees:

- **Hybrid retrieval** by default: semantic + keyword fused with Reciprocal Rank
  Fusion.
- **Source citations** for every answer.
- **Configurable "I don't know" threshold**: declines instead of hallucinating.
- **Contextual compression**: trims chunks to the query-relevant sentences.

Two ways to use it:

1. **Plug in your own LangChain objects** (max flexibility)::

       from prax_re_lc import RetrievalEngine, LCREConfig
       engine = RetrievalEngine(answer_model=my_chat_model,
                                semantic=my_retriever, config=LCREConfig())

2. **Batteries included** — build everything from environment variables
   (pgvector + Vertex), zero wiring::

       from prax_re_lc import build_engine_from_env
       engine = build_engine_from_env()           # reads POSTGRES_*, LLM_MODEL, ...
       result = engine.answer("How many vacation days do I get?")

Every parameter is externally settable via ``LCREConfig``; unset values fall back
to sensible defaults.
"""

from prax_re_lc.compression import compress_documents, compress_text
from prax_re_lc.confidence import Confidence, score_confidence
from prax_re_lc.config import LCREConfig
from prax_re_lc.documents import doc_id, doc_score, doc_source_type
from prax_re_lc.engine import AnswerResult, RetrievalEngine
from prax_re_lc.fusion import reciprocal_rank_fusion

__version__ = "0.1.0"

__all__ = [
    "RetrievalEngine",
    "AnswerResult",
    "LCREConfig",
    "Confidence",
    "score_confidence",
    "reciprocal_rank_fusion",
    "compress_text",
    "compress_documents",
    "doc_id",
    "doc_score",
    "doc_source_type",
    "__version__",
]


def __getattr__(name: str):
    # Lazy re-export of the batteries-included builders so importing prax_re_lc
    # never pulls in the heavy optional backends (langchain_postgres, Vertex).
    builders = {
        "build_engine_from_env",
        "build_vertex_llm",
        "build_vertex_embeddings",
        "build_vectorstore",
        "build_semantic_retriever",
        "build_keyword_retriever",
    }
    if name in builders:
        from prax_re_lc import defaults

        return getattr(defaults, name)
    raise AttributeError(f"module 'prax_re_lc' has no attribute '{name}'")
