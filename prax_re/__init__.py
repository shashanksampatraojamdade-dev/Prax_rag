"""prax_re — the Prax retrieval-engine SDK.

A reusable, versioned, backend-agnostic retrieval/answer engine:

- **Hybrid retrieval** by default: semantic (vector) + keyword (lexical) results
  fused with Reciprocal Rank Fusion.
- **Source citations** assembled for every answer from free-form hit metadata.
- **Configurable "I don't know" threshold**: declines instead of hallucinating
  when retrieval confidence is too low.
- **Plug-and-play**: inject any retrievers + answer model satisfying the
  protocols in :mod:`prax_re.interfaces`; the SDK imports nothing app-specific.

Example::

    from prax_re import RetrievalEngine, REConfig, Retrieved

    engine = RetrievalEngine(
        answer_model=my_llm,
        semantic=my_vector_retriever,
        keyword=my_keyword_retriever,
        config=REConfig(idk_threshold=0.35),
    )
    result = engine.answer("How many vacation days do I get?")
"""

from prax_re.compression import compress_content, compress_results
from prax_re.confidence import Confidence, score_confidence
from prax_re.config import REConfig
from prax_re.engine import AnswerResult, RetrievalEngine
from prax_re.fusion import reciprocal_rank_fusion
from prax_re.interfaces import (
    AnswerModel,
    Retrieved,
    Retriever,
    StreamingAnswerModel,
)

__version__ = "0.2.0"

__all__ = [
    "RetrievalEngine",
    "AnswerResult",
    "REConfig",
    "Retrieved",
    "Retriever",
    "AnswerModel",
    "StreamingAnswerModel",
    "Confidence",
    "score_confidence",
    "reciprocal_rank_fusion",
    "compress_content",
    "compress_results",
    "__version__",
]
