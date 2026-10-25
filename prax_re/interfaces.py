"""Integration contracts for the Prax retrieval-engine SDK.

``prax_re`` is backend-agnostic: it never imports your database, embedder or
LLM. Instead you inject small callables/objects that satisfy these protocols,
and the engine orchestrates hybrid retrieval, fusion, confidence gating and
answer generation on top of them.

The only data type the SDK owns is :class:`Retrieved` — a lightweight,
metadata-carrying search hit. ``metadata`` is a free-form dict so a rich KB can
attach source path, upload date and version while a lean KB can attach nothing;
the engine never *requires* any particular key.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Hashable, Protocol, runtime_checkable


@dataclass
class Retrieved:
    """One search hit from a retriever.

    - ``id`` uniquely identifies the underlying chunk so fusion can dedupe hits
      that both retrievers returned.
    - ``score`` is the retriever's own relevance, normalised to 0-1 where higher
      is better (semantic: ``1 - cosine_distance``; keyword: a normalised lexical
      rank). Used for confidence; fusion itself is rank-based so exact scales do
      not need to match.
    - ``source_type`` records which retriever produced it (``"semantic"`` /
      ``"keyword"``), or ``"hybrid"`` once fused from both.
    - ``metadata`` is whatever the host wants to carry through to citations.
    """

    id: Hashable
    content: str
    score: float = 0.0
    source_type: str = "semantic"
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Retriever(Protocol):
    """Returns the top-``k`` hits for a query string."""

    def search(self, query: str, k: int) -> list[Retrieved]: ...


@runtime_checkable
class AnswerModel(Protocol):
    """Generates a grounded answer from a question and assembled context."""

    def generate(self, question: str, context: str) -> str: ...


@runtime_checkable
class StreamingAnswerModel(AnswerModel, Protocol):
    """An :class:`AnswerModel` that can also stream tokens."""

    def generate_stream(self, question: str, context: str) -> Iterator[str]: ...
