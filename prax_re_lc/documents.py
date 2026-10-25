"""Helpers for reading the Prax metadata contract off LangChain ``Document``s.

The SDK speaks LangChain's native ``Document`` type end to end. To stay
backend-agnostic we rely on a tiny, optional metadata contract:

- ``metadata['id']``          — stable id so fusion can dedupe (falls back to a
                                content hash when absent).
- ``metadata['score']``       — 0-1 relevance, higher is better. Default
                                retrievers populate it; injected retrievers that
                                don't get a rank-decayed pseudo-score so
                                confidence still works.
- ``metadata['source_type']`` — ``"semantic"`` / ``"keyword"`` / ``"hybrid"``.

None of these are required on *input* documents — only ``page_content`` is.
"""

from __future__ import annotations

import hashlib
from typing import Any

from langchain_core.documents import Document


def doc_id(doc: Document) -> str:
    """Stable identifier for a document (explicit id, else a content hash)."""
    for key in ("id", "chunk_id", "_id"):
        value = doc.metadata.get(key)
        if value is not None:
            return str(value)
    digest = hashlib.sha1((doc.page_content or "").encode("utf-8")).hexdigest()
    return f"sha1:{digest}"


def doc_score(doc: Document, rank: int | None = None) -> float:
    """Relevance score (0-1). Uses ``metadata['score']`` when present.

    When a document carries no score (e.g. a plain injected retriever), fall back
    to a rank-decayed pseudo-score so confidence scoring still discriminates.
    """
    raw = doc.metadata.get("score")
    if raw is not None:
        try:
            return max(0.0, min(1.0, float(raw)))
        except (TypeError, ValueError):
            pass
    if rank is not None:
        return max(0.0, 1.0 - 0.1 * rank)
    return 0.0


def doc_source_type(doc: Document, default: str = "semantic") -> str:
    return str(doc.metadata.get("source_type", default))


def set_meta(doc: Document, **values: Any) -> Document:
    """Return the same document with metadata keys set (mutates in place)."""
    doc.metadata.update({k: v for k, v in values.items() if v is not None})
    return doc
