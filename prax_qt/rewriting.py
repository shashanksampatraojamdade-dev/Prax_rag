"""Technique 13 — Query rewriting.

Vague, ambiguous or conversational questions retrieve poorly because their
wording rarely matches how answers are written in the knowledge base. This
module asks the language model to rewrite the user's question into a single,
explicit, search-friendly statement.

Example::

    "How do I fix that error when logging into the portal?"
      -> "Troubleshooting guide for user login authentication errors on the
          employee portal."

The SDK only orchestrates; the actual text generation is delegated to the
injected :class:`~prax_qt.interfaces.LanguageModel`.
"""

from __future__ import annotations

import logging

from prax_qt.interfaces import LanguageModel

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You rewrite search queries for a document retrieval system. Rewrite the "
    "user's question into one clear, specific, self-contained query optimised "
    "for keyword and semantic search. Resolve vague references, expand "
    "abbreviations, and name the subject explicitly. Do not answer the "
    "question. Respond with the rewritten query only — no preamble, quotes or "
    "explanation."
)


def rewrite_query(model: LanguageModel, query: str) -> str:
    """Return a clearer, search-friendly version of ``query``.

    Falls back to the original query if the model returns nothing usable, so a
    transient model hiccup never blocks retrieval.
    """
    prompt = f"Rewrite this query:\n{query}"
    logger.debug("Rewriting prompt | query=%r", query)
    try:
        raw = model.complete(prompt, system=_SYSTEM)
    except Exception:  # pragma: no cover - defensive; never block retrieval
        logger.exception("Query rewriting failed; using original query")
        return query

    logger.debug("Rewriting raw model output=%r", raw)
    rewritten = _clean(raw)
    if not rewritten:
        logger.warning("Query rewriting returned empty output; using original")
        return query
    logger.info("Rewrote query: %r -> %r", query, rewritten)
    return rewritten


def _clean(text: str) -> str:
    """Normalise a single-line model response into a bare query string."""
    line = (text or "").strip()
    # Models sometimes return a quoted string or a "Query:" label.
    for prefix in ("rewritten query:", "query:", "search query:"):
        if line.lower().startswith(prefix):
            line = line[len(prefix):].strip()
    return line.strip().strip('"').strip("'").strip()
