"""Technique 14 — Query expansion.

A single phrasing can miss relevant chunks that use synonyms, acronyms or
related terminology. Query expansion asks the language model for a handful of
closely-related terms and appends them to the search query so no match is
missed.

Example::

    "US passport rules"
      -> ["United States passport renewal requirements",
          "travel visa documentation", "State Department guidelines"]

The SDK only orchestrates; text generation is delegated to the injected
:class:`~prax_qt.interfaces.LanguageModel`.
"""

from __future__ import annotations

import logging
import re

from prax_qt.interfaces import LanguageModel

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You expand search queries for a document retrieval system. Given a query, "
    "list closely related search terms, synonyms, expanded acronyms and "
    "adjacent concepts that help find relevant documents. Respond with a single "
    "line of comma-separated terms only — no numbering, preamble or "
    "explanation. Do not repeat the original query verbatim."
)


def expand_query(
    model: LanguageModel, query: str, max_terms: int = 6
) -> list[str]:
    """Return up to ``max_terms`` related terms for ``query``.

    Returns an empty list (never raises) if the model fails or yields nothing,
    so expansion degrades gracefully to "no extra terms".
    """
    if max_terms <= 0:
        logger.debug("Query expansion skipped (max_terms<=0)")
        return []
    prompt = f"Expand this query into related search terms:\n{query}"
    logger.debug("Expansion prompt | query=%r max_terms=%d", query, max_terms)
    try:
        raw = model.complete(prompt, system=_SYSTEM)
    except Exception:  # pragma: no cover - defensive; never block retrieval
        logger.exception("Query expansion failed; using no extra terms")
        return []

    logger.debug("Expansion raw model output=%r", raw)
    terms = _parse_terms(raw, original=query, limit=max_terms)
    if terms:
        logger.info("Expanded query %r with %d term(s): %s", query, len(terms), terms)
    else:
        logger.info("Query expansion produced no usable terms for %r", query)
    return terms


def _parse_terms(text: str, original: str, limit: int) -> list[str]:
    """Split a comma/newline separated model response into clean, unique terms."""
    if not text:
        return []
    raw_parts = re.split(r"[,\n;]+", text)
    seen: set[str] = set()
    original_norm = original.strip().lower()
    terms: list[str] = []
    for part in raw_parts:
        term = part.strip().strip('"').strip("'").strip("-•").strip()
        # Drop list numbering like "1." or "2)" a model may leak.
        term = re.sub(r"^\d+[.)]\s*", "", term).strip()
        if not term:
            continue
        norm = term.lower()
        if norm == original_norm or norm in seen:
            continue
        seen.add(norm)
        terms.append(term)
        if len(terms) >= limit:
            break
    return terms
