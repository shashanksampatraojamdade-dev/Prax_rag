"""Technique 15 — Multi-query retrieval.

A single phrasing of a question only matches documents written the same way.
Multi-query retrieval asks the language model for several *alternative
phrasings* of the question; the host runs a search for each and merges the
results, catching relevant documents that any one phrasing would miss.

Example::

    "How do I lower server latency?"
      -> ["Ways to optimize server response times",
          "Reducing network lag in databases",
          "Best practices for server performance tuning"]

The SDK only generates the variations; the host embeds each and merges the hits.
Text generation is delegated to the injected
:class:`~prax_qt.interfaces.LanguageModel`.
"""

from __future__ import annotations

import logging
import re

from prax_qt.interfaces import LanguageModel

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You generate alternative phrasings of a search question for a document "
    "retrieval system. Produce diverse rewordings that approach the question "
    "from different angles, vocabularies and levels of specificity, so that "
    "together they match more relevant documents. Respond with one phrasing per "
    "line, no numbering, quotes, preamble or explanation. Do not answer the "
    "question."
)


def generate_query_variations(
    model: LanguageModel, query: str, count: int = 3
) -> list[str]:
    """Return up to ``count`` alternative phrasings of ``query``.

    Returns an empty list (never raises) if the model fails or yields nothing,
    so multi-query retrieval degrades gracefully to "just the base query".
    """
    if count <= 0:
        return []
    prompt = f"Generate {count} alternative phrasings of this question:\n{query}"
    logger.debug("Multi-query prompt | query=%r count=%d", query, count)
    try:
        raw = model.complete(prompt, system=_SYSTEM)
    except Exception:  # pragma: no cover - defensive; never block retrieval
        logger.exception("Multi-query generation failed; using base query only")
        return []

    logger.debug("Multi-query raw model output=%r", raw)
    variations = _parse_lines(raw, original=query, limit=count)
    if variations:
        logger.info(
            "Generated %d query variation(s) for %r: %s",
            len(variations),
            query,
            variations,
        )
    else:
        logger.info("Multi-query produced no usable variations for %r", query)
    return variations


def _parse_lines(text: str, original: str, limit: int) -> list[str]:
    """Split a multi-line model response into clean, unique variation strings."""
    if not text:
        return []
    seen: set[str] = set()
    original_norm = original.strip().lower()
    variations: list[str] = []
    for line in text.splitlines():
        item = line.strip().strip('"').strip("'").strip("-•").strip()
        # Drop list numbering like "1." or "2)" a model may leak.
        item = re.sub(r"^\d+[.)]\s*", "", item).strip()
        if not item:
            continue
        norm = item.lower()
        if norm == original_norm or norm in seen:
            continue
        seen.add(norm)
        variations.append(item)
        if len(variations) >= limit:
            break
    return variations
