"""Technique 21 — Query classification.

Different questions need different retrieval strategies. Classifying a query up
front lets the pipeline (and the host) adapt: an exact-record lookup wants a
precise ID match, an analytical question wants a broad multi-source summary, a
multi-hop question needs a step-by-step plan.

Categories
----------
- ``exact_lookup``  — a specific record / ID (e.g. "Show me invoice #4059").
- ``semantic``      — a conceptual question answered by similar passages.
- ``analytical``    — a summary / reasoning over many sources ("Why did sales drop?").
- ``multi_hop``     — needs chained lookups ("manager of the person who wrote X").

Text generation is delegated to the injected
:class:`~prax_qt.interfaces.LanguageModel`.
"""

from __future__ import annotations

import logging

from prax_qt.interfaces import LanguageModel

logger = logging.getLogger(__name__)

CATEGORIES = ("exact_lookup", "semantic", "analytical", "multi_hop")
_DEFAULT = "semantic"

_SYSTEM = (
    "You classify search queries for a retrieval system into exactly one "
    "category:\n"
    "- exact_lookup: asks for one specific record, ID, number or named entity.\n"
    "- semantic: a conceptual question answered by finding similar passages.\n"
    "- analytical: needs a summary or reasoning over multiple sources.\n"
    "- multi_hop: needs several chained lookups where one answer feeds the next.\n"
    "Respond with the single category word only — no punctuation or explanation."
)


def classify_query(model: LanguageModel, query: str) -> str:
    """Return one of :data:`CATEGORIES` for ``query`` (defaults to ``semantic``)."""
    prompt = f"Classify this query:\n{query}"
    logger.debug("Classification prompt | query=%r", query)
    try:
        raw = model.complete(prompt, system=_SYSTEM)
    except Exception:  # pragma: no cover - defensive; never block retrieval
        logger.exception("Query classification failed; defaulting to %s", _DEFAULT)
        return _DEFAULT

    logger.debug("Classification raw model output=%r", raw)
    label = _match(raw)
    logger.info("Classified query %r as %s", query, label)
    return label


def _match(text: str) -> str:
    """Map a free-form model response onto a known category."""
    norm = (text or "").strip().lower()
    for category in CATEGORIES:
        if category in norm:
            return category
    return _DEFAULT
