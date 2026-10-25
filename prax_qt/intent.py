"""Technique 23 — Query intent detection.

Detects the *format* of answer the user wants, so the host can shape retrieval
and generation accordingly — pull structured comparison data for a "vs"
question, a single value for a fact lookup, or broader context for an
explanation.

Intents
-------
- ``fact``        — a single direct value ("What is the SLA uptime target?").
- ``summary``     — a brief overview of a topic.
- ``explanation`` — a "how" / "why" that needs reasoning and context.
- ``comparison``  — weigh two or more options ("Mac vs Windows for programming").
- ``list``        — an enumeration of items / steps.
- ``other``       — none of the above.

Text generation is delegated to the injected
:class:`~prax_qt.interfaces.LanguageModel`.
"""

from __future__ import annotations

import logging

from prax_qt.interfaces import LanguageModel

logger = logging.getLogger(__name__)

INTENTS = ("fact", "summary", "explanation", "comparison", "list", "other")
_DEFAULT = "other"

_SYSTEM = (
    "You detect what format of answer a user wants from their query, as exactly "
    "one label:\n"
    "- fact: a single direct value.\n"
    "- summary: a brief overview.\n"
    "- explanation: a how/why needing reasoning.\n"
    "- comparison: weighing two or more options.\n"
    "- list: an enumeration of items or steps.\n"
    "- other: none of the above.\n"
    "Respond with the single label word only — no punctuation or explanation."
)


def detect_intent(model: LanguageModel, query: str) -> str:
    """Return one of :data:`INTENTS` for ``query`` (defaults to ``other``)."""
    prompt = f"Detect the answer intent of this query:\n{query}"
    logger.debug("Intent prompt | query=%r", query)
    try:
        raw = model.complete(prompt, system=_SYSTEM)
    except Exception:  # pragma: no cover - defensive; never block retrieval
        logger.exception("Intent detection failed; defaulting to %s", _DEFAULT)
        return _DEFAULT

    logger.debug("Intent raw model output=%r", raw)
    label = _match(raw)
    logger.info("Detected intent of query %r as %s", query, label)
    return label


def _match(text: str) -> str:
    """Map a free-form model response onto a known intent."""
    norm = (text or "").strip().lower()
    for intent in INTENTS:
        if intent in norm:
            return intent
    return _DEFAULT
