"""Technique 16 — HyDE (Hypothetical Document Embeddings).

Short questions and real answer passages live in different "regions" of the
embedding space — a terse query rarely looks like the paragraph that answers it.
HyDE bridges that gap: the language model drafts a *hypothetical* ideal answer,
and the host embeds **that passage** (not the raw question) to search for real
documents whose tone and technical phrasing match it.

Example::

    Question: "Why do I get error X903 when syncing?"
      -> hypothetical answer paragraph describing likely causes & fixes
      -> embed the paragraph -> retrieve real troubleshooting docs that read
         like it.

The generated document may be factually wrong — that is fine; it is used only as
a *search probe*, never shown to the user. Text generation is delegated to the
injected :class:`~prax_qt.interfaces.LanguageModel`.
"""

from __future__ import annotations

import logging

from prax_qt.interfaces import LanguageModel

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You write a short, hypothetical passage that would plausibly answer the "
    "user's question, as if it were an excerpt from an internal company "
    "document or knowledge-base article. Write 2-4 sentences in a factual, "
    "technical tone. It does not need to be correct — it is used only to search "
    "for similar real documents. Respond with the passage only, no preamble, "
    "headings or disclaimers."
)


def generate_hypothetical_document(model: LanguageModel, query: str) -> str | None:
    """Return a hypothetical answer passage for ``query``, or ``None``.

    Returns ``None`` (never raises) on model failure or empty output, so HyDE
    degrades gracefully to not contributing a probe.
    """
    prompt = f"Write a hypothetical answer passage for this question:\n{query}"
    logger.debug("HyDE prompt | query=%r", query)
    try:
        raw = model.complete(prompt, system=_SYSTEM)
    except Exception:  # pragma: no cover - defensive; never block retrieval
        logger.exception("HyDE generation failed; skipping hypothetical document")
        return None

    logger.debug("HyDE raw model output=%r", raw)
    doc = (raw or "").strip()
    if not doc:
        logger.info("HyDE produced no usable document for %r", query)
        return None
    logger.info("Generated HyDE document (%d chars) for %r", len(doc), query)
    return doc
