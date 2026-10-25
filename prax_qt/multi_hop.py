"""Technique 24 — Multi-hop retrieval planning.

Some questions can only be answered by chaining lookups, where each step depends
on the previous answer::

    "Who is the manager of the person who wrote the security policy?"
      Hop 1: find the security policy's author.
      Hop 2: find that author's manager in the org chart.

This module asks the language model to decompose such a question into an ordered
list of standalone sub-queries (hops). The host can then retrieve for each hop —
either sequentially (feeding one hop's answer into the next) or, in the simple
case, by adding the hops as extra retrieval probes to widen coverage.

Text generation is delegated to the injected
:class:`~prax_qt.interfaces.LanguageModel`.
"""

from __future__ import annotations

import logging
import re

from prax_qt.interfaces import LanguageModel

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You plan multi-hop retrieval for a question that needs several chained "
    "lookups. Break the question into an ordered list of simple, standalone "
    "sub-queries, where each later sub-query may depend on earlier answers. If "
    "the question needs only one lookup, return just that one. Respond with one "
    "sub-query per line, in order, no numbering, quotes, preamble or explanation."
)


def plan_hops(model: LanguageModel, query: str, max_hops: int = 3) -> list[str]:
    """Return an ordered list of up to ``max_hops`` sub-queries for ``query``.

    Returns an empty list (never raises) if the model fails or yields nothing,
    so multi-hop planning degrades gracefully to single-shot retrieval.
    """
    if max_hops <= 0:
        return []
    prompt = (
        f"Plan the retrieval hops (max {max_hops}) for this question:\n{query}"
    )
    logger.debug("Multi-hop prompt | query=%r max_hops=%d", query, max_hops)
    try:
        raw = model.complete(prompt, system=_SYSTEM)
    except Exception:  # pragma: no cover - defensive; never block retrieval
        logger.exception("Multi-hop planning failed; using single-shot retrieval")
        return []

    logger.debug("Multi-hop raw model output=%r", raw)
    hops = _parse_lines(raw, limit=max_hops)
    if hops:
        logger.info("Planned %d hop(s) for %r: %s", len(hops), query, hops)
    else:
        logger.info("Multi-hop planning produced no usable hops for %r", query)
    return hops


def _parse_lines(text: str, limit: int) -> list[str]:
    """Split a multi-line model response into clean, ordered sub-query strings."""
    if not text:
        return []
    hops: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        item = line.strip().strip('"').strip("'").strip("-•").strip()
        item = re.sub(r"^(hop\s*\d+\s*[:.)-]?\s*)", "", item, flags=re.IGNORECASE)
        item = re.sub(r"^\d+[.)]\s*", "", item).strip()
        if not item:
            continue
        norm = item.lower()
        if norm in seen:
            continue
        seen.add(norm)
        hops.append(item)
        if len(hops) >= limit:
            break
    return hops
