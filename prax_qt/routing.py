"""Technique 22 — Adaptive query routing.

Given the query (and its classification), pick the best retrieval target or
strategy from the set the host actually supports. The SDK stays backend-agnostic:
the host declares its candidate routes in ``QTConfig.routes`` (e.g.
``["knowledge_base", "sql_directory", "zendesk"]``) and the SDK returns the
single best-fit route name; the host decides what that route *does*.

Example::

    "What is the phone number for Jane Doe?"  -> route "sql_directory"
    "How do I reset my VPN?"                  -> route "zendesk"

Text generation is delegated to the injected
:class:`~prax_qt.interfaces.LanguageModel`.
"""

from __future__ import annotations

import logging

from prax_qt.interfaces import LanguageModel

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a routing layer for a retrieval system. Given a user query, its "
    "category, and a list of available retrieval routes, choose the single best "
    "route to answer it. Respond with the exact route name only — no "
    "punctuation or explanation."
)


def route_query(
    model: LanguageModel,
    query: str,
    query_class: str | None,
    routes: list[str],
) -> str | None:
    """Return the best-fit route from ``routes`` for ``query``.

    Short-circuits when there is nothing to choose (0 or 1 route) and falls back
    to the first route on any ambiguity or model failure, so routing never
    blocks retrieval.
    """
    if not routes:
        logger.debug("Routing skipped (no candidate routes configured)")
        return None
    if len(routes) == 1:
        logger.info("Routing to only candidate route: %s", routes[0])
        return routes[0]

    options = ", ".join(routes)
    prompt = (
        f"Query: {query}\nCategory: {query_class or 'unknown'}\n"
        f"Available routes: {options}\nBest route:"
    )
    logger.debug("Routing prompt | query=%r routes=%s", query, routes)
    try:
        raw = model.complete(prompt, system=_SYSTEM)
    except Exception:  # pragma: no cover - defensive; never block retrieval
        logger.exception("Routing failed; defaulting to %s", routes[0])
        return routes[0]

    logger.debug("Routing raw model output=%r", raw)
    route = _match(raw, routes)
    logger.info("Routed query %r (class=%s) to %s", query, query_class, route)
    return route


def _match(text: str, routes: list[str]) -> str:
    """Map a free-form model response onto a configured route name."""
    norm = (text or "").strip().lower()
    for route in routes:
        if route.lower() in norm:
            return route
    return routes[0]
