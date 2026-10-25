"""Fast path — consolidated single-call query planner.

Running every technique as its own LLM round-trip is accurate but slow (8
sequential calls per query). This module does all of the query *understanding*
(classification, intent, rewriting, expansion, multi-query variations, HyDE,
multi-hop hops) in **one** structured JSON call, cutting latency by ~8x while
keeping the same outputs.

Only the fields for *enabled* techniques are requested, and the model is told to
leave the rest empty. The call is **fail-safe**: on any model or parse error it
returns ``None`` so the caller can fall back to the slower per-technique path.
"""

from __future__ import annotations

import json
import logging
import re

from prax_qt.classification import CATEGORIES
from prax_qt.config import QTConfig
from prax_qt.intent import INTENTS
from prax_qt.interfaces import LanguageModel

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a query-understanding engine for a document retrieval system. You "
    "analyse ONE user query and return a SINGLE JSON object. Do not answer the "
    "query. Only output valid JSON — no markdown, comments or prose."
)


def plan_query(model: LanguageModel, query: str, cfg: QTConfig) -> dict | None:
    """Run all enabled understanding techniques in one call; return a dict.

    Returns ``None`` on model failure or unparseable output so the caller can
    fall back to the per-technique pipeline.
    """
    schema, instructions = _build_schema(cfg)
    if not schema:
        return None  # nothing to do in a single call

    prompt = (
        f"Query: {query}\n\n"
        f"Return a JSON object with exactly these keys: {', '.join(schema)}.\n"
        + "\n".join(instructions)
    )
    logger.debug("Planner prompt | query=%r | keys=%s", query, schema)
    try:
        raw = model.complete(prompt, system=_SYSTEM)
    except Exception:  # pragma: no cover - defensive; never block retrieval
        logger.exception("Single-call planner failed; falling back to per-technique")
        return None

    logger.debug("Planner raw model output=%r", raw)
    data = _parse_json(raw)
    if data is None:
        logger.warning("Planner returned unparseable JSON; falling back")
        return None
    return _normalise(data, query, cfg)


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
def _build_schema(cfg: QTConfig) -> tuple[list[str], list[str]]:
    keys: list[str] = []
    instr: list[str] = []
    if cfg.enable_classification:
        keys.append("classification")
        instr.append(
            f"- classification: one of {list(CATEGORIES)} — the query type."
        )
    if cfg.enable_intent_detection:
        keys.append("intent")
        instr.append(f"- intent: one of {list(INTENTS)} — the answer format wanted.")
    if cfg.enable_routing and len(cfg.routes) > 1:
        keys.append("route")
        instr.append(
            f"- route: the single best retrieval route from {cfg.routes}."
        )
    # Rewriting is the backbone; always request it when anything downstream runs.
    if cfg.enable_rewriting or cfg.enable_expansion or cfg.enable_multi_query:
        keys.append("rewritten")
        instr.append(
            "- rewritten: one clear, explicit, search-friendly version of the "
            "query (resolve vague references, name the subject)."
        )
    if cfg.enable_expansion:
        keys.append("expansions")
        instr.append(
            f"- expansions: up to {cfg.max_expansion_terms} related search terms "
            "(synonyms, expanded acronyms, adjacent concepts) as a string array."
        )
    if cfg.enable_multi_query:
        keys.append("variations")
        instr.append(
            f"- variations: {cfg.multi_query_count} alternative phrasings of the "
            "query as a string array."
        )
    if cfg.enable_hyde:
        keys.append("hypothetical_document")
        instr.append(
            "- hypothetical_document: a short 2-4 sentence hypothetical passage "
            "that would answer the query, in a factual technical tone."
        )
    if cfg.enable_multi_hop:
        keys.append("hops")
        instr.append(
            f"- hops: if the query needs chained lookups, an ordered array of up "
            f"to {cfg.max_hops} standalone sub-queries; otherwise an empty array."
        )
    return keys, instr


# --------------------------------------------------------------------------- #
# Parsing / normalisation
# --------------------------------------------------------------------------- #
def _parse_json(text: str) -> dict | None:
    if not text:
        return None
    cleaned = text.strip()
    # Strip ```json ... ``` fences if present.
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    # Isolate the outermost JSON object.
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(cleaned[start : end + 1])
    except (ValueError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def _str_list(value, limit: int) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        s = str(item).strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= limit:
            break
    return out


def _match_label(value, allowed: tuple[str, ...], default: str) -> str:
    norm = str(value or "").strip().lower()
    for label in allowed:
        if label in norm:
            return label
    return default


def _normalise(data: dict, query: str, cfg: QTConfig) -> dict:
    result: dict = {}

    if cfg.enable_classification:
        result["classification"] = _match_label(
            data.get("classification"), CATEGORIES, "semantic"
        )
    if cfg.enable_intent_detection:
        result["intent"] = _match_label(data.get("intent"), INTENTS, "other")
    if cfg.enable_routing:
        if len(cfg.routes) <= 1:
            result["route"] = cfg.routes[0] if cfg.routes else None
        else:
            result["route"] = _match_label(
                data.get("route"), tuple(cfg.routes), cfg.routes[0]
            )

    rewritten = str(data.get("rewritten") or "").strip() or query.strip()
    result["rewritten"] = rewritten

    if cfg.enable_expansion:
        result["expansions"] = _str_list(
            data.get("expansions"), cfg.max_expansion_terms
        )
    if cfg.enable_multi_query:
        result["variations"] = _str_list(
            data.get("variations"), cfg.multi_query_count
        )
    if cfg.enable_hyde:
        doc = str(data.get("hypothetical_document") or "").strip()
        result["hypothetical_document"] = doc or None
    if cfg.enable_multi_hop:
        hops = _str_list(data.get("hops"), cfg.max_hops)
        # Honour classification gating: only keep hops for multi_hop queries.
        if cfg.enable_classification and result.get("classification") != "multi_hop":
            hops = []
        result["hops"] = hops

    return result
