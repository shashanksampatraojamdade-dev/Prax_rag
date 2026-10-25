"""Hybrid fusion — combine semantic + keyword results with Reciprocal Rank Fusion.

Semantic (vector) and keyword (lexical) retrievers are good at different things:
vectors capture meaning/paraphrase, keywords nail exact terms, IDs and rare
tokens. Fusing them gives better recall than either alone — the core of "hybrid
retrieval".

We use **Reciprocal Rank Fusion (RRF)** because it combines rankings without
needing the two retrievers' scores to be on the same scale (cosine similarity vs
lexical rank). Each hit contributes ``1 / (rrf_k + rank)`` from every list it
appears in; hits found by *both* retrievers naturally rise to the top.
"""

from __future__ import annotations

import logging

from prax_re.interfaces import Retrieved

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    result_lists: list[list[Retrieved]],
    *,
    rrf_k: int = 60,
    top_k: int | None = None,
) -> list[Retrieved]:
    """Fuse several ranked lists of :class:`Retrieved` into one.

    Hits are deduplicated by ``id``; a hit's best (highest) ``score`` and a merged
    ``source_type`` are preserved. Returns the fused list ordered by RRF score,
    truncated to ``top_k`` when given.
    """
    fused: dict = {}
    for result_list in result_lists:
        for rank, hit in enumerate(result_list):
            contribution = 1.0 / (rrf_k + rank)
            existing = fused.get(hit.id)
            if existing is None:
                fused[hit.id] = {
                    "hit": hit,
                    "rrf": contribution,
                    "types": {hit.source_type},
                    "best_score": hit.score,
                }
            else:
                existing["rrf"] += contribution
                existing["types"].add(hit.source_type)
                if hit.score > existing["best_score"]:
                    existing["best_score"] = hit.score
                    # Prefer the higher-scoring hit's content/metadata.
                    existing["hit"] = hit

    ranked = sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)
    out: list[Retrieved] = []
    for entry in ranked:
        hit = entry["hit"]
        types = entry["types"]
        hit.source_type = "hybrid" if len(types) > 1 else next(iter(types))
        hit.score = entry["best_score"]
        out.append(hit)

    if top_k is not None:
        out = out[:top_k]
    logger.info(
        "RRF fused %d list(s) into %d unique hit(s); keeping %s",
        len(result_lists),
        len(fused),
        top_k if top_k is not None else "all",
    )
    return out
