"""Reciprocal Rank Fusion over LangChain ``Document`` lists.

Mirrors ``prax_re.fusion`` but operates on ``Document``s. Semantic and keyword
retrievers excel at different things (meaning vs. exact terms/IDs); fusing their
rankings gives better recall than either alone. RRF needs no score
normalisation — each hit contributes ``1 / (rrf_k + rank)`` from every list it
appears in, so hits found by *both* retrievers rise to the top.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document

from prax_re_lc.documents import doc_id, doc_score, doc_source_type, set_meta

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    *,
    rrf_k: int = 60,
    top_k: int | None = None,
) -> list[Document]:
    """Fuse several ranked lists of ``Document`` into one deduplicated ranking.

    Hits are deduplicated by their id; the best (highest) score and a merged
    ``source_type`` (``"hybrid"`` when a hit came from more than one retriever)
    are written back into metadata. Returns the fused list ordered by RRF score,
    truncated to ``top_k`` when given.
    """
    fused: dict = {}
    for result_list in result_lists:
        for rank, hit in enumerate(result_list):
            key = doc_id(hit)
            contribution = 1.0 / (rrf_k + rank)
            score = doc_score(hit, rank)
            existing = fused.get(key)
            if existing is None:
                fused[key] = {
                    "hit": hit,
                    "rrf": contribution,
                    "types": {doc_source_type(hit)},
                    "best_score": score,
                }
            else:
                existing["rrf"] += contribution
                existing["types"].add(doc_source_type(hit))
                if score > existing["best_score"]:
                    existing["best_score"] = score
                    existing["hit"] = hit

    ranked = sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)
    out: list[Document] = []
    for entry in ranked:
        hit = entry["hit"]
        types = entry["types"]
        set_meta(
            hit,
            source_type="hybrid" if len(types) > 1 else next(iter(types)),
            score=entry["best_score"],
        )
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
