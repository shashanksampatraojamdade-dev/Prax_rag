"""Confidence scoring + the "I don't know" threshold (LangChain documents).

Identical scoring to ``prax_re.confidence`` so answers calibrate the same way:
weight the single best match (is anything strongly relevant?) and the mean of the
top matches (is it corroborated?), then decline when the score is below the
configured threshold instead of hallucinating.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.documents import Document

from prax_re_lc.documents import doc_score

logger = logging.getLogger(__name__)


@dataclass
class Confidence:
    score: float
    label: str  # "high" | "medium" | "low"
    answerable: bool


def score_confidence(
    results: list[Document], *, idk_threshold: float = 0.5
) -> Confidence:
    """Compute answer confidence (0-1) and whether it clears the IDK threshold."""
    if not results:
        logger.info("Confidence: no results -> score=0.0 (not answerable)")
        return Confidence(0.0, "low", idk_threshold <= 0.0)

    try:
        scores = [doc_score(r, rank) for rank, r in enumerate(results)]
        best = max(scores)
        top = sorted(scores, reverse=True)[:3]
        mean_top = sum(top) / len(top)
        score = round(0.6 * best + 0.4 * mean_top, 4)
    except Exception:
        logger.warning("Confidence scoring failed; defaulting to 0.0", exc_info=True)
        return Confidence(0.0, "low", idk_threshold <= 0.0)

    if score >= 0.7:
        label = "high"
    elif score >= 0.5:
        label = "medium"
    else:
        label = "low"
    answerable = score >= idk_threshold
    logger.info(
        "Confidence: score=%.4f (%s) threshold=%.2f -> %s",
        score,
        label,
        idk_threshold,
        "answer" if answerable else "DECLINE (below threshold)",
    )
    return Confidence(score, label, answerable)
