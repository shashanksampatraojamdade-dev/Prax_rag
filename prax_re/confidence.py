"""Confidence scoring and the "I don't know" threshold.

A retrieval engine that always answers will happily hallucinate when the
knowledge base has nothing relevant. We guard against that with a confidence
score derived from retrieval relevance, and decline to answer when it falls
below a configurable threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from prax_re.interfaces import Retrieved

logger = logging.getLogger(__name__)


@dataclass
class Confidence:
    score: float
    label: str  # "high" | "medium" | "low"
    answerable: bool


def score_confidence(
    results: list[Retrieved], *, idk_threshold: float = 0.35
) -> Confidence:
    """Compute answer confidence (0-1) and whether it clears the IDK threshold.

    The score weights the single best match (is anything strongly relevant?) and
    the mean of the top matches (is the answer corroborated, not one lucky hit?).
    ``answerable`` is ``False`` when the score is below ``idk_threshold`` — the
    engine then declines instead of generating.
    """
    if not results:
        logger.info("Confidence: no results -> score=0.0 (not answerable)")
        return Confidence(0.0, "low", idk_threshold <= 0.0)

    try:
        scores = [max(0.0, min(1.0, float(r.score))) for r in results]
        best = max(scores)
        top = sorted(scores, reverse=True)[:3]
        mean_top = sum(top) / len(top)
        score = round(0.6 * best + 0.4 * mean_top, 4)
    except Exception:  # never let scoring break the pipeline
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
