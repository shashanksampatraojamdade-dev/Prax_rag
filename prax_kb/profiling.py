"""Adaptive chunk sizing.

Picks chunk sizes from the *shape* of a document so retrieval granularity fits
the content:

- **Dense / legal** text (short clauses, numbered sub-sections, legal diction)
  -> small chunks that isolate individual clauses.
- **Narrative / general** text (long paragraphs, long sentences) -> large chunks
  that preserve the flow of an argument or story.
- Otherwise fall back to the provided base sizes.
"""

from dataclasses import dataclass
import logging
import re

logger = logging.getLogger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_RE = re.compile(r"^\s*(\(?[0-9]+[.)]|[0-9]+\.[0-9]+|\([a-z]\)|[a-z]\))\s+")
_LEGAL_TERMS = (
    "shall",
    "hereby",
    "herein",
    "pursuant",
    "clause",
    "section",
    "liability",
    "indemnif",
    "warrant",
    "agreement",
    "party",
    "parties",
    "obligation",
    "terms and conditions",
    "whereas",
    "notwithstanding",
)


@dataclass
class DocumentProfile:
    kind: str  # "dense" | "narrative" | "default"
    child_size: int
    parent_size: int


def profile_document(
    text: str, base_child: int, base_parent: int
) -> DocumentProfile:
    sample = text.strip()
    if len(sample) < 400:
        return DocumentProfile("default", base_child, base_parent)

    lines = [ln for ln in sample.splitlines() if ln.strip()]
    sentences = [s for s in _SENTENCE_RE.split(sample) if s.strip()]
    avg_sentence = sum(len(s) for s in sentences) / max(len(sentences), 1)

    clause_lines = sum(1 for ln in lines if _CLAUSE_RE.match(ln))
    clause_ratio = clause_lines / max(len(lines), 1)

    lowered = sample.lower()
    legal_hits = sum(lowered.count(term) for term in _LEGAL_TERMS)
    legal_density = legal_hits / max(len(sample) / 1000, 1)  # hits per 1k chars

    dense = clause_ratio >= 0.25 or legal_density >= 3 or avg_sentence < 60
    narrative = avg_sentence > 160 and clause_ratio < 0.05 and legal_density < 1

    if dense:
        profile = DocumentProfile(
            "dense",
            child_size=max(250, base_child // 2),
            parent_size=max(800, base_parent // 2),
        )
    elif narrative:
        profile = DocumentProfile(
            "narrative",
            child_size=min(1600, base_child * 2),
            parent_size=min(4000, base_parent * 2),
        )
    else:
        profile = DocumentProfile("default", base_child, base_parent)

    logger.info(
        "Adaptive profile=%s (avg_sentence=%.0f, clause_ratio=%.2f, "
        "legal_density=%.1f) -> child=%d parent=%d",
        profile.kind,
        avg_sentence,
        clause_ratio,
        legal_density,
        profile.child_size,
        profile.parent_size,
    )
    return profile
