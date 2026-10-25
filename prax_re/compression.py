"""Contextual compression — trim retrieved chunks to only the relevant sentences.

Before handing retrieved chunks to the LLM, most of a chunk is usually noise for
any *given* question. Contextual compression keeps only the sentences that are
actually relevant to the query, which:

- **saves tokens** (smaller, cheaper, faster prompts), and
- **boosts precision** (the model isn't distracted by irrelevant text).

The default compressor here is a dependency-free *extractive* one: it splits each
chunk into sentences, scores every sentence by lexical overlap with the query,
and keeps the best-scoring sentences (above a relevance floor, capped to a
maximum) in their original order. Because it is purely lexical it adds no latency
and no extra model calls.

Hosts that want semantic/LLM-based compression can inject their own callable with
signature ``(query: str, content: str) -> str`` via ``RetrievalEngine`` — this
module only provides the safe, zero-dependency default.
"""

from __future__ import annotations

import logging
import re

from prax_re.config import REConfig
from prax_re.interfaces import Retrieved

logger = logging.getLogger(__name__)

# Minimal English stop-word list so overlap scoring keys on meaningful terms.
_STOPWORDS = frozenset(
    """
    a an and are as at be but by for from has have how i if in into is it its
    of on or that the their them then there these they this to was were what
    when where which who why will with you your do does did can could should
    would me my we our us about over under more most such not no than too very
    """.split()
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n+")
_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS}


def _split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENTENCE_SPLIT.split(text or "") if s.strip()]
    return parts


def compress_content(
    query: str,
    content: str,
    *,
    max_sentences: int = 4,
    min_overlap: float = 0.1,
) -> str:
    """Return only the query-relevant sentences of ``content``.

    - ``max_sentences`` caps how many sentences are kept.
    - ``min_overlap`` is the minimum fraction of query terms a sentence must
      share to be considered relevant.

    Always returns at least the single best sentence (never an empty string) so
    the model still has *something* grounded to work with. If the content has few
    sentences, it is returned unchanged.
    """
    sentences = _split_sentences(content)
    if len(sentences) <= 1:
        return content.strip()

    q_terms = _tokens(query)
    if not q_terms:
        return content.strip()

    scored: list[tuple[float, int, str]] = []
    for idx, sent in enumerate(sentences):
        s_terms = _tokens(sent)
        if not s_terms:
            continue
        overlap = len(q_terms & s_terms)
        # Fraction of the *query* covered, nudged by raw overlap for ties.
        score = overlap / len(q_terms) + 0.01 * overlap
        scored.append((score, idx, sent))

    if not scored:
        return content.strip()

    scored.sort(key=lambda t: t[0], reverse=True)
    relevant = [t for t in scored if t[0] >= min_overlap][:max_sentences]
    if not relevant:
        # Nothing cleared the floor — keep just the single best sentence.
        relevant = scored[:1]

    # Restore original reading order for coherence.
    kept = sorted(relevant, key=lambda t: t[1])
    return " ".join(sent for _, _, sent in kept)


def compress_results(
    query: str,
    results: list[Retrieved],
    config: REConfig | None = None,
) -> list[Retrieved]:
    """Compress every hit's ``content`` to its query-relevant sentences.

    Returns new ``Retrieved`` objects (scores/metadata/ids untouched) so
    confidence scoring and citations keep working on the originals. The
    pre-compression text is preserved in ``metadata['uncompressed_content']``.
    """
    cfg = config or REConfig()
    if not cfg.enable_compression:
        return results

    compressed: list[Retrieved] = []
    saved_chars = 0
    for r in results:
        try:
            new_content = compress_content(
                query,
                r.content,
                max_sentences=cfg.compression_max_sentences,
                min_overlap=cfg.compression_min_overlap,
            )
        except Exception:  # never let compression break answering
            logger.exception("Compression failed for hit %s; using full content", r.id)
            compressed.append(r)
            continue

        saved_chars += max(0, len(r.content or "") - len(new_content))
        meta = dict(r.metadata or {})
        meta.setdefault("uncompressed_content", r.content)
        compressed.append(
            Retrieved(
                id=r.id,
                content=new_content,
                score=r.score,
                source_type=r.source_type,
                metadata=meta,
            )
        )

    logger.info(
        "Contextual compression trimmed ~%d char(s) across %d hit(s)",
        saved_chars,
        len(results),
    )
    return compressed
