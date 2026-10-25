"""Contextual compression over LangChain ``Document``s.

Same guarantee as ``prax_re.compression``: before building the prompt, trim each
retrieved document to only the sentences relevant to the query. This saves
tokens and boosts precision. The built-in compressor is a dependency-free
extractive one (lexical overlap scoring, no extra model call); hosts can inject a
LangChain ``BaseDocumentCompressor`` instead for semantic/LLM-based compression.
"""

from __future__ import annotations

import logging
import re

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

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
    return [s.strip() for s in _SENTENCE_SPLIT.split(text or "") if s.strip()]


def compress_text(
    query: str,
    content: str,
    *,
    max_sentences: int = 4,
    min_overlap: float = 0.1,
) -> str:
    """Return only the query-relevant sentences of ``content``.

    Always returns at least the single best sentence (never empty). Content with
    one sentence or fewer is returned unchanged.
    """
    sentences = _split_sentences(content)
    if len(sentences) <= 1:
        return (content or "").strip()

    q_terms = _tokens(query)
    if not q_terms:
        return (content or "").strip()

    scored: list[tuple[float, int, str]] = []
    for idx, sent in enumerate(sentences):
        s_terms = _tokens(sent)
        if not s_terms:
            continue
        overlap = len(q_terms & s_terms)
        score = overlap / len(q_terms) + 0.01 * overlap
        scored.append((score, idx, sent))

    if not scored:
        return (content or "").strip()

    scored.sort(key=lambda t: t[0], reverse=True)
    relevant = [t for t in scored if t[0] >= min_overlap][:max_sentences]
    if not relevant:
        relevant = scored[:1]

    kept = sorted(relevant, key=lambda t: t[1])
    return " ".join(sent for _, _, sent in kept)


def compress_documents(
    query: str,
    documents: list[Document],
    *,
    max_sentences: int = 4,
    min_overlap: float = 0.1,
) -> list[Document]:
    """Compress each document to its query-relevant sentences.

    Returns new ``Document`` objects (metadata preserved) with the original text
    kept in ``metadata['uncompressed_content']``. Never raises.
    """
    compressed: list[Document] = []
    saved = 0
    for doc in documents:
        try:
            new_content = compress_text(
                query,
                doc.page_content,
                max_sentences=max_sentences,
                min_overlap=min_overlap,
            )
        except Exception:
            logger.exception("Compression failed for a document; using full content")
            compressed.append(doc)
            continue
        saved += max(0, len(doc.page_content or "") - len(new_content))
        meta = dict(doc.metadata or {})
        meta.setdefault("uncompressed_content", doc.page_content)
        compressed.append(Document(page_content=new_content, metadata=meta))

    logger.info(
        "Contextual compression trimmed ~%d char(s) across %d doc(s)",
        saved,
        len(documents),
    )
    return compressed
