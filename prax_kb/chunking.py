"""Knowledge-base chunking.

Pipeline per document:

1. Segment by Markdown headings so sections stay together; every chunk is
   prefixed with its heading breadcrumb (e.g. "Report > Revenue") so retrieval
   and citations keep structural context.
2. Inside a section, Markdown tables are isolated as *atomic* chunks
   (``content_type="table"``) and never split, so table-aware retrieval can
   surface a full table intact.
3. Remaining prose is chunked either by a semantic breakpoint strategy
   (embedding-distance aware) or a recursive size-based splitter.
4. ``build_parent_child`` additionally groups small child chunks under a larger
   parent block for parent-child retrieval.
"""

from collections.abc import Callable
from dataclasses import dataclass
import re

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

Embedder = Callable[[list[str]], list[list[float]]]


@dataclass
class TextChunk:
    """A chunk ready for embedding, tagged with its content type."""

    content: str
    content_type: str = "text"  # "text" | "table"


@dataclass
class ParentChunk:
    """A larger context block plus the small child chunks that index it."""

    content: str
    content_type: str
    children: list[TextChunk]


def split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    strategy: str = "recursive",
    embedder: Embedder | None = None,
    semantic_percentile: float = 90.0,
) -> list[TextChunk]:
    """Split Markdown into heading-aware, table-aware, (optionally) semantic chunks."""
    text = text.strip()
    if not text:
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    use_semantic = strategy == "semantic" and embedder is not None

    chunks: list[TextChunk] = []
    for heading_path, body in _iter_sections(text):
        prefix = f"{heading_path}\n\n" if heading_path else ""
        budget = max(chunk_size - len(prefix), 1)

        for kind, block in _iter_blocks(body):
            if kind == "table":
                chunks.append(TextChunk(f"{prefix}{block}", content_type="table"))
                continue

            if use_semantic:
                pieces = _semantic_split(
                    block, budget, chunk_overlap, embedder, semantic_percentile
                )
            else:
                pieces = _split_plain(block, budget, chunk_overlap)
            for piece in pieces:
                chunks.append(TextChunk(f"{prefix}{piece}", content_type="text"))

    return chunks


def build_parent_child(
    text: str,
    parent_size: int,
    child_size: int,
    chunk_overlap: int,
    strategy: str = "recursive",
    embedder: Embedder | None = None,
    semantic_percentile: float = 90.0,
) -> list[ParentChunk]:
    """Build larger parent blocks, each with the small child chunks that index it.

    Children are what gets embedded and searched (precise retrieval); the parent
    is the broader block handed to the LLM for context. Heading breadcrumbs,
    table-atomicity and the semantic/recursive strategies are all preserved.
    """
    text = text.strip()
    if not text:
        return []
    if chunk_overlap >= child_size:
        raise ValueError("chunk_overlap must be smaller than child_size")

    use_semantic = strategy == "semantic" and embedder is not None

    parents: list[ParentChunk] = []
    for heading_path, body in _iter_sections(text):
        prefix = f"{heading_path}\n\n" if heading_path else ""
        p_budget = max(parent_size - len(prefix), 1)
        c_budget = max(child_size - len(prefix), 1)

        for kind, block in _iter_blocks(body):
            if kind == "table":
                table = TextChunk(f"{prefix}{block}", content_type="table")
                parents.append(ParentChunk(f"{prefix}{block}", "table", [table]))
                continue

            for parent_body in _split_plain(block, p_budget, chunk_overlap):
                if use_semantic:
                    child_bodies = _semantic_split(
                        parent_body, c_budget, chunk_overlap, embedder,
                        semantic_percentile,
                    )
                else:
                    child_bodies = _split_plain(parent_body, c_budget, chunk_overlap)
                children = [
                    TextChunk(f"{prefix}{cb}", content_type="text")
                    for cb in child_bodies
                ]
                if not children:
                    continue
                parents.append(
                    ParentChunk(f"{prefix}{parent_body}", "text", children)
                )

    return parents


# --------------------------------------------------------------------------- #
# Heading-aware sectioning
# --------------------------------------------------------------------------- #
def _iter_sections(text: str):
    """Yield (heading_breadcrumb, section_body) following the heading tree."""
    stack: list[tuple[int, str]] = []
    body_lines: list[str] = []
    current_path = ""

    def breadcrumb() -> str:
        return " > ".join(title for _, title in stack)

    for line in text.splitlines():
        match = _HEADING_RE.match(line.strip())
        if match:
            if body_lines:
                yield current_path, "\n".join(body_lines).strip()
                body_lines = []
            level = len(match.group(1))
            title = match.group(2).strip().strip("*").strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            current_path = breadcrumb()
        else:
            body_lines.append(line)

    if body_lines:
        yield current_path, "\n".join(body_lines).strip()


# --------------------------------------------------------------------------- #
# Table-aware block separation
# --------------------------------------------------------------------------- #
def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def _iter_blocks(body: str):
    """Yield ("table", block) / ("text", block) preserving document order."""
    lines = body.splitlines()
    buffer: list[str] = []
    mode: str | None = None

    def flush():
        nonlocal buffer, mode
        if buffer and "\n".join(buffer).strip():
            yield mode or "text", "\n".join(buffer).strip()
        buffer = []
        mode = None

    for line in lines:
        line_mode = "table" if _is_table_line(line) else "text"
        if mode is None:
            mode = line_mode
        elif line_mode != mode:
            yield from flush()
            mode = line_mode
        buffer.append(line)
    yield from flush()


# --------------------------------------------------------------------------- #
# Semantic (embedding-breakpoint) chunking
# --------------------------------------------------------------------------- #
def _sentence_split(text: str) -> list[str]:
    sentences: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        sentences.extend(s.strip() for s in _SENTENCE_RE.split(para) if s.strip())
    return sentences


def _cosine_distance(a: list[float], b: list[float]) -> float:
    import numpy as np

    va, vb = np.asarray(a, dtype="float32"), np.asarray(b, dtype="float32")
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 1.0
    return 1.0 - float(np.dot(va, vb) / denom)


def _semantic_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    embedder: Embedder,
    percentile: float,
) -> list[str]:
    """Group adjacent sentences, breaking where meaning shifts or size overflows."""
    sentences = _sentence_split(text)
    if len(sentences) <= 1:
        return _split_plain(text, chunk_size, chunk_overlap)

    try:
        embeddings = embedder(sentences)
    except Exception:  # pragma: no cover - fall back to size-based splitting
        return _split_plain(text, chunk_size, chunk_overlap)

    distances = [
        _cosine_distance(embeddings[i], embeddings[i + 1])
        for i in range(len(sentences) - 1)
    ]
    threshold = _percentile(distances, percentile)

    chunks: list[str] = []
    buffer = sentences[0]
    for i in range(1, len(sentences)):
        sentence = sentences[i]
        break_here = distances[i - 1] >= threshold
        too_big = len(buffer) + len(sentence) + 1 > chunk_size
        if break_here or too_big:
            chunks.append(buffer.strip())
            buffer = sentence
        else:
            buffer = f"{buffer} {sentence}".strip()
    if buffer.strip():
        chunks.append(buffer.strip())

    normalized: list[str] = []
    for chunk in chunks:
        if len(chunk) > chunk_size:
            normalized.extend(_hard_split(chunk, chunk_size, chunk_overlap))
        else:
            normalized.append(chunk)
    return _apply_overlap(normalized, chunk_overlap)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 1.0
    import numpy as np

    return float(np.percentile(np.asarray(values, dtype="float32"), percentile))


# --------------------------------------------------------------------------- #
# Recursive size-based splitting
# --------------------------------------------------------------------------- #
def _split_plain(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []
    buffer = ""

    def flush() -> None:
        nonlocal buffer
        if buffer.strip():
            chunks.append(buffer.strip())
        buffer = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            flush()
            chunks.extend(_hard_split(para, chunk_size, chunk_overlap))
            continue

        if len(buffer) + len(para) + 1 <= chunk_size:
            buffer = f"{buffer}\n{para}".strip()
        else:
            flush()
            buffer = para
    flush()

    return _apply_overlap(chunks, chunk_overlap)


def _hard_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    step = chunk_size - chunk_overlap
    return [text[i : i + chunk_size] for i in range(0, len(text), step)]


def _apply_overlap(chunks: list[str], chunk_overlap: int) -> list[str]:
    if chunk_overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped: list[str] = [chunks[0]]
    for prev, current in zip(chunks, chunks[1:]):
        tail = prev[-chunk_overlap:]
        overlapped.append(f"{tail} {current}".strip())
    return overlapped
