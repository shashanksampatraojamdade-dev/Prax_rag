"""Parent-aware context assembly for RAG prompts.

Backend-agnostic: operates on plain :class:`RetrievedChunk` values the host
produces from its own similarity search. Expands precise child matches to their
larger parent blocks and deduplicates, so the LLM gets broad context while
citations still map to the retrieved children.
"""

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    chunk_id: int | str
    source: str
    title: str | None = None
    content: str = ""
    content_type: str = "text"
    parent_id: int | str | None = None
    parent_content: str | None = None


def build_context(retrieved: list[RetrievedChunk]) -> str:
    """Expand retrieved child chunks to their parent blocks (deduplicated).

    When several children share one parent the block appears once, labelled with
    all the citation numbers it covers (e.g. ``[1, 2]``).
    """
    order: list[str] = []
    blocks: dict[str, dict] = {}
    for i, r in enumerate(retrieved):
        key = f"p{r.parent_id}" if r.parent_id is not None else f"c{r.chunk_id}"
        if key not in blocks:
            order.append(key)
            label = r.title or r.source
            kind = " | table" if r.content_type == "table" else ""
            blocks[key] = {
                "content": r.parent_content or r.content,
                "label": f"{label}{kind}",
                "nums": [],
            }
        blocks[key]["nums"].append(i + 1)

    out = []
    for key in order:
        b = blocks[key]
        nums = ", ".join(str(n) for n in b["nums"])
        out.append(f"[{nums}] (source: {b['label']})\n{b['content']}")
    return "\n\n".join(out)
