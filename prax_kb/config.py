"""Configuration for the Prax knowledge-base SDK.

A plain dataclass so host applications can build it however they like (env vars,
pydantic settings, a config file, hard-coded defaults). Every KB technique is an
independent on/off switch, so teams can adopt them incrementally.
"""

from dataclasses import dataclass


@dataclass
class KBConfig:
    # Chunk sizing
    chunk_size: int = 800
    chunk_overlap: int = 120
    parent_chunk_size: int = 2000

    # Chunking strategy
    chunk_strategy: str = "semantic"  # "semantic" | "recursive"
    semantic_breakpoint_percentile: float = 90.0

    # Technique switches (plug-and-play)
    enable_ocr: bool = True
    enable_parent_child: bool = True
    enable_deduplication: bool = True
    enable_versioning: bool = True
    enable_incremental: bool = True
    enable_adaptive_chunking: bool = True
