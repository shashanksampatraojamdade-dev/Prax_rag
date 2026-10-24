"""Prax KB — a plug-and-play knowledge-base enhancement SDK for RAG systems.

Bring your own embedding model and vector store; the SDK applies layout-aware
parsing, semantic / table-aware / parent-child chunking, adaptive chunk sizing,
deduplication, document versioning and incremental indexing.

Quick start::

    from prax_kb import KnowledgeBase, KBConfig

    kb = KnowledgeBase(embedder=my_embedder, store=my_store, config=KBConfig())
    kb.ingest_text("...", source="policy.md", title="Policy")
"""

from prax_kb.config import KBConfig
from prax_kb.context import RetrievedChunk, build_context
from prax_kb.interfaces import (
    Embedder,
    KnowledgeStore,
    PreparedChild,
    PreparedDocument,
    PreparedParent,
    StoredDocument,
)
from prax_kb.loaders import UnsupportedFileError, extract_text
from prax_kb.pipeline import IngestResult, KnowledgeBase

__all__ = [
    "KnowledgeBase",
    "KBConfig",
    "IngestResult",
    "Embedder",
    "KnowledgeStore",
    "PreparedDocument",
    "PreparedParent",
    "PreparedChild",
    "StoredDocument",
    "RetrievedChunk",
    "build_context",
    "extract_text",
    "UnsupportedFileError",
]

__version__ = "0.1.0"
