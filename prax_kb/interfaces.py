"""Integration contracts for the Prax knowledge-base SDK.

Host applications plug the SDK into their own stack by providing two adapters:

- an :class:`Embedder` (any embedding model — Vertex, OpenAI, local, ...)
- a :class:`KnowledgeStore` (any vector DB — pgvector, Pinecone, Qdrant, ...)

The SDK owns the *techniques* (parsing, chunking, dedup, versioning, incremental
indexing, adaptive sizing) and calls back into these protocols. Nothing in the
SDK imports a specific database or model client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


# --------------------------------------------------------------------------- #
# Data passed from the SDK to the host store
# --------------------------------------------------------------------------- #
@dataclass
class PreparedChild:
    """A small, embedded chunk used for precise retrieval."""

    content: str
    content_type: str  # "text" | "table"
    content_hash: str
    embedding: list[float]


@dataclass
class PreparedParent:
    """A larger context block owning the child chunks that index it."""

    content: str
    content_type: str
    children: list[PreparedChild]


@dataclass
class PreparedDocument:
    """A fully prepared document ready for the host store to persist."""

    source: str
    title: str | None
    content_hash: str
    version: int
    parents: list[PreparedParent]


@dataclass
class StoredDocument:
    """A lightweight handle to a document already held by the host store."""

    id: int | str
    source: str
    title: str | None
    content_hash: str | None
    version: int


# --------------------------------------------------------------------------- #
# Protocols the host implements
# --------------------------------------------------------------------------- #
@runtime_checkable
class Embedder(Protocol):
    """Any embedding backend. Must be deterministic for a given text."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...


@runtime_checkable
class KnowledgeStore(Protocol):
    """Any vector store that can support the KB techniques.

    Only the handful of operations the techniques need are required. A store
    that does not support versioning can return ``None``/``0`` from the
    versioning hooks and simply disable that feature via :class:`KBConfig`.
    """

    def get_active_document_by_source(self, source: str) -> StoredDocument | None:
        """Current (non-archived) document for a logical source, if any."""

    def get_active_document_by_hash(self, content_hash: str) -> StoredDocument | None:
        """Any active document whose full content matches this hash."""

    def archive_source(self, source: str) -> int:
        """Archive all active documents for a source. Returns rows archived."""

    def get_embeddings_by_hashes(
        self, hashes: set[str], source: str
    ) -> dict[str, list[float]]:
        """Return stored embeddings for prior chunks of a source, keyed by hash."""

    def add_document(self, document: PreparedDocument) -> StoredDocument:
        """Persist a prepared document (parents + embedded children)."""
