"""Adapters binding the backend-agnostic prax_kb SDK to this app's stack.

- ``EmbeddingService`` already satisfies ``prax_kb.Embedder`` (it exposes
  ``embed_documents``), so it is used directly as the embedder.
- ``PgVectorKnowledgeStore`` implements ``prax_kb.KnowledgeStore`` on top of the
  pgvector-backed ``VectorStore``.
- ``settings_to_kb_config`` maps the app's pydantic Settings to a ``KBConfig``.
"""

from app.core.config import Settings
from app.repositories.vector_store import VectorStore
from prax_kb import KBConfig, KnowledgeStore, PreparedDocument, StoredDocument


def settings_to_kb_config(settings: Settings) -> KBConfig:
    return KBConfig(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        parent_chunk_size=settings.parent_chunk_size,
        chunk_strategy=settings.chunk_strategy,
        semantic_breakpoint_percentile=settings.semantic_breakpoint_percentile,
        enable_ocr=settings.enable_ocr,
        enable_parent_child=settings.enable_parent_child,
        enable_deduplication=settings.enable_deduplication,
        enable_versioning=settings.enable_versioning,
        enable_incremental=settings.enable_incremental,
        enable_adaptive_chunking=settings.enable_adaptive_chunking,
    )


class PgVectorKnowledgeStore(KnowledgeStore):
    """Implements the SDK's KnowledgeStore protocol over pgvector."""

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    @staticmethod
    def _to_stored(doc) -> StoredDocument | None:
        if doc is None:
            return None
        return StoredDocument(
            id=doc.id,
            source=doc.source,
            title=doc.title,
            content_hash=doc.content_hash,
            version=doc.version,
        )

    def get_active_document_by_source(self, source: str) -> StoredDocument | None:
        return self._to_stored(self._store.get_active_document_by_source(source))

    def get_active_document_by_hash(self, content_hash: str) -> StoredDocument | None:
        return self._to_stored(self._store.get_active_document_by_hash(content_hash))

    def archive_source(self, source: str) -> int:
        return self._store.archive_source(source)

    def get_embeddings_by_hashes(
        self, hashes: set[str], source: str
    ) -> dict[str, list[float]]:
        return self._store.get_embeddings_by_hashes(hashes, source)

    def add_document(self, document: PreparedDocument) -> StoredDocument:
        flat = [c for p in document.parents for c in p.children]
        doc = self._store.add_document_with_parents(
            source=document.source,
            title=document.title,
            parents=document.parents,
            child_embeddings=[c.embedding for c in flat],
            child_hashes=[c.content_hash for c in flat],
            version=document.version,
            content_hash=document.content_hash,
        )
        return self._to_stored(doc)
