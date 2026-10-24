"""The Prax knowledge-base pipeline — the plug-and-play entry point.

Compose it with any :class:`Embedder` and :class:`KnowledgeStore`:

    kb = KnowledgeBase(embedder=my_embedder, store=my_store, config=KBConfig())
    result = kb.ingest_text("...", source="policy.md", title="Policy")

The pipeline owns every KB technique (layout-aware parsing, semantic / table /
parent-child chunking, adaptive sizing, deduplication, versioning, incremental
indexing) and calls back into the injected adapters for embedding and storage.
"""

from dataclasses import dataclass, field
import logging

from prax_kb.chunking import build_parent_child, split_text
from prax_kb.config import KBConfig
from prax_kb.hashing import content_hash
from prax_kb.interfaces import (
    Embedder,
    KnowledgeStore,
    PreparedChild,
    PreparedDocument,
    PreparedParent,
)
from prax_kb.loaders import extract_text
from prax_kb.profiling import profile_document

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    document_id: int | str | None
    source: str
    title: str | None
    chunks_ingested: int
    status: str  # indexed | updated | unchanged | duplicate
    version: int = 1
    chunks_reused: int = 0
    chunks_embedded: int = 0
    profile: str | None = None
    duplicate_of: str | None = None

    def as_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "source": self.source,
            "title": self.title,
            "chunks_ingested": self.chunks_ingested,
            "status": self.status,
            "version": self.version,
            "chunks_reused": self.chunks_reused,
            "chunks_embedded": self.chunks_embedded,
            "profile": self.profile,
            "duplicate_of": self.duplicate_of,
        }


class KnowledgeBase:
    """Backend-agnostic knowledge-base ingestion pipeline."""

    def __init__(
        self,
        embedder: Embedder,
        store: KnowledgeStore,
        config: KBConfig | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._config = config or KBConfig()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def ingest_file(
        self, data: bytes, filename: str, title: str | None = None
    ) -> IngestResult:
        """Extract text from raw file bytes (layout-aware + OCR) then ingest."""
        text = extract_text(data, filename=filename, enable_ocr=self._config.enable_ocr)
        return self.ingest_text(text, source=filename, title=title)

    def ingest_text(
        self, text: str, source: str, title: str | None = None
    ) -> IngestResult:
        cfg = self._config
        if not text or not text.strip():
            raise ValueError("No ingestible content found in the provided text")

        doc_hash = content_hash(text)
        prev = self._store.get_active_document_by_source(source)

        # --- Deduplication / unchanged ---------------------------------- #
        if prev is not None and prev.content_hash == doc_hash:
            logger.info("Skip ingest: source=%s unchanged (v%d)", source, prev.version)
            return IngestResult(
                document_id=prev.id,
                source=source,
                title=prev.title,
                chunks_ingested=0,
                status="unchanged",
                version=prev.version,
            )
        if cfg.enable_deduplication:
            dup = self._store.get_active_document_by_hash(doc_hash)
            if dup is not None and dup.source != source:
                logger.info(
                    "Skip ingest: source=%s duplicates source=%s", source, dup.source
                )
                return IngestResult(
                    document_id=dup.id,
                    source=source,
                    title=dup.title,
                    chunks_ingested=0,
                    status="duplicate",
                    duplicate_of=dup.source,
                )

        # --- Adaptive chunk sizing -------------------------------------- #
        if cfg.enable_adaptive_chunking:
            profile = profile_document(text, cfg.chunk_size, cfg.parent_chunk_size)
            child_size, parent_size, profile_kind = (
                profile.child_size,
                profile.parent_size,
                profile.kind,
            )
        else:
            child_size, parent_size, profile_kind = (
                cfg.chunk_size,
                cfg.parent_chunk_size,
                "config",
            )

        # --- Chunking --------------------------------------------------- #
        parents = self._chunk(text, child_size, parent_size)
        if not parents:
            raise ValueError("No ingestible content found in the provided text")

        # chunk-level dedup: drop identical children within each parent
        for parent in parents:
            seen: set[str] = set()
            unique = []
            for child in parent.children:
                h = content_hash(child.content)
                if h in seen:
                    continue
                seen.add(h)
                unique.append(child)
            parent.children = unique

        flat_children = [c for p in parents for c in p.children]
        contents = [c.content for c in flat_children]
        hashes = [content_hash(c) for c in contents]

        # --- Incremental indexing: reuse unchanged embeddings ----------- #
        reuse: dict[str, list[float]] = {}
        if cfg.enable_incremental and prev is not None:
            reuse = self._store.get_embeddings_by_hashes(set(hashes), source)

        missing: dict[str, str] = {}
        for h, c in zip(hashes, contents):
            if h not in reuse and h not in missing:
                missing[h] = c
        new_map: dict[str, list[float]] = {}
        if missing:
            missing_hashes = list(missing)
            new_embeddings = self._embedder.embed_documents(
                [missing[h] for h in missing_hashes]
            )
            new_map = dict(zip(missing_hashes, new_embeddings))

        embeddings = [reuse[h] if h in reuse else new_map[h] for h in hashes]
        reused_count = sum(1 for h in hashes if h in reuse)

        # --- Document versioning ---------------------------------------- #
        version = (prev.version + 1) if prev is not None else 1
        if cfg.enable_versioning and prev is not None:
            self._store.archive_source(source)

        prepared = self._prepare(parents, hashes, embeddings, source, title,
                                 doc_hash, version)
        stored = self._store.add_document(prepared)

        logger.info(
            "Ingested id=%s source=%s v%d parents=%d children=%d "
            "(reused=%d, embedded=%d, profile=%s, strategy=%s)",
            stored.id,
            source,
            version,
            len(parents),
            len(contents),
            reused_count,
            len(contents) - reused_count,
            profile_kind,
            cfg.chunk_strategy,
        )
        return IngestResult(
            document_id=stored.id,
            source=source,
            title=title,
            chunks_ingested=len(contents),
            status="updated" if version > 1 else "indexed",
            version=version,
            chunks_reused=reused_count,
            chunks_embedded=len(contents) - reused_count,
            profile=profile_kind,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _chunk(self, text: str, child_size: int, parent_size: int):
        cfg = self._config
        if cfg.enable_parent_child:
            return build_parent_child(
                text,
                parent_size=parent_size,
                child_size=child_size,
                chunk_overlap=cfg.chunk_overlap,
                strategy=cfg.chunk_strategy,
                embedder=self._embedder.embed_documents,
                semantic_percentile=cfg.semantic_breakpoint_percentile,
            )
        # Flat mode: each chunk is its own parent (one child == the chunk).
        from prax_kb.chunking import ParentChunk

        flat = split_text(
            text,
            chunk_size=child_size,
            chunk_overlap=cfg.chunk_overlap,
            strategy=cfg.chunk_strategy,
            embedder=self._embedder.embed_documents,
            semantic_percentile=cfg.semantic_breakpoint_percentile,
        )
        return [ParentChunk(c.content, c.content_type, [c]) for c in flat]

    @staticmethod
    def _prepare(
        parents, hashes, embeddings, source, title, doc_hash, version
    ) -> PreparedDocument:
        emb = iter(embeddings)
        h = iter(hashes)
        prepared_parents = []
        for parent in parents:
            children = [
                PreparedChild(
                    content=child.content,
                    content_type=child.content_type,
                    content_hash=next(h),
                    embedding=next(emb),
                )
                for child in parent.children
            ]
            prepared_parents.append(
                PreparedParent(parent.content, parent.content_type, children)
            )
        return PreparedDocument(
            source=source,
            title=title,
            content_hash=doc_hash,
            version=version,
            parents=prepared_parents,
        )
