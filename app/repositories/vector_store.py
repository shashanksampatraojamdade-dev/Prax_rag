from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, or_, select, update
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document, Parent


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    source: str
    title: str | None
    content: str
    distance: float
    content_type: str = "text"
    parent_id: int | None = None
    parent_content: str | None = None
    uploaded_at: datetime | None = None
    version: int = 1


class VectorStore:
    """pgvector-backed repository for document chunks."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add_document(
        self,
        source: str,
        title: str | None,
        chunks: list[str],
        embeddings: list[list[float]],
        content_types: list[str] | None = None,
    ) -> Document:
        """Persist a document and its embedded chunks in a single transaction."""
        document = Document(source=source, title=title)
        types = content_types or ["text"] * len(chunks)
        for index, (content, embedding, ctype) in enumerate(
            zip(chunks, embeddings, types)
        ):
            document.chunks.append(
                Chunk(
                    chunk_index=index,
                    content=content,
                    embedding=embedding,
                    content_type=ctype,
                )
            )
        self._session.add(document)
        self._session.commit()
        self._session.refresh(document)
        return document

    def add_document_with_parents(
        self,
        source: str,
        title: str | None,
        parents: list,
        child_embeddings: list[list[float]],
        child_hashes: list[str] | None = None,
        version: int = 1,
        content_hash: str | None = None,
    ) -> Document:
        """Persist a document as parent blocks each owning embedded child chunks.

        ``parents`` is a list of ``ParentChunk`` (content, content_type,
        children); ``child_embeddings`` are the embeddings of the children in
        flattened document order.
        """
        document = Document(
            source=source,
            title=title,
            content_hash=content_hash,
            version=version,
            is_archived=False,
        )
        self._session.add(document)
        emb = iter(child_embeddings)
        hashes = iter(child_hashes) if child_hashes is not None else None
        child_index = 0
        for p_index, parent in enumerate(parents):
            parent_row = Parent(
                parent_index=p_index,
                content=parent.content,
                content_type=parent.content_type,
            )
            document.parents.append(parent_row)
            for child in parent.children:
                document.chunks.append(
                    Chunk(
                        chunk_index=child_index,
                        content=child.content,
                        content_type=child.content_type,
                        content_hash=next(hashes) if hashes is not None else None,
                        embedding=next(emb),
                        parent=parent_row,
                    )
                )
                child_index += 1
        self._session.commit()
        self._session.refresh(document)
        return document

    # --- versioning / dedup / incremental helpers --------------------------- #
    def get_active_document_by_source(self, source: str) -> Document | None:
        """Return the current (non-archived) document for a source, if any."""
        stmt = (
            select(Document)
            .where(Document.source == source, Document.is_archived.is_(False))
            .order_by(Document.version.desc())
            .limit(1)
        )
        return self._session.execute(stmt).scalars().first()

    def get_active_document_by_hash(self, content_hash: str) -> Document | None:
        """Return any active document whose full content matches this hash."""
        stmt = (
            select(Document)
            .where(
                Document.content_hash == content_hash,
                Document.is_archived.is_(False),
            )
            .limit(1)
        )
        return self._session.execute(stmt).scalars().first()

    def archive_source(self, source: str) -> int:
        """Mark all currently-active documents for a source as archived."""
        result = self._session.execute(
            update(Document)
            .where(Document.source == source, Document.is_archived.is_(False))
            .values(is_archived=True)
        )
        self._session.commit()
        return result.rowcount or 0

    def get_embeddings_by_hashes(
        self, hashes: set[str], source: str
    ) -> dict[str, list[float]]:
        """Return stored embeddings for prior chunks of a source, keyed by hash.

        Used for incremental indexing: unchanged chunks keep their embedding
        instead of being re-embedded.
        """
        if not hashes:
            return {}
        stmt = (
            select(Chunk.content_hash, Chunk.embedding)
            .join(Document, Chunk.document_id == Document.id)
            .where(
                Document.source == source,
                Chunk.content_hash.in_(hashes),
            )
        )
        found: dict[str, list[float]] = {}
        for h, embedding in self._session.execute(stmt).all():
            if h is not None and h not in found:
                found[h] = list(embedding)
        return found

    def similarity_search(
        self, query_embedding: list[float], top_k: int
    ) -> list[RetrievedChunk]:
        """Return the top_k nearest child chunks (active documents only)."""
        distance = Chunk.embedding.cosine_distance(query_embedding).label("distance")
        stmt = (
            select(Chunk, Document, Parent, distance)
            .join(Document, Chunk.document_id == Document.id)
            .outerjoin(Parent, Chunk.parent_id == Parent.id)
            .where(Document.is_archived.is_(False))
            .order_by(distance)
            .limit(top_k)
        )
        results = self._session.execute(stmt).all()
        return [
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=document.id,
                source=document.source,
                title=document.title,
                content=chunk.content,
                distance=float(dist),
                content_type=chunk.content_type,
                parent_id=parent.id if parent is not None else None,
                parent_content=parent.content if parent is not None else None,
                uploaded_at=document.created_at,
                version=document.version,
            )
            for chunk, document, parent, dist in results
        ]

    def keyword_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Lexical (keyword) search over chunk text using Postgres full-text.

        Complements vector search for the "hybrid" strategy: exact terms, IDs,
        codes and rare tokens that embeddings often miss. Ranks with
        ``ts_rank_cd`` over ``websearch_to_tsquery``; the normalised rank is
        mapped onto ``distance`` (``1 - rank``) so it is directly comparable to
        the cosine-distance field used elsewhere. Falls back to an ``ILIKE``
        match when the query has no usable tsquery terms.
        """
        query = (query or "").strip()
        if not query:
            return []

        tsvector = func.to_tsvector("english", Chunk.content)
        tsquery = func.websearch_to_tsquery("english", query)
        rank = func.ts_rank_cd(tsvector, tsquery).label("rank")
        stmt = (
            select(Chunk, Document, Parent, rank)
            .join(Document, Chunk.document_id == Document.id)
            .outerjoin(Parent, Chunk.parent_id == Parent.id)
            .where(Document.is_archived.is_(False))
            .where(tsvector.op("@@")(tsquery))
            .order_by(rank.desc())
            .limit(top_k)
        )
        rows = self._session.execute(stmt).all()

        if not rows:
            # Fallback: simple substring match so single-word / symbol queries
            # (e.g. an invoice id) still return something to fuse.
            like = f"%{query}%"
            stmt = (
                select(Chunk, Document, Parent)
                .join(Document, Chunk.document_id == Document.id)
                .outerjoin(Parent, Chunk.parent_id == Parent.id)
                .where(Document.is_archived.is_(False))
                .where(or_(Chunk.content.ilike(like)))
                .limit(top_k)
            )
            rows = [(c, d, p, 0.5) for c, d, p in self._session.execute(stmt).all()]

        max_rank = max((float(r) for *_rest, r in rows), default=0.0) or 1.0
        results: list[RetrievedChunk] = []
        for chunk, document, parent, r in rows:
            # Normalise lexical rank to 0-1 then express as a distance so the
            # fusion layer can treat it like the cosine-distance field.
            score = min(1.0, float(r) / max_rank) if max_rank else 0.0
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=document.id,
                    source=document.source,
                    title=document.title,
                    content=chunk.content,
                    distance=1.0 - score,
                    content_type=chunk.content_type,
                    parent_id=parent.id if parent is not None else None,
                    parent_content=parent.content if parent is not None else None,
                    uploaded_at=document.created_at,
                    version=document.version,
                )
            )
        return results

    def count_documents(self) -> int:
        return self._session.query(Document).count()

    def count_chunks(self) -> int:
        return self._session.query(Chunk).count()

    def clear(self) -> int:
        """Delete all documents and their chunks. Returns documents removed."""
        count = self._session.query(Document).count()
        self._session.query(Document).delete()
        self._session.commit()
        return count

    def list_documents(self, include_archived: bool = False) -> list[dict]:
        """Return documents with chunk counts, newest first.

        By default only active (non-archived) versions are returned.
        """
        from sqlalchemy import func as sa_func

        query = (
            self._session.query(
                Document.id,
                Document.source,
                Document.title,
                Document.created_at,
                Document.version,
                Document.is_archived,
                sa_func.count(Chunk.id).label("chunk_count"),
            )
            .outerjoin(Chunk, Chunk.document_id == Document.id)
            .group_by(Document.id)
            .order_by(Document.created_at.desc())
        )
        if not include_archived:
            query = query.filter(Document.is_archived.is_(False))
        rows = query.all()
        return [
            {
                "id": r.id,
                "source": r.source,
                "title": r.title,
                "created_at": r.created_at,
                "version": r.version,
                "is_archived": r.is_archived,
                "chunk_count": r.chunk_count,
            }
            for r in rows
        ]

    def delete_by_id(self, document_id: int) -> bool:
        """Delete a single document by id. Returns True if it existed."""
        doc = self._session.get(Document, document_id)
        if doc is None:
            return False
        self._session.delete(doc)
        self._session.commit()
        return True

    def delete_by_source(self, source: str) -> int:
        """Delete documents matching a source. Returns documents removed."""
        docs = self._session.query(Document).filter(Document.source == source).all()
        for doc in docs:
            self._session.delete(doc)
        self._session.commit()
        return len(docs)
