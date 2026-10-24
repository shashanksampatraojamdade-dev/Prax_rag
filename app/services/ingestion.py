"""Thin host-side wrapper around the prax_kb KnowledgeBase SDK.

All knowledge-base techniques live in the backend-agnostic ``prax_kb`` package.
This service just wires the SDK to this app's Vertex embedder and pgvector store,
and exposes the dict-returning API the routes/CLI already expect.
"""

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.repositories.vector_store import VectorStore
from app.services.embeddings import EmbeddingService
from app.services.kb_adapters import PgVectorKnowledgeStore, settings_to_kb_config
from prax_kb import KnowledgeBase

logger = get_logger(__name__)


class IngestionService:
    """Pipeline: raw text/file -> prax_kb techniques -> pgvector."""

    def __init__(self, session: Session, embeddings: EmbeddingService) -> None:
        settings = get_settings()
        store = PgVectorKnowledgeStore(VectorStore(session))
        self._kb = KnowledgeBase(
            embedder=embeddings,
            store=store,
            config=settings_to_kb_config(settings),
        )

    def ingest_text(
        self, text: str, source: str, title: str | None = None
    ) -> dict:
        return self._kb.ingest_text(text, source=source, title=title).as_dict()

    def ingest_file(
        self, data: bytes, filename: str, title: str | None = None
    ) -> dict:
        return self._kb.ingest_file(data, filename=filename, title=title).as_dict()
