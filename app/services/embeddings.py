from google.genai import types

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.vertex_client import get_vertex_client

logger = get_logger(__name__)


class EmbeddingService:
    """Generates embeddings using Vertex AI text-embedding models."""

    def __init__(self) -> None:
        self._settings = get_settings()

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        if not texts:
            return []

        client = get_vertex_client()
        response = client.models.embed_content(
            model=self._settings.embedding_model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self._settings.embedding_dim,
            ),
        )
        return [list(e.values) for e in response.embeddings]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks for storage/retrieval."""
        logger.info("Embedding %d document chunk(s)", len(texts))
        return self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        return self._embed([text], task_type="RETRIEVAL_QUERY")[0]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Embed several search queries in one call (used for multi-probe retrieval)."""
        logger.info("Embedding %d query probe(s)", len(texts))
        return self._embed(texts, task_type="RETRIEVAL_QUERY")
