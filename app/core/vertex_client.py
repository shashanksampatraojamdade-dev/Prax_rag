from google import genai

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_client: genai.Client | None = None


def get_vertex_client() -> genai.Client:
    """Return a cached Vertex AI Gemini client, creating it on first use."""
    global _client
    if _client is None:
        settings = get_settings()
        logger.info(
            "Connecting to Vertex AI (project=%s, location=%s)",
            settings.vertex_project,
            settings.vertex_location,
        )
        _client = genai.Client(
            vertexai=True,
            project=settings.vertex_project,
            location=settings.vertex_location,
        )
    return _client
