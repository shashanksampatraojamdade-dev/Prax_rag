from functools import lru_cache

from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


@lru_cache
def get_llm_service() -> LLMService:
    return LLMService()
