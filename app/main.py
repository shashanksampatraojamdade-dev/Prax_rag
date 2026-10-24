from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import documents, health, ingest, query
from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import init_db

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s", get_settings().app_name)
    init_db()
    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Enterprise RAG service: pgVector + Vertex AI (LLM + embeddings)",
        lifespan=lifespan,
    )

    app.include_router(health.router, prefix=settings.api_prefix)
    app.include_router(ingest.router, prefix=settings.api_prefix)
    app.include_router(documents.router, prefix=settings.api_prefix)
    app.include_router(query.router, prefix=settings.api_prefix)
    return app


app = create_app()
