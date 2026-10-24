from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

settings = get_settings()

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    """Ensure the pgvector extension and ORM tables exist."""
    from app.db.models import Base

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS "
                "content_type VARCHAR(16) NOT NULL DEFAULT 'text'"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS "
                "parent_id INTEGER REFERENCES parents(id) ON DELETE CASCADE"
            )
        )
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_chunks_parent_id ON chunks(parent_id)")
        )
        conn.execute(
            text(
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_chunks_content_hash "
                "ON chunks(content_hash)"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64)"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS "
                "version INTEGER NOT NULL DEFAULT 1"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS "
                "is_archived BOOLEAN NOT NULL DEFAULT false"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_documents_is_archived "
                "ON documents(is_archived)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_documents_content_hash "
                "ON documents(content_hash)"
            )
        )
    logger.info("Database initialized (pgvector extension + tables ready)")


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a scoped DB session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
