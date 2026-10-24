from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import get_session
from app.repositories.vector_store import VectorStore

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/health/db")
def health_db(session: Session = Depends(get_session)) -> dict:
    session.execute(text("SELECT 1"))
    store = VectorStore(session)
    return {
        "status": "ok",
        "documents": store.count_documents(),
        "chunks": store.count_chunks(),
    }
