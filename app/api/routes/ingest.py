from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_embedding_service
from app.db.session import get_session
from app.schemas.documents import IngestResponse, IngestTextRequest
from app.services.embeddings import EmbeddingService
from app.services.ingestion import IngestionService
from prax_kb import UnsupportedFileError

router = APIRouter(tags=["ingestion"])


@router.post("/ingest", response_model=IngestResponse)
def ingest_text(
    payload: IngestTextRequest,
    session: Session = Depends(get_session),
    embeddings: EmbeddingService = Depends(get_embedding_service),
) -> IngestResponse:
    service = IngestionService(session, embeddings)
    try:
        result = service.ingest_text(
            text=payload.text, source=payload.source, title=payload.title
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return IngestResponse(**result)


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(..., description="PDF, DOCX, or UTF-8 text/markdown file"),
    title: str | None = Form(None, description="Optional human-readable title"),
    session: Session = Depends(get_session),
    embeddings: EmbeddingService = Depends(get_embedding_service),
) -> IngestResponse:
    if not file.filename:
        raise HTTPException(
            status_code=422,
            detail="No file received. Attach a real file in the 'file' field.",
        )

    raw = await file.read()
    title = title if title and title.strip() else None

    service = IngestionService(session, embeddings)
    try:
        result = service.ingest_file(raw, filename=file.filename, title=title)
    except UnsupportedFileError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return IngestResponse(**result)
