from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.session import get_session
from app.repositories.vector_store import VectorStore
from app.schemas.documents import DeleteResponse, DocumentList, DocumentSummary

router = APIRouter(tags=["documents"])


@router.get("/documents", response_model=DocumentList)
def list_documents(
    include_archived: bool = Query(
        False, description="Include archived (older-version) documents"
    ),
    session: Session = Depends(get_session),
) -> DocumentList:
    store = VectorStore(session)
    docs = store.list_documents(include_archived=include_archived)
    return DocumentList(
        total=len(docs),
        documents=[DocumentSummary(**d) for d in docs],
    )


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
def delete_document(
    document_id: int, session: Session = Depends(get_session)
) -> DeleteResponse:
    store = VectorStore(session)
    if not store.delete_by_id(document_id):
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    return DeleteResponse(
        deleted_documents=1, detail=f"Deleted document {document_id}"
    )


@router.delete("/documents", response_model=DeleteResponse)
def delete_documents_by_source(
    source: str = Query(..., description="Source identifier to delete"),
    session: Session = Depends(get_session),
) -> DeleteResponse:
    store = VectorStore(session)
    removed = store.delete_by_source(source)
    return DeleteResponse(
        deleted_documents=removed,
        detail=f"Deleted {removed} document(s) with source={source}",
    )


@router.post("/documents/reset", response_model=DeleteResponse)
def reset_documents(
    confirm: bool = Query(
        False, description="Must be true to wipe the entire knowledge base"
    ),
    session: Session = Depends(get_session),
) -> DeleteResponse:
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Refusing to reset. Pass confirm=true to delete all documents.",
        )
    store = VectorStore(session)
    removed = store.clear()
    return DeleteResponse(
        deleted_documents=removed,
        detail=f"Knowledge base reset. Removed {removed} document(s).",
    )
