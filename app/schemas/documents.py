from datetime import datetime

from pydantic import BaseModel, Field


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw document text to ingest")
    source: str = Field(..., min_length=1, description="Identifier/URI of the source")
    title: str | None = Field(None, description="Optional human-readable title")


class IngestResponse(BaseModel):
    document_id: int
    source: str
    title: str | None = None
    chunks_ingested: int
    status: str = "indexed"  # indexed | updated | unchanged | duplicate
    version: int = 1
    chunks_reused: int = 0
    chunks_embedded: int = 0
    profile: str | None = None
    duplicate_of: str | None = None


class DocumentSummary(BaseModel):
    id: int
    source: str
    title: str | None = None
    created_at: datetime
    version: int = 1
    is_archived: bool = False
    chunk_count: int


class DocumentList(BaseModel):
    total: int
    documents: list[DocumentSummary]


class DeleteResponse(BaseModel):
    deleted_documents: int
    detail: str
