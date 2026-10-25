from pydantic import BaseModel, Field
from datetime import datetime


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(None, ge=1, le=20)


class Source(BaseModel):
    rank: int
    chunk_id: int
    document_id: int
    source: str
    title: str | None = None
    content_type: str = "text"
    score: float
    preview: str
    uploaded_at: datetime | None = None
    version: int = 1


class Timings(BaseModel):
    query_transform_ms: float = 0.0
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class QueryTransformation(BaseModel):
    original: str
    rewritten: str
    expansions: list[str] = []
    search_query: str
    variations: list[str] = []
    hypothetical_document: str | None = None
    hops: list[str] = []
    search_queries: list[str] = []
    query_class: str | None = None
    route: str | None = None
    intent: str | None = None
    applied: list[str] = []


class QueryResponse(BaseModel):
    answer: str
    confidence: float = 0.0
    confidence_label: str = "low"
    answerable: bool = True
    retrieval_mode: str = "hybrid"
    sources: list[Source]
    query_transformation: QueryTransformation | None = None
    timings: Timings
