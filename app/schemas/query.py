from pydantic import BaseModel, Field


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


class Timings(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    timings: Timings
