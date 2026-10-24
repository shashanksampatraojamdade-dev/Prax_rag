import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_embedding_service, get_llm_service
from app.db.session import get_session
from app.schemas.query import QueryRequest, QueryResponse
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.services.rag import RAGService

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    session: Session = Depends(get_session),
    embeddings: EmbeddingService = Depends(get_embedding_service),
    llm: LLMService = Depends(get_llm_service),
) -> QueryResponse:
    service = RAGService(session, embeddings, llm)
    result = service.answer(question=payload.question, top_k=payload.top_k)
    return QueryResponse(**result)


@router.post(
    "/query/stream",
    summary="Query (streaming)",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Server-Sent Events stream. Each line is "
            "`data: {json}` where json has a `type` of `sources`, `token`, or "
            "`done`.",
            "content": {"text/event-stream": {}},
        }
    },
)
def query_stream(
    payload: QueryRequest,
    session: Session = Depends(get_session),
    embeddings: EmbeddingService = Depends(get_embedding_service),
    llm: LLMService = Depends(get_llm_service),
) -> StreamingResponse:
    service = RAGService(session, embeddings, llm)

    def event_generator():
        for event in service.answer_stream(
            question=payload.question, top_k=payload.top_k
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
