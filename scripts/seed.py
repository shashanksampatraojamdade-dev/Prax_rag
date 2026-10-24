"""Seed the knowledge base with a few sample enterprise documents.

Usage:
    .venv/bin/python scripts/seed.py
"""

from app.db.session import SessionLocal, init_db
from app.services.embeddings import EmbeddingService
from app.services.ingestion import IngestionService

SAMPLE_DOCS = [
    {
        "source": "handbook/pto-policy.md",
        "title": "Paid Time Off Policy",
        "text": (
            "Full-time employees at Prax accrue 20 days of paid time off (PTO) per "
            "year. PTO accrues monthly and unused days up to 5 may be carried over "
            "into the next calendar year. Requests must be submitted at least two "
            "weeks in advance through the HR portal and require manager approval."
        ),
    },
    {
        "source": "infra/vector-db.md",
        "title": "Vector Database Standard",
        "text": (
            "Prax standardizes on PostgreSQL with the pgvector extension as the "
            "approved vector store for retrieval augmented generation workloads. "
            "Embeddings are generated with Vertex AI text-embedding models and "
            "similarity search uses cosine distance. HNSW indexes are recommended "
            "for collections larger than one million chunks."
        ),
    },
    {
        "source": "ai/llm-guidelines.md",
        "title": "LLM Usage Guidelines",
        "text": (
            "All generative AI features at Prax must use Vertex AI Gemini models "
            "through the central platform project. Responses to end users must be "
            "grounded in retrieved context and cite their sources. Temperature "
            "should be kept at or below 0.3 for factual enterprise assistants."
        ),
    },
]


def main() -> None:
    init_db()
    embeddings = EmbeddingService()
    session = SessionLocal()
    try:
        service = IngestionService(session, embeddings)
        for doc in SAMPLE_DOCS:
            result = service.ingest_text(
                text=doc["text"], source=doc["source"], title=doc["title"]
            )
            print(f"Ingested: {result}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
