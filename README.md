# Prax RAG — Enterprise RAG Application

Production-style Retrieval-Augmented Generation service built with a clean, layered
architecture:

- **pgVector** (PostgreSQL + `pgvector`) as the vector store
- **Vertex AI Gemini** for answer generation
- **Vertex AI text-embedding** models for document & query embeddings
- **FastAPI** HTTP API

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│  API layer (FastAPI)   /health  /ingest  /ingest/file /query│
├────────────────────────────────────────────────────────────┤
│  Service layer                                              │
│   • chunking      • ingestion     • rag (orchestration)     │
│   • embeddings (Vertex)           • llm (Vertex Gemini)     │
├────────────────────────────────────────────────────────────┤
│  Repository layer   VectorStore (cosine similarity search)  │
├────────────────────────────────────────────────────────────┤
│  Data layer   Postgres + pgvector  (Document, Chunk ORM)    │
└────────────────────────────────────────────────────────────┘
Core: config (pydantic-settings), logging, Vertex client singleton
```

### Request flow
- **Ingest:** text → chunking → Vertex embeddings → pgvector (`chunks.embedding`)
- **Query:** question → Vertex query embedding → pgvector cosine search (top-k) →
  context assembly → Vertex Gemini grounded answer with citations

## Project layout

```
app/
  core/        config.py, logging.py, vertex_client.py
  db/          session.py, models.py
  repositories/vector_store.py
  services/    chunking.py, embeddings.py, llm.py, ingestion.py, rag.py
  schemas/     documents.py, query.py
  api/         deps.py, routes/{health,ingest,query}.py
  main.py      app factory
sql/init.sql   enable pgvector extension
docker-compose.yml   pgvector/pgvector:pg17
scripts/seed.py      sample data loader
vertex_run.py        standalone Vertex connectivity check
```

## Documentation

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — system design, request flows,
  data model, module map.
- **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** — setup, conventions, and the
  code-review checklist.

The config/routes/module/dependency tables in those guides are **generated from
the source** by `scripts/gen_docs.py`. After changing code, run `make docs` to
refresh them; `make check` fails if they are stale (use it in CI).

## Prerequisites

- Python 3.12+, [`uv`](https://github.com/astral-sh/uv), Docker
- Authenticated access to Vertex AI (Application Default Credentials):
  `gcloud auth application-default login`

## Setup

```bash
# 1. Install dependencies into the project venv
uv pip install -e .

# 2. Configure environment
cp .env.example .env     # adjust project/model/db values as needed

# 3. Start pgvector
docker compose up -d

# 4. (optional) Verify Vertex connectivity
.venv/bin/python vertex_run.py

# 5. Seed sample documents
.venv/bin/python scripts/seed.py

# 6. Run the API
.venv/bin/uvicorn app.main:app --reload
```

Open http://localhost:8000/docs for the interactive API.

## Chatbot UI (Streamlit)

A Streamlit chatbot provides a knowledge-base sidebar (upload PDF/DOCX/text,
list & delete documents, reset KB) and a streaming chat window.

```bash
uv pip install -e '.[ui]'     # install UI deps (streamlit, requests)
make run                      # terminal 1: start the API
make ui                       # terminal 2: start the UI  → http://localhost:8501
```

The UI talks to the API over HTTP (`PRAX_API_URL`, default
`http://localhost:8000/api/v1`) and streams answers token-by-token via the
`/query/stream` SSE endpoint, showing retrieved sources under each reply.

## API

| Method | Path                  | Description                            |
|--------|-----------------------|----------------------------------------|
| GET    | `/api/v1/health`      | Liveness                               |
| GET    | `/api/v1/health/db`   | DB check + document/chunk counts       |
| POST   | `/api/v1/ingest`      | Ingest raw text                        |
| POST   | `/api/v1/ingest/file` | Ingest an uploaded PDF, DOCX, or UTF-8 text file |
| GET    | `/api/v1/documents`   | List ingested documents + chunk counts |
| DELETE | `/api/v1/documents/{id}` | Delete one document by id           |
| DELETE | `/api/v1/documents?source=` | Delete documents by source        |
| POST   | `/api/v1/documents/reset?confirm=true` | Wipe all documents        |
| POST   | `/api/v1/query`       | Ask a grounded question                |
| POST   | `/api/v1/query/stream` | Ask a question, streamed as SSE (sources + tokens) |

### Examples

```bash
curl -X POST localhost:8000/api/v1/ingest \
  -H 'Content-Type: application/json' \
  -d '{"text":"Prax uses pgvector for RAG.","source":"notes.md","title":"Notes"}'

curl -X POST localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What vector database does Prax use?"}'
```

## Configuration

All settings are environment-driven (`app/core/config.py`). Key values:

| Variable          | Default               | Purpose                        |
|-------------------|-----------------------|--------------------------------|
| `VERTEX_PROJECT`  | pid-...-01            | Vertex AI project              |
| `LLM_MODEL`       | gemini-3.8-flash      | Generation model               |
| `EMBEDDING_MODEL` | text-embedding-005    | Vertex embedding model         |
| `EMBEDDING_DIM`   | 768                   | Vector dimension (pgvector)    |
| `POSTGRES_PORT`   | 5433                  | Host port for pgvector         |
| `CHUNK_SIZE`      | 800                   | Chunk size (chars)             |
| `RETRIEVAL_TOP_K` | 5                     | Chunks retrieved per query     |

> If you change `EMBEDDING_MODEL`, update `EMBEDDING_DIM` to match the model's
> output dimensionality before ingesting.
