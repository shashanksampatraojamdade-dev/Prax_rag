# prax_re_lc — Prax Retrieval Engine SDK (LangChain edition)

A **functional clone of [`prax_re`](../prax_re/README.md) built on LangChain**.
Same four guarantees, same calibration, same API shape — but it speaks LangChain
natively (`Document`, `BaseRetriever`, chat `Runnable`s) so it drops straight into
a LangChain stack:

1. **Hybrid retrieval by default** — semantic (vector) + keyword (lexical) fused
   with Reciprocal Rank Fusion (RRF).
2. **Source citations in every answer** — from each hit's `Document.metadata`.
3. **Configurable "I don't know" threshold** — declines instead of hallucinating
   when retrieval confidence is below the threshold.
4. **Contextual compression** — trims each chunk to only the query-relevant
   sentences before answering (saves tokens, boosts precision).

Everything is **pluggable** and **externally configurable**: set any parameter on
`LCREConfig` (or let it read env vars), inject your own LangChain retrievers/LLM,
*or* let the batteries-included factory open its own pgvector connection from env.

## Install / version

```bash
pip install "prax-rag[langchain]"      # core + pgvector + Vertex backends
python -c "import prax_re_lc; print(prax_re_lc.__version__)"   # 0.1.0
```

The `langchain` extra pulls `langchain-core`, `langchain-postgres` and
`langchain-google-vertexai`. Only `langchain-core` is needed if you inject your
own components — the DB/Vertex backends are imported lazily by the factory.

## Two ways to integrate

### A) Batteries included — build from environment (zero wiring)

Point it at your pgvector DB + Vertex and go. The SDK reads DB credentials and
model names from environment variables and opens the connection itself:

```python
from prax_re_lc import build_engine_from_env

engine = build_engine_from_env()                 # reads POSTGRES_*, LLM_MODEL, ...
# override anything inline:
engine = build_engine_from_env(top_k=8, idk_threshold=0.4)

result = engine.answer("How many vacation days do I get?")
print(result.answer, result.confidence, result.answerable)
for c in result.citations:
    print(c["rank"], c.get("source"), c["score"])
```

Environment variables (all optional — defaults shown):

| Env var | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | *(built from parts)* | Full SQLAlchemy URL (overrides the `POSTGRES_*` parts) |
| `POSTGRES_HOST` | `localhost` | pgvector host |
| `POSTGRES_PORT` | `5433` | pgvector port |
| `POSTGRES_DB` | `praxrag` | database name |
| `POSTGRES_USER` | `praxrag` | user |
| `POSTGRES_PASSWORD` | `praxrag` | password |
| `PGVECTOR_COLLECTION` | `prax_documents` | langchain_postgres collection |
| `VERTEX_PROJECT` | `pid-…` | GCP project |
| `VERTEX_LOCATION` | `global` | Vertex location |
| `LLM_MODEL` | `gemini-3.8-flash-lite` | Gemini chat model |
| `EMBEDDING_MODEL` | `text-embedding-005` | Vertex embedding model |
| `LLM_TEMPERATURE` | `0.2` | generation temperature |
| `ENABLE_SEMANTIC` / `ENABLE_KEYWORD` | `true` | which retrievers run |
| `RETRIEVAL_CANDIDATE_K` | `10` | candidates per retriever pre-fusion |
| `RETRIEVAL_TOP_K` | `5` | final hits after fusion |
| `RRF_K` | `60` | RRF constant |
| `ENABLE_COMPRESSION` | `true` | contextual compression on/off |
| `COMPRESSION_MAX_SENTENCES` | `4` | max sentences kept per chunk |
| `COMPRESSION_MIN_OVERLAP` | `0.1` | min query-term overlap to keep a sentence |
| `IDK_THRESHOLD` | `0.5` | min confidence to answer; below it declines |

### B) Plug in your own LangChain objects (max flexibility)

Inject any LangChain `BaseRetriever` (semantic and/or keyword) and any chat
`Runnable` (or a plain `callable(question, context) -> str`). Nothing DB-specific
is touched:

```python
from prax_re_lc import RetrievalEngine, LCREConfig
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_postgres import PGVector

emb = VertexAIEmbeddings(model_name="text-embedding-005")
store = PGVector(embeddings=emb, collection_name="docs", connection="postgresql+psycopg://…")

engine = RetrievalEngine(
    answer_model=ChatVertexAI(model="gemini-3.8-flash-lite", temperature=0.2),
    semantic=store.as_retriever(search_kwargs={"k": 10}),
    keyword=my_bm25_or_fts_retriever,      # optional; omit for semantic-only
    config=LCREConfig(top_k=5, idk_threshold=0.5),
)
result = engine.answer("How many vacation days do I get?")
```

> **Scores for confidence.** `vectorstore.as_retriever()` drops relevance scores,
> so the IDK gate falls back to rank-decayed pseudo-scores. For accurate
> confidence, either use `build_semantic_retriever()` (which preserves scores in
> `metadata['score']`) or put a 0–1 `score` into each `Document.metadata`.

## Streaming

```python
results, confidence, mode = engine.retrieve_and_gate(question)
if not confidence.answerable:
    stream(engine._config.idk_message)
else:
    for token in engine.answer_stream(question):   # compresses + streams
        stream(token)
```

## Metadata contract (optional)

The SDK reads a tiny, **optional** contract off each `Document.metadata`:

| Key | Purpose | If missing |
|---|---|---|
| `id` | dedupe identical hits across retrievers | a content hash is used |
| `score` | 0–1 relevance (higher = better) for confidence | rank-decayed pseudo-score |
| `source_type` | `semantic` / `keyword` / `hybrid` | defaults to the retriever's role |

Any other keys you attach (`source`, `title`, `uploaded_at`, `version`, …) flow
straight through into citations — nothing is required, so it works over lean KBs.

## `LCREConfig`

| Field | Default | Purpose |
|---|---|---|
| `enable_semantic` | `True` | include the vector retriever |
| `enable_keyword` | `True` | include the keyword retriever (hybrid) |
| `candidate_k` | `10` | candidates each retriever returns pre-fusion |
| `top_k` | `5` | final hits kept after fusion |
| `rrf_k` | `60` | Reciprocal Rank Fusion constant |
| `enable_compression` | `True` | contextual compression on/off |
| `compression_max_sentences` | `4` | max sentences kept per chunk |
| `compression_min_overlap` | `0.1` | min query-term overlap to keep a sentence |
| `idk_threshold` | `0.5` | min confidence to answer; below it declines |
| `idk_message` | *(sentence)* | text returned when declining |
| `temperature` | `0.2` | generation temperature (default LLM) |
| `system_prompt` | *(grounded)* | system prompt for the default prompt template |
| `vertex_project` / `vertex_location` | `pid-…` / `global` | Vertex settings |
| `llm_model` / `embedding_model` | `gemini-3.8-flash-lite` / `text-embedding-005` | models |
| `database_url` / `postgres_*` / `pgvector_collection` | *(see table above)* | DB connection |
| `llm_kwargs` / `embedding_kwargs` / `vectorstore_kwargs` | `{}` | extra kwargs forwarded to default builders |

Build it however you like: `LCREConfig()` (all defaults), `LCREConfig(top_k=8)`
(override), or `LCREConfig.from_env(idk_threshold=0.4)` (env + overrides).

### Custom contextual compressor

Pass any LangChain `BaseDocumentCompressor` to replace the built-in lexical one:

```python
from langchain.retrievers.document_compressors import EmbeddingsFilter
engine = RetrievalEngine(answer_model=..., semantic=...,
                         compressor=EmbeddingsFilter(embeddings=emb, k=4))
```

Disable compression entirely with `LCREConfig(enable_compression=False)`.

## Modules

| Module | Responsibility |
|---|---|
| `prax_re_lc/engine.py` | `RetrievalEngine` orchestrator + `AnswerResult` |
| `prax_re_lc/config.py` | `LCREConfig` (+ `from_env`, connection string) |
| `prax_re_lc/documents.py` | `Document` metadata contract helpers |
| `prax_re_lc/fusion.py` | Reciprocal Rank Fusion over `Document`s |
| `prax_re_lc/confidence.py` | Confidence scoring + IDK threshold |
| `prax_re_lc/compression.py` | Contextual compression |
| `prax_re_lc/defaults.py` | Batteries-included pgvector + Vertex builders |

## Relationship to `prax_re`

`prax_re` is dependency-free and backend-agnostic (you adapt your own objects to
small protocols). `prax_re_lc` is the same engine expressed in LangChain types,
so if your stack already uses LangChain retrievers, vector stores and chat models
you get the same hybrid + citations + IDK + compression behaviour with less glue.
