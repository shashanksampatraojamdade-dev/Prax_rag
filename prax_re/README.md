# prax_re — Prax Retrieval Engine SDK

A reusable, versioned, **backend-agnostic** retrieval/answer engine. Drop it on
top of any vector store, keyword index and LLM to get production-grade RAG
retrieval with three guarantees built in:

1. **Hybrid retrieval by default** — semantic (vector) + keyword (lexical)
   results fused with Reciprocal Rank Fusion (RRF).
2. **Source citations in every answer** — assembled from each hit's free-form
   metadata (path, upload date, version, … — whatever you attach).
3. **Configurable "I don't know" threshold** — declines instead of
   hallucinating when retrieval confidence is below the threshold.
4. **Contextual compression** — trims each retrieved chunk to only the
   query-relevant sentences before answering (saves tokens, boosts precision).

It imports nothing app-specific: you inject objects that satisfy small protocols
(`prax_re.interfaces`) and the SDK owns the reusable logic (fusion, confidence
gating, citation + context assembly, orchestration).

## Install / version

Packaged with the repo (`prax_re*`); `prax_re.__version__` is the SDK version
(currently `0.2.0`).

```bash
# Inside this repo it is already importable (PYTHONPATH=.).
# To use it from another project, install the package that ships it:
pip install prax-rag            # or: uv pip install prax-rag
python -c "import prax_re; print(prax_re.__version__)"
```

`prax_re` has **no third-party runtime dependencies** — it only orchestrates the
objects you inject, so it drops into any stack.

## How to integrate (step by step)

Integrating the SDK is four small steps: implement two contracts, map your hits
to `Retrieved`, build the engine, call it. The SDK never talks to your database,
embedder or LLM directly — you hand it callables and it does the rest.

### Step 1 — Implement the two contracts

The SDK needs a **retriever** (one for semantic, optionally one for keyword) and
an **answer model**. A retriever is anything with
`search(query: str, k: int) -> list[Retrieved]`; an answer model is anything with
`generate(question: str, context: str) -> str`. No base class to inherit —
duck typing against `prax_re.interfaces` is enough.

```python
from prax_re import Retrieved

class PgVectorRetriever:
    """Semantic retriever backed by your embedder + vector store."""
    def __init__(self, embedder, store):
        self._embedder, self._store = embedder, store

    def search(self, query: str, k: int) -> list[Retrieved]:
        vector = self._embedder.embed_query(query)
        rows = self._store.similarity_search(vector, top_k=k)
        return [_to_retrieved(r, "semantic") for r in rows]

class KeywordRetriever:
    """Lexical retriever backed by full-text / BM25 search."""
    def __init__(self, store):
        self._store = store

    def search(self, query: str, k: int) -> list[Retrieved]:
        rows = self._store.keyword_search(query, top_k=k)
        return [_to_retrieved(r, "keyword") for r in rows]

class MyLLM:
    def generate(self, question: str, context: str) -> str:
        return call_your_llm(system=GROUNDED_SYSTEM_PROMPT,
                             prompt=f"Context:\n{context}\n\nQuestion: {question}")
```

### Step 2 — Map your rows to `Retrieved`

`Retrieved` is the only type the SDK owns. Map each of your search hits onto it.
`score` must be **0–1 where higher is better** (semantic: `1 - cosine_distance`;
keyword: a normalised lexical rank). `metadata` is free-form — attach whatever
you want to appear in citations; the SDK requires *no* specific key.

```python
def _to_retrieved(row, source_type: str) -> Retrieved:
    return Retrieved(
        id=row.chunk_id,                       # stable id so fusion can dedupe
        content=row.text,                      # what the LLM/citation will use
        score=max(0.0, 1.0 - row.distance),    # 0..1, higher = better
        source_type=source_type,               # "semantic" | "keyword"
        metadata={                             # optional, surfaced in citations
            "source": row.path,
            "title": row.title,
            "uploaded_at": row.created_at.isoformat() if row.created_at else None,
            "version": row.version,
        },
    )
```

### Step 3 — Build the engine

```python
from prax_re import RetrievalEngine, REConfig

engine = RetrievalEngine(
    answer_model=MyLLM(),
    semantic=PgVectorRetriever(embedder, store),
    keyword=KeywordRetriever(store),     # omit (or None) for semantic-only
    config=REConfig(
        top_k=5,
        idk_threshold=0.5,               # decline below this confidence
    ),
)
```

> **Keyword retriever is optional.** Pass `keyword=None` and the engine runs
> semantic-only (no fusion). Pass both and you get hybrid retrieval for free.

### Step 4 — Ask questions

```python
result = engine.answer("How many vacation days do I get?")

result.answer            # grounded answer, or the IDK message when it declines
result.answerable        # False when confidence < idk_threshold
result.confidence        # 0..1  (+ result.confidence_label: high/medium/low)
result.retrieval_mode    # "hybrid" | "semantic" | "keyword"
result.citations         # [{rank, id, score, source_type, **your metadata}, ...]
```

### Step 5 — Wire it into a web API (FastAPI example)

```python
from fastapi import FastAPI
app = FastAPI()
engine = RetrievalEngine(answer_model=MyLLM(), semantic=..., keyword=...)

@app.post("/query")
def query(q: str):
    r = engine.answer(q)
    return {
        "answer": r.answer,
        "answerable": r.answerable,
        "confidence": r.confidence,
        "retrieval_mode": r.retrieval_mode,
        "citations": r.citations,
    }
```

### Integration checklist

- [ ] `search()` returns `Retrieved` with `score` in **0–1 (higher = better)**.
- [ ] Each `Retrieved.id` is **stable & unique** per chunk (fusion dedupes on it).
- [ ] `metadata` carries the fields you want in citations (all optional).
- [ ] `answer_model.generate` is **grounded** — instruct it to answer only from
      the provided context (the SDK supplies a numbered, citable context block).
- [ ] Pick an `idk_threshold` for your corpus (see "Tuning" below).

### Working reference implementation

This repo wires the SDK to pgvector + Vertex AI — copy it as a template:

- `app/services/re_adapters.py` — `SemanticRetriever`, `KeywordRetriever`,
  `chunk_to_retrieved()` mapping, and `build_engine()`.
- `app/services/rag.py` — uses the SDK's `reciprocal_rank_fusion` +
  `score_confidence` across multiple query probes, with streaming + decline.

## Quick start

```python
from prax_re import RetrievalEngine, REConfig, Retrieved

class MyVectorRetriever:
    def search(self, query: str, k: int) -> list[Retrieved]:
        # embed + ANN search in your store
        return [Retrieved(id=42, content="...", score=0.81,
                          source_type="semantic",
                          metadata={"source": "leave.md", "uploaded_at": "2026-10-24"})]

class MyKeywordRetriever:
    def search(self, query: str, k: int) -> list[Retrieved]:
        # full-text / BM25 search in your store
        ...

class MyLLM:
    def generate(self, question: str, context: str) -> str:
        ...

engine = RetrievalEngine(
    answer_model=MyLLM(),
    semantic=MyVectorRetriever(),
    keyword=MyKeywordRetriever(),
    config=REConfig(top_k=5, idk_threshold=0.5),
)

result = engine.answer("How many vacation days do I get?")
print(result.answer)            # grounded answer, or the IDK message
print(result.confidence, result.confidence_label, result.answerable)
print(result.retrieval_mode)    # "hybrid" | "semantic" | "keyword"
for c in result.citations:
    print(c["rank"], c.get("source"), c.get("uploaded_at"), c["score"])
```

### Streaming hosts

Use `retrieve_and_gate` to emit citations + confidence first, then stream tokens
yourself and apply the decline when `confidence.answerable` is `False`:

```python
results, confidence, mode = engine.retrieve_and_gate(question)
if not confidence.answerable:
    stream(config.idk_message)
else:
    context = engine.build_context(results)
    for token in my_llm.generate_stream(question, context):
        stream(token)
```

## Configuration (`REConfig`)

| Field | Default | Purpose |
|---|---|---|
| `enable_semantic` | `True` | Include the vector retriever |
| `enable_keyword` | `True` | Include the keyword retriever (hybrid) |
| `candidate_k` | `10` | Candidates each retriever returns pre-fusion |
| `top_k` | `5` | Final hits kept after fusion |
| `rrf_k` | `60` | Reciprocal Rank Fusion constant |
| `enable_compression` | `True` | Contextual compression (trim chunks to relevant sentences) |
| `compression_max_sentences` | `4` | Max sentences kept per chunk when compressing |
| `compression_min_overlap` | `0.1` | Min fraction of query terms a sentence must share to be kept |
| `idk_threshold` | `0.5` | Min confidence to answer; below it declines |
| `idk_message` | *(sentence)* | Text returned when declining |

### Contextual compression

Before building the prompt, the engine trims each retrieved chunk to only the
sentences relevant to the question (the rest is noise for *this* query). This
**saves tokens** (cheaper, faster generations) and **boosts precision** (the
model isn't distracted by irrelevant text). It runs automatically inside
`engine.answer()` — for streaming hosts, call `engine.compress(query, results)`
on the hits before `build_context`:

```python
results, confidence, mode = engine.retrieve_and_gate(question)
context = engine.build_context(engine.compress(question, results))
```

The built-in compressor is purely lexical (no extra model calls, zero added
latency). To use a semantic/LLM-based compressor instead, inject a callable:

```python
engine = RetrievalEngine(
    answer_model=MyLLM(), semantic=..., keyword=...,
    compressor=lambda query, content: my_llm.extract_relevant(query, content),
)
```

Disable it with `REConfig(enable_compression=False)` to pass full chunks through.
The pre-compression text is preserved in each hit's
`metadata['uncompressed_content']`.

### Tuning `idk_threshold`

Confidence is derived from retrieval relevance, so the right cut-off depends on
your embedder/corpus. A quick way to calibrate: log `result.confidence` for a
handful of *known-answerable* and *known-unanswerable* questions and set the
threshold between the two clusters. Typical starting point: **0.5** (clearly
unrelated queries land ~0.45, genuine hits ~0.65+). Set `0` to always answer and
rely solely on the LLM's own grounding.

### Config via environment (this app)

When embedded in the Prax RAG service, every field maps to an env var:

| Env var | Field |
|---|---|
| `ENABLE_HYBRID_RETRIEVAL` | fuse semantic + keyword |
| `ENABLE_KEYWORD_RETRIEVAL` | `enable_keyword` |
| `RETRIEVAL_CANDIDATE_K` | `candidate_k` |
| `RETRIEVAL_TOP_K` | `top_k` |
| `RRF_K` | `rrf_k` |
| `ENABLE_COMPRESSION` | `enable_compression` |
| `COMPRESSION_MAX_SENTENCES` | `compression_max_sentences` |
| `COMPRESSION_MIN_OVERLAP` | `compression_min_overlap` |
| `IDK_THRESHOLD` | `idk_threshold` |

## Why these choices

- **RRF** combines rankings without needing the retrievers' scores on the same
  scale (cosine similarity vs lexical rank). Hits found by *both* retrievers
  rise to the top — exactly the hybrid-recall win.
- **Confidence** weights the best match (is anything strongly relevant?) and the
  mean of the top matches (is it corroborated?). The threshold turns that into a
  safe decline so the engine never answers past its evidence.
- **Contextual compression** keeps only the query-relevant sentences of each
  chunk, cutting prompt tokens and removing distractor text — cheaper, faster,
  and more precise answers, with no extra model call by default.
- **Dependency inversion** keeps the SDK reusable: swap pgvector for FAISS, or
  Vertex for OpenAI, without touching `prax_re`.

## Modules

| Module | Responsibility |
|---|---|
| `prax_re/engine.py` | `RetrievalEngine` orchestrator + `AnswerResult` |
| `prax_re/interfaces.py` | `Retrieved` + `Retriever` / `AnswerModel` protocols |
| `prax_re/fusion.py` | Reciprocal Rank Fusion |
| `prax_re/compression.py` | Contextual compression (trim chunks to relevant sentences) |
| `prax_re/confidence.py` | Confidence scoring + IDK threshold |
| `prax_re/config.py` | `REConfig` |
