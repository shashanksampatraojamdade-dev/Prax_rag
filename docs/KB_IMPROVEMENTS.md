# Knowledge Base Improvements

> **Scope:** This is the single, dedicated log of changes made to improve
> retrieval **quality** of the knowledge base — each with the problem it solves
> and the justification. It is intentionally kept **separate from the
> architecture/developer guides** (`ARCHITECTURE.md`, `DEVELOPMENT.md`), which
> describe the system as-built rather than the reasoning behind KB tuning.
>
> When you add a new KB technique, append a new `##` section here using the
> **Problem / Change / Justification** format. The configuration table below is
> regenerated automatically by `make docs`, so keep tuning knobs in
> `app/core/config.py` and they will surface here.

## Current KB configuration (auto-generated)

<!-- AUTOGEN:kb_settings -->
| Env var | Default | Purpose |
|---|---|---|
| `CHUNK_STRATEGY` | `semantic` | Chunking strategy: 'semantic' (embedding breakpoints) or 'recursive'. |
| `SEMANTIC_BREAKPOINT_PERCENTILE` | `90.0` | Sentence-distance percentile at which semantic chunking starts a new chunk. |
| `ENABLE_OCR` | `True` | Run RapidOCR on scanned pages and images embedded in text PDFs. |
| `ENABLE_PARENT_CHILD` | `True` | Retrieve small child chunks but feed their larger parent block to the LLM. |
| `PARENT_CHUNK_SIZE` | `2000` | Max characters per parent block (broader context for the LLM). |
| `ENABLE_DEDUPLICATION` | `True` | Skip ingesting a document whose content is identical to an existing one. |
| `ENABLE_VERSIONING` | `True` | Version documents by source and archive older versions so retrieval ignores them. |
| `ENABLE_INCREMENTAL` | `True` | Reuse embeddings for unchanged chunks when re-indexing a modified document. |
| `ENABLE_ADAPTIVE_CHUNKING` | `True` | Pick smaller chunks for dense/legal docs and larger chunks for narrative docs. |
| `CHUNK_SIZE` | `800` | Max characters per chunk. |
| `CHUNK_OVERLAP` | `120` | Character overlap carried between adjacent chunks. |
| `RETRIEVAL_TOP_K` | `5` | Number of nearest chunks retrieved per query. |
<!-- /AUTOGEN:kb_settings -->

---

## 1. Layout-aware parsing

**Problem.** A plain text extractor reads a PDF left-to-right across the whole
page. For multi-column layouts (financial reports, papers) this interleaves the
columns and destroys reading order; headings, sections and tables collapse into
an undifferentiated blob, so retrieved chunks are often gibberish.

**Change.** PDF extraction uses **PyMuPDF4LLM** (`pymupdf4llm.to_markdown`),
which detects columns, reconstructs reading order, preserves heading levels, and
emits tables as Markdown. DOCX extraction (`python-docx`) walks the document in
order, mapping heading styles to Markdown headings and converting tables to
Markdown. Chunking is **heading-aware**: the document is segmented by its
heading tree and every chunk is prefixed with its heading breadcrumb
(e.g. `Report > Revenue`).

**Justification.** Preserving structure keeps semantically related text together
and gives each chunk a structural "address". This measurably improved retrieval
on two-column PDFs (a fact in the right-hand column is now retrieved correctly
instead of being scrambled with the left column).

**Files.** `app/services/loaders.py`, `app/services/chunking.py`.

---

## 2. Semantic chunking

**Problem.** Fixed-size (recursive) chunking cuts text at arbitrary character
offsets. A single topic can be split across two chunks, or two unrelated topics
packed into one, which dilutes the embedding and hurts retrieval precision.

**Change.** Added an embedding-driven **semantic breakpoint** strategy
(`chunk_strategy="semantic"`, the default). Within each heading section the body
is split into sentences, each sentence is embedded, and the cosine distance
between consecutive sentences is computed. A new chunk starts wherever the
distance exceeds a percentile threshold (`SEMANTIC_BREAKPOINT_PERCENTILE`) — i.e.
at a genuine topic shift — while still respecting `chunk_size` as an upper bound.
If embeddings are unavailable the code falls back to the recursive splitter, so
ingestion never fails because of this feature.

**Justification.** Breaking at semantic boundaries produces topically coherent
chunks whose embeddings are sharper, improving both retrieval recall and the
quality of citations. Heading breadcrumbs and the size cap are preserved, so the
benefit is additive to layout-aware parsing.

**Files.** `app/services/chunking.py` (`_semantic_split`, `_sentence_split`),
`app/services/ingestion.py` (passes the embedder), `app/core/config.py`.

---

## 3. Table-aware retrieval

**Problem.** Tables carry dense, highly structured facts (figures per region,
per period, etc.). If a table is split by a size-based chunker, rows get
separated from their header and from each other, so a query like "APAC growth
rate" can no longer be answered from the fragments.

**Change.** Markdown tables are detected during chunking and emitted as **atomic
chunks that are never split**, tagged with `content_type = "table"` (a new
column on the `chunks` table). The content type is propagated end-to-end:
`VectorStore.add_document` / `similarity_search` → `RetrievedChunk` → the RAG
`sources` payload and API schema. The LLM context marks table chunks explicitly
so the model treats them as structured data.

**Justification.** Keeping a table intact preserves the header↔cell
relationships the answer depends on. In testing, a query about a specific
region's growth rate retrieved the full table as the top-ranked source and the
model answered correctly, whereas a split table could not.

**Files.** `app/services/chunking.py` (`_iter_blocks`), `app/db/models.py`
(+`content_type`), `app/db/session.py` (`ALTER TABLE ... ADD COLUMN IF NOT
EXISTS`), `app/repositories/vector_store.py`, `app/services/rag.py`,
`app/schemas/query.py`.

---

## 4. OCR for scanned PDFs and embedded images

**Problem.** Scanned documents and image-only pages contain **no extractable
text**, so they contribute nothing to the knowledge base. Equally, a normal text
PDF may embed an image (a chart, screenshot or signature) whose text is
invisible to the parser.

**Change.** Installed **RapidOCR** (`rapidocr-onnxruntime` + `onnxruntime`).
PyMuPDF4LLM automatically OCRs pages that have no extractable text (fully
scanned pages). On top of that, `loaders._ocr_images` renders each image
embedded in a *text* page and runs RapidOCR over it, appending the recovered
text under an `## Image OCR (page N)` heading so it becomes searchable too. OCR
is gated by `ENABLE_OCR` and degrades gracefully (a missing dependency or a bad
image is logged and skipped rather than failing ingestion).

**Justification.** This makes scanned archives and image-embedded reports
first-class members of the knowledge base. Verified on a synthetic scanned
invoice (text fully recovered) and on a text PDF with an embedded chart image
(the chart's figures were recovered and became retrievable).

**Files.** `app/services/loaders.py` (`_ocr_images`, `_extract_pdf`),
`app/core/config.py` (`ENABLE_OCR`), `pyproject.toml` (OCR dependencies).

---

## 5. Parent-child chunking

**Problem.** There is a tension in chunk sizing. **Small** chunks embed and
retrieve precisely (a single sentence matches a specific query cleanly), but
they are too narrow to answer from — the LLM lacks surrounding context. **Large**
chunks carry context but their embeddings are diluted across many topics, so
retrieval becomes imprecise.

**Change.** Decoupled the unit of *retrieval* from the unit of *context*. During
ingestion each heading section is split into a larger **parent** block
(`PARENT_CHUNK_SIZE`), and each parent is further split into small **child**
chunks (`CHUNK_SIZE`, semantic where possible). Only the children are embedded
and searched. At query time the matching children are resolved to their parent
blocks, **deduplicated**, and the parent text is what gets sent to the LLM —
while citations and source previews still point at the precise child. New
`parents` table + `chunks.parent_id`; retrieval joins the parent and the RAG
context builder expands children to parents. Gated by `ENABLE_PARENT_CHILD`.

**Justification.** This gives the best of both: precise matching on tight child
embeddings, plus full-context answers from the parent. Verified with a security
policy: the query matched the single sentence "The server password must be
changed every 90 days", but the LLM received the entire "Security & Compliance"
section, and two sibling children sharing that parent collapsed into one context
block cited as `[1, 2]`.

**Files.** `app/services/chunking.py` (`build_parent_child`, `ParentChunk`),
`app/db/models.py` (`Parent` + `chunks.parent_id`), `app/db/session.py`
(migration), `app/repositories/vector_store.py`
(`add_document_with_parents`, parent join), `app/services/ingestion.py`,
`app/services/rag.py` (`_build_context` parent expansion + dedup),
`app/core/config.py`.

---

## 6. Deduplication

**Problem.** The same document often arrives many times — three employees upload
the identical "2026 Vacation Policy.pdf", or the same file is re-submitted. Each
copy wastes storage, inflates embedding cost, and pollutes retrieval with
redundant near-identical results.

**Change.** Every document is fingerprinted with a whitespace/case-normalised
**SHA-256** (`content_hash`) of its extracted text, stored on `documents`. On
ingest:
- If an **active** document with the same hash exists under the *same* source,
  ingestion is skipped and returns `status="unchanged"`.
- If an active document with the same hash exists under a *different* source, it
  is treated as a duplicate (`status="duplicate"`, pointing at the original) and
  not re-indexed.
Identical child chunks within a parent are also dropped before embedding.

**Justification.** Exact-hash matching reliably collapses true duplicates
without false positives, saving storage and Vertex embedding calls and keeping
retrieval results clean. Verified: re-uploading an identical file returned
`unchanged`; the same content under a new filename returned `duplicate`.

**Files.** `app/core/hashing.py`, `app/db/models.py` (`documents.content_hash`,
`chunks.content_hash`), `app/services/ingestion.py`,
`app/repositories/vector_store.py` (`get_active_document_by_hash`).

---

## 7. Document versioning

**Problem.** Policies change. If "Remote Work Policy v2" is uploaded but v1 stays
searchable, the assistant may cite a rule that was retired years ago. Stale
content must not be retrievable once superseded.

**Change.** A document's `source` identifies it logically. When new content is
ingested for an existing source, the previous active row is **archived**
(`is_archived=true`) and the new row is stored with an incremented `version`.
Similarity search filters to `is_archived = false`, so only the current version
is ever retrieved; old versions are retained for audit and can be listed with
`GET /documents?include_archived=true`.

**Justification.** Archiving rather than hard-deleting keeps history while
guaranteeing the LLM only sees current information. Verified: after uploading a
modified policy, the query "how many days per week can I work remotely?" returned
the **v2** answer ("four days") and the archived v1 ("three days") was excluded.

**Files.** `app/db/models.py` (`documents.version`, `documents.is_archived`),
`app/repositories/vector_store.py` (`archive_source`,
`get_active_document_by_source`, retrieval filter, `list_documents`),
`app/services/ingestion.py`, `app/api/routes/documents.py`.

---

## 8. Incremental indexing

**Problem.** Re-indexing a large corpus from scratch because one document changed
by a sentence is wasteful — it re-embeds thousands of unchanged chunks and costs
time and money.

**Change.** Change-detection operates at two levels. Document level: if the new
content hash equals the current version's hash, ingestion is skipped entirely.
Chunk level: each child chunk carries its own `content_hash`; when re-indexing a
modified document, embeddings for **unchanged** children are reused from the
prior version (`get_embeddings_by_hashes`) and only new/changed children are sent
to Vertex. The response reports `chunks_reused` vs `chunks_embedded`.

**Justification.** Only the delta is embedded, so updating a lightly-edited
document is fast and cheap while retrieval stays fully up to date. Verified:
editing one section of a 3-chunk policy re-embedded 2 chunks and **reused 1**.

**Files.** `app/db/models.py` (`chunks.content_hash`),
`app/repositories/vector_store.py` (`get_embeddings_by_hashes`),
`app/services/ingestion.py`.

---

## 9. Adaptive chunk sizing

**Problem.** One chunk size does not fit all documents. Dense legal contracts
need small chunks so a single clause can be isolated; narrative articles need
large chunks so the flow of an argument isn't fragmented. A fixed size hurts one
or the other.

**Change.** A lightweight profiler (`profile_document`) inspects the text —
average sentence length, ratio of numbered/clause lines, and legal-term density —
and classifies it as **dense**, **narrative**, or **default**. Dense documents
get roughly half-size child/parent chunks (isolate clauses); narrative documents
get roughly double-size chunks (preserve flow); otherwise the configured defaults
apply. The chosen profile is returned on the ingest response and logged.

**Justification.** Matching granularity to content improves retrieval precision
for contracts and context coherence for prose, without any manual tuning per
upload. Verified: a clause-numbered agreement was detected as `dense`
(clause ratio 0.85, high legal density) and chunked at child=400/parent=1000
instead of the 800/2000 defaults.

**Files.** `app/services/profiling.py`, `app/services/ingestion.py`,
`app/core/config.py` (`ENABLE_ADAPTIVE_CHUNKING`).
