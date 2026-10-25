# prax_qt

A plug-and-play **query understanding & transformation** SDK for RAG systems.

Bring your own language model; `prax_qt` reshapes each incoming query before
retrieval:

- **Query rewriting** — rewrites vague, conversational questions into clear,
  search-friendly statements.
- **Query expansion** — enriches the query with synonyms, expanded acronyms and
  related concepts so retrieval misses fewer relevant chunks.
- **Multi-query retrieval** — generates alternative phrasings of the question so
  the host can search each and merge the hits.
- **HyDE** — drafts a hypothetical ideal answer and searches for real documents
  that look like it.

Techniques that widen retrieval (multi-query, HyDE) add extra *probes* to
`TransformedQuery.search_queries`; the host embeds each, searches, and merges.

The SDK is **backend-agnostic**: it depends only on a tiny `LanguageModel`
protocol and never imports a specific model client, web framework or database.

## Install / vendor

The package has **no third-party dependencies**. Copy the `prax_qt/` folder into
your project (or install it as a local package) and import it.

## Quick start

```python
from prax_qt import QueryTransformer, QTConfig

# 1. Adapt your model to the one-method protocol.
class MyLLM:
    def complete(self, prompt: str, *, system: str | None = None) -> str:
        return my_client.generate(system=system, user=prompt)

# 2. Compose the transformer (enable the SDK → all enabled techniques apply).
qt = QueryTransformer(model=MyLLM(), config=QTConfig(max_expansion_terms=6))

# 3. Transform, then retrieve on search_query but answer the original question.
result = qt.transform("How do I fix that login thing on the portal?")
hits   = my_retriever(result.search_query)
answer = my_llm_answer(question=result.original, context=hits)
```

`transform()` returns a `TransformedQuery`:

```python
result.original      # the user's untouched question (ground answers on this)
result.rewritten     # the clarified, search-friendly query
result.expansions    # list[str] of related terms added
result.search_query  # rewritten + expansions → the primary probe
result.variations    # list[str] of alternative phrasings (multi-query)
result.hypothetical_document  # HyDE probe passage, or None
result.hops          # ordered multi-hop sub-queries (also added as probes)
result.search_queries  # ALL probes to embed, search & merge
result.query_class   # exact_lookup | semantic | analytical | multi_hop
result.route         # chosen retrieval route (advisory)
result.intent        # fact | summary | explanation | comparison | list | other
result.applied       # e.g. ["classification", "rewriting", "multi_hop"]
result.as_dict()     # JSON-friendly dict for API responses / logging
```

## Configuration

`QTConfig` fields (every technique is an independent switch):

| Field | Default | Purpose |
|---|---|---|
| `enable_rewriting` | `True` | Apply query rewriting. |
| `enable_expansion` | `True` | Apply query expansion. |
| `enable_multi_query` | `True` | Generate alternative phrasings to search and merge. |
| `enable_hyde` | `True` | Draft a hypothetical answer and search for similar docs. |
| `enable_classification` | `True` | Label the query type (exact_lookup/semantic/analytical/multi_hop). |
| `enable_routing` | `True` | Pick the best-fit retrieval route from `routes`. |
| `enable_intent_detection` | `True` | Detect answer-format intent (fact/summary/comparison/...). |
| `enable_multi_hop` | `True` | Decompose chained questions into ordered sub-queries. |
| `max_expansion_terms` | `6` | Cap on related terms added by expansion. |
| `multi_query_count` | `3` | Number of alternative phrasings generated. |
| `max_hops` | `3` | Max sub-queries in a multi-hop plan. |
| `routes` | `["knowledge_base"]` | Candidate retrieval routes for adaptive routing. |
| `expand_rewritten` | `True` | Expand the rewritten query (vs. the raw original). |

## The contract

Implement one method:

```python
class LanguageModel(Protocol):
    def complete(self, prompt: str, *, system: str | None = None) -> str: ...
```

Any provider works — Vertex, OpenAI, Anthropic, a local model.

## Fail-safe by design

Both techniques catch model errors and degrade gracefully (rewriting → original
query, expansion → no extra terms), so a transient model issue never blocks
retrieval.
