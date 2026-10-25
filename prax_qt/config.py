"""Configuration for the Prax query-transformation SDK.

A plain dataclass so host applications can build it however they like (env vars,
pydantic settings, a config file, hard-coded defaults). Every technique is an
independent on/off switch, so teams can adopt them incrementally — enable the
SDK and every enabled method is applied to each incoming query.
"""

from dataclasses import dataclass, field


@dataclass
class QTConfig:
    # Technique switches (plug-and-play)
    enable_rewriting: bool = True
    enable_expansion: bool = True
    enable_multi_query: bool = True
    enable_hyde: bool = True
    enable_classification: bool = True
    enable_routing: bool = True
    enable_intent_detection: bool = True
    enable_multi_hop: bool = True

    # Query expansion tuning
    max_expansion_terms: int = 6

    # Multi-query retrieval tuning
    multi_query_count: int = 3

    # Multi-hop retrieval planning tuning
    max_hops: int = 3

    # Adaptive query routing: candidate retrieval targets the host supports.
    # The SDK picks the best-fit route from this list; the host acts on it.
    routes: list[str] = field(default_factory=lambda: ["knowledge_base"])

    # Performance: run all understanding techniques in ONE structured LLM call
    # instead of one call per technique (~8x fewer round-trips). Falls back to
    # the per-technique path automatically if the single call fails to parse.
    single_call: bool = True

    # Performance: when the per-technique path runs (single_call disabled or it
    # failed to parse), execute independent techniques concurrently instead of
    # one-after-another. The LLM calls are I/O-bound, so a thread pool collapses
    # ~8 sequential round-trips into two dependency "waves". Set to 1 to force
    # the old strictly-sequential behaviour.
    max_workers: int = 8

    # When True, expansion runs on the rewritten query (recommended) so the
    # added terms build on the clarified intent rather than the raw question.
    expand_rewritten: bool = True
