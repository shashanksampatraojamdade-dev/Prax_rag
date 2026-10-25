"""Configuration for the Prax retrieval-engine SDK (``prax_re``).

A plain dataclass so host applications can build it however they like (env vars,
pydantic settings, a config file, hard-coded defaults). The engine is designed
to be reusable across knowledge bases: every behaviour is an explicit, tunable
knob with safe defaults.
"""

from dataclasses import dataclass


@dataclass
class REConfig:
    # --- Hybrid retrieval ------------------------------------------------- #
    # Hybrid = semantic (vector) + keyword (lexical) retrieval, fused together.
    # Both are ON by default; disable either to fall back to a single strategy.
    enable_semantic: bool = True
    enable_keyword: bool = True

    # Number of candidates each retriever returns before fusion, and the final
    # number kept after fusion.
    candidate_k: int = 10
    top_k: int = 5

    # Reciprocal Rank Fusion constant. Larger values flatten the contribution
    # of top ranks; 60 is the value from the original RRF paper and a good
    # default for combining heterogeneous retrievers.
    rrf_k: int = 60

    # --- Contextual compression ------------------------------------------ #
    # Trim retrieved chunks to only the sentences relevant to the query before
    # answering (saves tokens, boosts precision). ON by default; set
    # ``enable_compression=False`` to pass full chunks through unchanged.
    enable_compression: bool = True
    # Max sentences kept per chunk, and the minimum fraction of query terms a
    # sentence must share to count as relevant.
    compression_max_sentences: int = 4
    compression_min_overlap: float = 0.1

    # --- "I don't know" threshold ---------------------------------------- #
    # Minimum answer confidence (0-1) required to generate an answer. Below it
    # the engine declines instead of risking a hallucination. Set to 0 to always
    # answer. ``idk_message`` is what the engine returns when it declines.
    idk_threshold: float = 0.5
    idk_message: str = (
        "I don't have enough information in the knowledge base to answer that "
        "confidently."
    )
