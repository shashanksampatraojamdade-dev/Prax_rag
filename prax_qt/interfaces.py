"""Integration contract for the Prax query-transformation SDK.

Host applications plug the SDK into their own stack by providing a single
adapter: a :class:`LanguageModel` (any text-completion backend — Vertex, OpenAI,
Anthropic, a local model, ...).

The SDK owns the *techniques* (query rewriting, query expansion) and calls back
into this protocol for the one capability it needs: turning a prompt into text.
Nothing in the SDK imports a specific model client.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LanguageModel(Protocol):
    """Any text-completion backend.

    A single method keeps the contract tiny: given a prompt (and an optional
    system instruction) return the model's text response. Implementations
    should be side-effect free and reasonably low-latency, since a transform
    runs before every retrieval.
    """

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        ...
