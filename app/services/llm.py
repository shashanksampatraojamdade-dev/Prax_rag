from collections.abc import Iterator

from google.genai import types

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.vertex_client import get_vertex_client

logger = get_logger(__name__)

SYSTEM_INSTRUCTION = (
    "You are an enterprise knowledge assistant. Answer the user's question "
    "strictly using the provided context. If the context does not contain the "
    "answer, say you don't have enough information. Be concise and cite the "
    "source numbers (e.g. [1], [2]) you used."
)


class LLMService:
    """Wraps Vertex AI Gemini text generation for grounded RAG answers."""

    def __init__(self) -> None:
        self._settings = get_settings()

    def _build_prompt(self, question: str, context: str) -> str:
        return (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the context above."
        )

    def _config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
            # We ground answers ourselves and expose no tools, so disable
            # automatic function calling — it otherwise buffers the stream.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            # Turn off "thinking" so the model emits tokens immediately instead
            # of pausing ~30s to reason and then flushing the whole answer at
            # once (which made streaming look like a single blob).
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

    def generate(self, question: str, context: str) -> str:
        client = get_vertex_client()
        logger.info("Generating answer with model=%s", self._settings.llm_model)
        response = client.models.generate_content(
            model=self._settings.llm_model,
            contents=self._build_prompt(question, context),
            config=self._config(),
        )
        return (response.text or "").strip()

    def generate_stream(self, question: str, context: str) -> Iterator[str]:
        """Yield answer text incrementally as Vertex streams it."""
        client = get_vertex_client()
        logger.info("Streaming answer with model=%s", self._settings.llm_model)
        stream = client.models.generate_content_stream(
            model=self._settings.llm_model,
            contents=self._build_prompt(question, context),
            config=self._config(),
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Generic single-shot text completion.

        Satisfies the ``prax_qt.LanguageModel`` protocol so the query
        transformation SDK can reuse this Vertex client for rewriting and
        expansion. Unlike ``generate`` it carries no RAG system instruction.
        """
        client = get_vertex_client()
        config = types.GenerateContentConfig(
            temperature=0.2,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        if system:
            config.system_instruction = system
        response = client.models.generate_content(
            model=self._settings.llm_model,
            contents=prompt,
            config=config,
        )
        return (response.text or "").strip()
