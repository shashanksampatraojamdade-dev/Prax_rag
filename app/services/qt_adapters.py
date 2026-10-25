"""Adapters binding the backend-agnostic prax_qt SDK to this app's stack.

- ``LLMService`` already satisfies ``prax_qt.LanguageModel`` (it exposes
  ``complete``), so it is used directly as the model.
- ``settings_to_qt_config`` maps the app's pydantic Settings to a ``QTConfig``.
- ``build_query_transformer`` returns a ready ``QueryTransformer`` or ``None``
  when the master toggle is off, so callers can treat transformation as a
  drop-in optional step.
"""

from app.core.config import Settings
from app.services.llm import LLMService
from prax_qt import QTConfig, QueryTransformer


def settings_to_qt_config(settings: Settings) -> QTConfig:
    routes = [r.strip() for r in settings.query_routes.split(",") if r.strip()]
    return QTConfig(
        enable_rewriting=settings.enable_query_rewriting,
        enable_expansion=settings.enable_query_expansion,
        enable_multi_query=settings.enable_multi_query,
        enable_hyde=settings.enable_hyde,
        enable_classification=settings.enable_query_classification,
        enable_routing=settings.enable_query_routing,
        enable_intent_detection=settings.enable_intent_detection,
        enable_multi_hop=settings.enable_multi_hop,
        max_expansion_terms=settings.query_expansion_terms,
        multi_query_count=settings.multi_query_count,
        max_hops=settings.max_hops,
        routes=routes or ["knowledge_base"],
        single_call=settings.query_single_call,
        max_workers=settings.query_max_workers,
    )


def build_query_transformer(
    settings: Settings, llm: LLMService
) -> QueryTransformer | None:
    """Build a QueryTransformer, or None when transformation is disabled."""
    if not settings.enable_query_transformation:
        return None
    return QueryTransformer(model=llm, config=settings_to_qt_config(settings))
