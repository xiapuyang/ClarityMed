"""Bridge to pydantic-ai.

Two helpers live here:

* ``build_model(provider) -> pydantic_ai.models.Model`` — the catalog row
  becomes a ready-to-use Model.
* ``build_model_settings(provider) -> pydantic_ai.settings.ModelSettings | None``
  — catalog-level defaults (``thinking`` for now) become a ``ModelSettings``
  dict the caller passes to ``Agent``.

For chat, structured output, tool use, etc., import from ``pydantic_ai``
directly — wrapping ``Agent`` is intentionally not this module's job.
"""

from claritymed.core.llm.model import build_model, build_model_settings

__all__ = ["build_model", "build_model_settings"]
