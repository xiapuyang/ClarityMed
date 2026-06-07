"""Bridge to pydantic-ai.

Only one helper lives here: ``build_model(provider) -> pydantic_ai.models.Model``.
For chat, structured output, tool use, etc., import from ``pydantic_ai``
directly — wrapping ``Agent`` is intentionally not this module's job.
"""

from claritymed.core.llm.model import build_model

__all__ = ["build_model"]
