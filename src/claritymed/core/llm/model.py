"""Bridge from our ``ProviderConfig`` to a pydantic-ai ``Model``.

This is a one-function module on purpose. Everything else — env-var lookup
for stock providers, default base URLs, wire-format dispatch, retries,
message graph, structured output — already lives in pydantic-ai. Wrapping
more than the catalog → ``Model`` step would just re-create the library.

Two paths:

1. **No ``base_url``** → ``infer_model(provider.model)``. pydantic-ai picks
   the right ``Model + Provider`` from the ``"<prefix>:<model>"`` string
   and reads the conventional env var for the API key
   (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ``DEEPSEEK_API_KEY``,
   ``MOONSHOTAI_API_KEY``, ``ALIBABA_API_KEY``/``DASHSCOPE_API_KEY``,
   ``OPENROUTER_API_KEY``, ``GEMINI_API_KEY``). A missing key surfaces as
   pydantic-ai's own ``UserError``.

2. **``base_url`` set** → construct ``OllamaProvider(base_url=..., api_key=...)``
   and wrap it in ``OpenAIChatModel``. We pick ``OllamaProvider`` because
   it is the only pydantic-ai Provider that does not require an API key —
   which makes it the right base class for any OpenAI-compatible local
   server (Ollama, MLX, llama.cpp, LM Studio, …) regardless of whether
   the server needs auth. If ``api_key_env`` is set on the config, we
   read the env var and pass it through; a missing value raises
   ``MissingApiKeyError`` (fail loud, do not silently fall back to
   ``OllamaProvider``'s placeholder).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pydantic_ai.models import infer_model
from pydantic_ai.settings import ModelSettings

from claritymed.core.schemas import ProviderConfig
from claritymed.errors import MissingApiKeyError

if TYPE_CHECKING:
    from pydantic_ai.models import Model


def build_model(provider: ProviderConfig) -> "Model":
    """Construct a pydantic-ai ``Model`` from a catalog entry."""
    if provider.base_url is None:
        model = infer_model(provider.model)
    else:
        api_key = _resolve_api_key(provider)

        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.ollama import OllamaProvider

        model = OpenAIChatModel(
            provider.model,
            provider=OllamaProvider(base_url=provider.base_url, api_key=api_key),
        )

    if os.environ.get("CLARITYMED_DEBUG"):
        from claritymed.core.observability.llm_logger import LoggingModel

        return LoggingModel(model)
    return model


def build_model_settings(provider: ProviderConfig) -> ModelSettings | None:
    """Translate catalog defaults into a pydantic-ai ``ModelSettings``.

    Returns ``None`` when the catalog entry sets no overrides, so callers
    can omit ``model_settings`` from the ``Agent`` constructor entirely.
    Right now the only knob is ``thinking`` — pydantic-ai's unified
    reasoning field, mapped to each vendor's native shape by the Model
    layer (``anthropic_thinking``, ``openai_reasoning_effort``,
    ``google_thinking_config``, …). Silently ignored on models that
    don't support reasoning, which is fine — the same setting can
    decorate ``claude-sonnet-4-5`` and ``qwen3:14b`` without branching.

    Per-request overrides belong at the orchestrator: merge the dict
    returned here with the per-call value before handing it to ``Agent``.
    """
    if provider.thinking is None:
        return None
    return ModelSettings(thinking=provider.thinking)


def _resolve_api_key(provider: ProviderConfig) -> str | None:
    """Read ``api_key_env`` from the environment, fail loud when declared-but-unset."""
    if provider.api_key_env is None:
        return None
    key = os.environ.get(provider.api_key_env)
    if not key:
        raise MissingApiKeyError(
            f"provider {provider.id!r} declares api_key_env="
            f"{provider.api_key_env!r}, but it is unset"
        )
    return key
