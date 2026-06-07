"""Unified async chat client over pydantic-ai's ``Model`` abstraction.

One ``LLMClient`` per ``ProviderConfig``. Construction picks the right
pydantic-ai ``Model`` + ``Provider`` pair based on ``provider.api``:

    api = openai     ->  OpenAIChatModel + OpenAIProvider(base_url, api_key)
    api = anthropic  ->  AnthropicModel  + AnthropicProvider(base_url, api_key)

Why pydantic-ai's low-level ``direct.model_request`` instead of ``Agent``:
``Agent`` is built around the system_prompt + tools + structured_output loop.
``LLMClient`` is the layer *below* that — a plain "send messages, get text"
boundary. ``direct.model_request`` is the supported public API for that, and
it leaves the door open for the orchestrator to wrap an ``Agent`` on top
when structured outputs (``GroundedAnswer``) come online.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pydantic_ai.direct import model_request
from pydantic_ai.messages import (
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.settings import ModelSettings

from claritymed.core.llm.request import ChatRequest, ChatResponse, Usage
from claritymed.core.schemas import ProviderConfig

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.models import Model

# pydantic-ai's providers require *some* api_key string even for local servers
# that ignore it. Use a recognisable placeholder so any unexpected leak in a
# log is obvious.
LOCAL_PLACEHOLDER_KEY = "local-no-key-required"


class MissingApiKeyError(RuntimeError):
    """Provider declared an ``api_key_env`` but the env var is unset.

    Raised at ``LLMClient`` construction (fail fast). A cloud provider is
    useless without its key, so deferring to request time would just hide a
    misconfiguration behind a more confusing error.
    """


class LLMClient:
    """Thin async chat wrapper around one provider."""

    def __init__(self, provider: ProviderConfig) -> None:
        self.provider = provider
        api_key = self._resolve_key(provider)
        self._model = self._build_model(provider, api_key)

    @staticmethod
    def _resolve_key(provider: ProviderConfig) -> str:
        if provider.api_key_env is None:
            return LOCAL_PLACEHOLDER_KEY
        key = os.environ.get(provider.api_key_env)
        if not key:
            raise MissingApiKeyError(
                f"provider {provider.id!r} needs ${provider.api_key_env}, "
                "but it is unset"
            )
        return key

    @staticmethod
    def _build_model(provider: ProviderConfig, api_key: str) -> "Model":
        # Imported lazily so test environments without the SDK extras installed
        # can still import ``LLMClient`` (and only fail at construction).
        if provider.api == "openai":
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider

            return OpenAIChatModel(
                provider.model,
                provider=OpenAIProvider(api_key=api_key, base_url=provider.base_url),
            )
        if provider.api == "anthropic":
            from pydantic_ai.models.anthropic import AnthropicModel
            from pydantic_ai.providers.anthropic import AnthropicProvider

            return AnthropicModel(
                provider.model,
                provider=AnthropicProvider(api_key=api_key, base_url=provider.base_url),
            )
        # Schema's Literal narrows this to unreachable; assert for runtime
        # safety if someone bypasses the validator.
        raise ValueError(f"unknown provider api {provider.api!r}")

    async def chat(self, req: ChatRequest) -> ChatResponse:
        """Send a chat request and return the assistant's text reply."""
        messages = self._to_model_messages(req)
        settings = self._to_settings(req)
        resp = await model_request(self._model, messages, model_settings=settings)
        return self._to_chat_response(resp)

    @staticmethod
    def _to_model_messages(req: ChatRequest) -> list["ModelMessage"]:
        """Pack ``ChatMessage`` turns into pydantic-ai's request/response graph.

        Consecutive system+user turns collapse into a single ``ModelRequest``;
        each assistant turn becomes a ``ModelResponse`` with one ``TextPart``.
        Empty intermediate runs are dropped so the wire payload stays compact.
        """
        out: list["ModelMessage"] = []
        pending: list[ModelRequestPart] = []

        def flush() -> None:
            if pending:
                out.append(ModelRequest(parts=list(pending)))
                pending.clear()

        for m in req.messages:
            if m.role == "system":
                pending.append(SystemPromptPart(m.content))
            elif m.role == "user":
                pending.append(UserPromptPart(m.content))
            else:  # assistant
                flush()
                out.append(ModelResponse(parts=[TextPart(m.content)]))
        flush()
        return out

    @staticmethod
    def _to_settings(req: ChatRequest) -> ModelSettings:
        settings: ModelSettings = {"max_tokens": req.max_tokens}
        if req.temperature is not None:
            settings["temperature"] = req.temperature
        return settings

    def _to_chat_response(self, resp: ModelResponse) -> ChatResponse:
        text = "".join(p.content for p in resp.parts if isinstance(p, TextPart))
        usage = None
        if resp.usage and (resp.usage.input_tokens or resp.usage.output_tokens):
            usage = Usage(
                input_tokens=resp.usage.input_tokens or 0,
                output_tokens=resp.usage.output_tokens or 0,
            )
        return ChatResponse(
            text=text,
            model=resp.model_name or self.provider.model,
            provider_id=self.provider.id,
            stop_reason=resp.finish_reason,
            usage=usage,
        )
