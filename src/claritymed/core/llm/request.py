"""Wire-format-agnostic chat contract.

These types are what the orchestrator hands to ``LLMClient.chat()`` and what
it gets back. The provider-specific message shape (OpenAI's ``role/content``
dicts, Anthropic's content blocks) is the adapter's problem, not ours.

Why a separate ``ChatMessage`` instead of reusing ``pydantic_ai.messages``:
their ``ModelMessage`` is a graph type optimized for agent loops with tool
calls and instructions. The orchestrator just wants "list of turns, last
turn is the new user prompt." Keeping that surface small now means callers
do not couple to pydantic-ai's wire types — if we ever switch frameworks,
this contract does not move.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant"]

# Anthropic's Messages API requires max_tokens. OpenAI accepts it as a cap.
# 1024 is enough for a typical answer block; the caller overrides for longer.
DEFAULT_MAX_TOKENS = 1024


class ChatMessage(BaseModel):
    """One turn in a chat."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Role
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    """A full chat request: messages + sampling knobs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, gt=0)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class Usage(BaseModel):
    """Token accounting echoed back by the provider."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class ChatResponse(BaseModel):
    """The adapter's normalized response."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    text: str
    model: str
    provider_id: str
    stop_reason: str | None = None
    usage: Usage | None = None
