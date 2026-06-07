"""Unified async chat client over the ``openai`` and ``anthropic`` SDKs.

One ``LLMClient`` per ``ProviderConfig``; dispatch is by ``provider.api``.
PHI guard runs in the orchestrator *before* this layer — by the time bytes
reach an adapter, the payload is already either local-allowed or redacted.
"""

from claritymed.core.llm.client import LLMClient, MissingApiKeyError
from claritymed.core.llm.request import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Usage,
)

__all__ = [
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "LLMClient",
    "MissingApiKeyError",
    "Usage",
]
