"""``TurnState`` Protocol: the read-only contract core code expects from a
per-turn dep container.

Core layers (``features``, ``rag.retrieval_pipeline``, ``rag.tools``) need
to read a small set of fields off the dep container that flows through
the LLM turn — but they do **not** care which concrete class supplies
those fields. The Protocol declares the contract here; the concrete
``AskDeps`` dataclass lives in ``orchestrator/agents/ask_deps.py`` and
satisfies it structurally.

This is the dependency-inversion knob that lets the orchestrator add
fields (cancellation events, chat-session refs, future web-layer
request metadata) to ``AskDeps`` without touching anything in core. As
long as the new field isn't something core reads, the Protocol stays
unchanged.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from claritymed.core.rag.strategies.base import RagStrategy
    from claritymed.core.schemas import ProviderConfig
    from claritymed.core.schemas.retrieval import RetrievedChunk
    from claritymed.core.translation import TranslationProvider


class TurnState(Protocol):
    """Per-turn state read by core-layer code.

    Implementations (currently just ``orchestrator.agents.ask_deps.AskDeps``)
    may carry additional orchestrator-specific fields that are invisible
    to this contract. Core code must not depend on anything outside this
    Protocol; if it needs a new field, add it here first, then add it on
    the concrete impl(s).
    """

    strategy: "RagStrategy | None"
    user_id: str
    user_whitelist: "list[str] | None"
    provider_config: "ProviderConfig | None"
    language: str
    translation_service: "TranslationProvider | None"
    event_queue: asyncio.Queue
    retrieved_chunks: "list[RetrievedChunk]"
    # Per-tool invocation counter. Tool bodies increment their slot on
    # entry so the service layer can detect "announced-but-skipped"
    # patterns at turn end.
    tool_calls: dict[str, int]
