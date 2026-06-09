"""Runtime dependency container for the ask agent."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claritymed.core.rag.strategies.base import RagStrategy
    from claritymed.core.schemas import ProviderConfig
    from claritymed.core.schemas.retrieval import RetrievedChunk
    from claritymed.core.translation import TranslationProvider


@dataclass
class AskDeps:
    """Runtime state threaded through the ask agent via ``RunContext.deps``."""

    strategy: "RagStrategy | None" = None
    user_id: str = ""
    user_whitelist: "list[str] | None" = None
    provider_config: "ProviderConfig | None" = None
    language: str = "en"
    translation_service: "TranslationProvider | None" = None
    event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    retrieved_chunks: "list[RetrievedChunk]" = field(default_factory=list)
    # Tagged True when the active strategy declares itself agentic via
    # ``is_agentic``. The retrieval pipeline is already tool-driven for
    # every mode, so this flag is observational rather than gating — it
    # lands in the audit payload so operators can confirm an agentic
    # rollout reached the tool loop instead of silently degrading.
    agentic: bool = False
