"""Runtime dependency container for the ask agent."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claritymed.core.emergency import EmergencyAssessment
    from claritymed.core.events import DifferentialReady
    from claritymed.core.interaction.prompt_channel import PromptChannel
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
    # Per-tool call counter. Each tool body increments its own slot on
    # entry; AskService reads this at turn end to spot
    # "announced-but-skipped" patterns (LLM says "I will retrieve…"
    # then ends the turn without actually firing the tool call) and
    # quantify how often each provider does it.
    tool_calls: dict[str, int] = field(default_factory=dict)
    # Name of the active RagMode (``deterministic`` / ``tool`` /
    # ``agentic``). Observational — modes own their own behaviour; this
    # rides on deps purely so the tool body and audit payload can name
    # which mode invoked them.
    mode: str = "tool"
    # Optional per-turn channel for the ``ask_user_question`` tool to
    # surface a UI prompt. ``None`` means the host is non-interactive
    # (eval, one-shot CLI, headless tests); the tool body returns a
    # plain-text fallback hint to the LLM in that case.
    prompt_channel: "PromptChannel | None" = None
    # Set by SymptomsFeature._predict when the tool returns a usable
    # differential. Picked up by the dynamic system_prompt registered in
    # make_ask_agent so the composing-guide is only present for the
    # second LLM call (reply composition), not the first (tool selection).
    # Stays None on user_declined / eligible:false / server_error turns.
    symptoms_reply_guide: str | None = None
    # Emergency triage gate result for this turn. Set by AskService
    # before agent.run from EmergencyTriage.assess(). Phase 3 wires the
    # dynamic system prompt that reads triage.level / suggested action /
    # missing_qualifiers and injects them as [SAFETY CONTEXT].
    triage: "EmergencyAssessment | None" = None
    # Effective sensitivity for this turn (after CLI / user-pref / app-
    # default resolution AND env-override downgrade). Drives the
    # disclaimer-suffix decision in AskService._finalize_turn.
    effective_sensitivity: str = "balanced"
    # Sidecar payload emitted by SymptomsFeature after a usable
    # differential. Set alongside the queue-emitted event so
    # AskService._finalize_turn can persist it on the assistant turn
    # (cards survive a page refresh instead of vanishing with the
    # in-memory stream state). ``None`` for every non-symptoms turn
    # and for the fall-through branches (user_declined / eligible=false
    # / server_error).
    differential_ready: "DifferentialReady | None" = None
