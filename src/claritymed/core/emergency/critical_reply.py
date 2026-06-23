"""Critical-level short-circuit reply composer.

Used by :class:`~claritymed.orchestrator.services.ask_service.AskService`
when ``triage.level == "critical"`` (KTD-E3). The agent loop is bypassed
entirely; this module produces the single user-facing reply directly.

Contract:

* The localized action sentence (``emergency.action.*``) is the load-
  bearing safety instruction. The caller prepends it verbatim — this
  module's LLM only adds the supporting 2–4 sentences.
* If the LLM fails or no model is wired, ``compose`` returns the empty
  string — the caller still sends the action text alone, which is a
  complete actionable reply. The gate's safety net does not depend on
  the composer being available.

Why separate from :class:`~claritymed.core.emergency.composer.LLMComposer`:
the composer writes a short *reasoning* paragraph that the agent then
embeds in a longer answer. This module's output is the *entire* reply
after the action sentence — no agent runs after it. The prompt
contract is therefore different (no questions, no "I am an AI"
boilerplate, hard 80-word cap), so it has its own YAML.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from claritymed.core.emergency.composer import _ComposerInput
from claritymed.core.emergency.schemas import ExtractedSymptoms, MatchedRule
from claritymed.core.prompts.registry import PromptRegistry

if TYPE_CHECKING:
    from pydantic_ai.models import Model

logger = logging.getLogger(__name__)

_CRITICAL_REPLY_PROMPT_NAME = "emergency_reply"


@dataclass(frozen=True)
class CriticalReplyResult:
    """Output bundle for the critical-reply composer.

    Carries both the supporting text the composer produced AND the
    ``messages_json`` bytes pydantic-ai emits — the caller persists the
    latter into the chat session so the critical exchange shows up on
    the next turn's history just like any other agent turn.
    """

    text: str
    messages_json: bytes
    usage: object | None  # pydantic_ai.usage.RunUsage; opaque to keep import light


class CriticalReplyComposer:
    """pydantic-ai-backed composer reading ``emergency_reply.yaml``.

    Mirrors :class:`LLMComposer` shape (cheap construction, Agent built
    per-call). Returns :class:`CriticalReplyResult` so the caller can
    persist messages_json without re-running the LLM.
    """

    def __init__(
        self,
        model: "Model",
        registry: PromptRegistry | None = None,
    ) -> None:
        self._model = model
        self._registry = registry or PromptRegistry()

    async def compose(
        self,
        matched_rules: list[MatchedRule],
        symptoms: ExtractedSymptoms,
        *,
        language: str,
    ) -> CriticalReplyResult:
        """Compose the 2–4 supporting sentences for a critical reply.

        The localized action sentence is owned by the caller — this
        method emits ONLY the supporting text. The prompt rules are
        documented in ``emergency_reply.yaml``.
        """
        from pydantic_ai import Agent

        system_prompt = self._registry.get(
            _CRITICAL_REPLY_PROMPT_NAME,
            language=language,  # type: ignore[arg-type]
        )
        agent: Agent[None, str] = Agent(
            self._model,
            output_type=str,
            system_prompt=system_prompt,
        )
        user_text = _ComposerInput(matched_rules, symptoms).to_text()
        result = await agent.run(user_text)
        return CriticalReplyResult(
            text=(result.output or "").strip(),
            messages_json=result.all_messages_json(),
            usage=result.usage,
        )


def build_default_critical_reply(
    *,
    registry: PromptRegistry | None = None,
) -> CriticalReplyComposer | None:
    """Construct a :class:`CriticalReplyComposer` against the first local provider.

    Returns ``None`` when no local provider is configured / available —
    AskService then sends the localized action text alone, which is
    still a complete actionable reply.
    """
    from claritymed.core.emergency._provider import build_local_gate_model

    model = build_local_gate_model("critical_reply")
    if model is None:
        return None
    return CriticalReplyComposer(model, registry=registry)
