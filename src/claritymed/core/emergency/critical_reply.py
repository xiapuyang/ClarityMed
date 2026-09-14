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

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from claritymed.core.emergency.composer import _ComposerInput
from claritymed.core.emergency.schemas import ExtractedSymptoms, MatchedRule
from claritymed.core.observability.audit import audit_event
from claritymed.core.prompts.registry import PromptRegistry

if TYPE_CHECKING:
    from pydantic_ai.models import Model

logger = logging.getLogger(__name__)

_CRITICAL_REPLY_PROMPT_NAME = "emergency_reply"
# Per-stage asyncio timeout budget. On timeout the critical_reply falls open
# to an action-text-only reply built from i18n alone (skip supporting
# sentences). The gate's safety instruction is still delivered; only the
# additional context paragraph is lost.
_CRITICAL_REPLY_TIMEOUT_S = 5.0


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

    Mirrors :class:`LLMComposer` shape but returns :class:`CriticalReplyResult`
    so the caller can persist messages_json without re-running the LLM.

    Agent instances are cached per language so repeated critical calls
    for the same language skip agent construction. pydantic-ai Agent
    instances are stateless w.r.t. user messages, so caching is safe.
    """

    def __init__(
        self,
        model: "Model",
        registry: PromptRegistry | None = None,
    ) -> None:
        self._model = model
        self._registry = registry or PromptRegistry()
        # Dict keyed by language → cached Agent for that language.
        self._agents: dict[str, Any] = {}

    def _get_agent(self, language: str) -> Any:
        """Return a cached (or freshly built) Agent for ``language``."""
        from pydantic_ai import Agent

        if language not in self._agents:
            system_prompt = self._registry.get(
                _CRITICAL_REPLY_PROMPT_NAME,
                language=language,  # type: ignore[arg-type]
            )
            self._agents[language] = Agent(
                self._model,
                output_type=str,
                system_prompt=system_prompt,
            )
        return self._agents[language]

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

        On asyncio.TimeoutError falls open to a CriticalReplyResult
        with empty text — the caller still sends the action text alone,
        which is a complete actionable reply (KTD-E3 contract).
        """
        agent = self._get_agent(language)
        user_text = _ComposerInput(matched_rules, symptoms).to_text()
        try:
            result = await asyncio.wait_for(
                agent.run(user_text), timeout=_CRITICAL_REPLY_TIMEOUT_S
            )
            return CriticalReplyResult(
                text=(result.output or "").strip(),
                messages_json=result.all_messages_json(),
                usage=result.usage,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "emergency.critical_reply timed out after %ss, failing open",
                _CRITICAL_REPLY_TIMEOUT_S,
            )
            try:
                audit_event(
                    "redflag.gate_component_timeout",
                    payload={
                        "component": "critical_reply",
                        "timeout_s": _CRITICAL_REPLY_TIMEOUT_S,
                    },
                )
            except Exception:  # noqa: BLE001
                pass
            return CriticalReplyResult(text="", messages_json=b"", usage=None)


def build_default_critical_reply(
    *,
    registry: PromptRegistry | None = None,
) -> CriticalReplyComposer | None:
    """Construct a :class:`CriticalReplyComposer` against the configured local provider.

    Reads ``configs/emergency.yaml::provider_id`` to pin selection;
    falls back to "first kind=local in models.yaml" when unset.
    Returns ``None`` when no usable local provider is found —
    AskService then sends the localized action text alone, which is
    still a complete actionable reply.
    """
    from claritymed.core.emergency._provider import build_local_gate_model
    from claritymed.core.emergency.config import load_emergency_config

    cfg = load_emergency_config()
    model = build_local_gate_model("critical_reply", prefer_id=cfg.provider_id)
    if model is None:
        return None
    return CriticalReplyComposer(model, registry=registry)
