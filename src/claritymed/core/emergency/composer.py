"""Composer LLM — turn matched rules into user-facing reasoning.

The composer is the *prose* half of the gate. The rule engine has
already decided the level + action; the composer's job is to explain
the finding in plain language the user can act on. The deterministic
action wording (call EMS, urgent eval, …) lives in i18n and is
inserted at render time — the composer never paraphrases it.

Defaults to a local provider (omlx) per the plan's "composer LLM
choice" decision: bilingual quality is acceptable from local models
and PHI never leaves the device. Operators can override by injecting
their own composer instance via :class:`EmergencyTriage`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from claritymed.core.emergency.schemas import (
    EmergencyAssessment,
    ExtractedSymptoms,
    MatchedRule,
)
from claritymed.core.prompts.registry import PromptRegistry

if TYPE_CHECKING:
    from pydantic_ai.models import Model

logger = logging.getLogger(__name__)

_COMPOSER_PROMPT_NAME = "emergency_composer"


class Composer(Protocol):
    """Composer interface — implementations may be LLM or stubs.

    The Protocol shape lets tests inject a deterministic fake while
    production wires the LLM-backed implementation.
    """

    async def compose(
        self,
        matched_rules: list[MatchedRule],
        symptoms: ExtractedSymptoms,
        *,
        language: str,
    ) -> str:
        """Return a short paragraph explaining the finding to the user."""
        ...


@dataclass
class _ComposerInput:
    """User-message payload sent to the composer LLM."""

    matched_rules: list[MatchedRule]
    symptoms: ExtractedSymptoms

    def to_text(self) -> str:
        """Render as plain text — keeps the prompt LLM-friendly without
        forcing structured outputs on small local models that handle
        free-form JSON poorly (see ``feedback_tool_prompt_few_shot``).
        """
        lines: list[str] = ["matched_rules:"]
        for r in self.matched_rules:
            lines.append(
                f"  - id: {r.rule_id}  level: {r.level}  "
                f"action_key: {r.suggested_action_i18n_key}  "
                f"matched_qualifiers: {r.matched_qualifiers}"
            )
        lines.append("symptoms:")
        lines.append(f"  primary_complaint: {self.symptoms.primary_complaint}")
        lines.append(f"  qualifiers: {self.symptoms.qualifiers}")
        if self.symptoms.age is not None:
            lines.append(f"  age: {self.symptoms.age}")
        if self.symptoms.sex is not None:
            lines.append(f"  sex: {self.symptoms.sex}")
        if self.symptoms.key_history:
            lines.append(f"  key_history: {self.symptoms.key_history}")
        return "\n".join(lines)


class LLMComposer:
    """pydantic-ai-backed composer reading ``emergency_composer.yaml``.

    Construction is cheap (registry + model handle); the LLM call only
    fires inside :meth:`compose`. Operators who want a different model
    pass it via ``model``; tests can either inject this class with a
    stub model or implement :class:`Composer` directly.
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
    ) -> str:
        from pydantic_ai import Agent

        system_prompt = self._registry.get(
            _COMPOSER_PROMPT_NAME,
            language=language,  # type: ignore[arg-type]
        )
        agent: Agent[None, str] = Agent(
            self._model,
            output_type=str,
            system_prompt=system_prompt,
        )
        user_text = _ComposerInput(matched_rules, symptoms).to_text()
        result = await agent.run(user_text)
        return (result.output or "").strip()


def build_default_composer(
    *,
    registry: PromptRegistry | None = None,
) -> LLMComposer | None:
    """Construct an :class:`LLMComposer` against the configured local provider.

    Reads ``configs/emergency.yaml::provider_id`` to pin selection;
    falls back to "first kind=local in models.yaml" when unset.
    Returns ``None`` when no usable local provider is found —
    :class:`EmergencyTriage` then runs the rule engine without prose;
    matched rules still drive ``red_flags[]`` and the i18n action
    wording, so the gate stays functional even without a composer.
    See the symmetric
    :func:`claritymed.core.emergency.extractor.build_default_extractor`.
    """
    from claritymed.core.emergency._provider import build_local_gate_model
    from claritymed.core.emergency.config import load_emergency_config

    cfg = load_emergency_config()
    model = build_local_gate_model("composer", prefer_id=cfg.provider_id)
    if model is None:
        return None
    return LLMComposer(model, registry=registry)


def build_assessment(
    matched_rules: list[MatchedRule],
    symptoms: ExtractedSymptoms,
    reasoning: str,
    *,
    missing_qualifiers: list[str] | None = None,
) -> EmergencyAssessment:
    """Assemble the final :class:`EmergencyAssessment` from parts.

    Lives in this module (not ``schemas.py``) so the assembly logic
    that depends on rule semantics — pick the worst-level rule's
    action key, union of citations — stays close to the composer
    that produced ``reasoning``.

    ``matched_rules`` is assumed pre-sorted by severity (the rule
    engine returns it sorted); index 0 is therefore the authoritative
    level and action key.
    """
    if not matched_rules:
        return EmergencyAssessment.routine_noop()
    top = matched_rules[0]
    citations: list[str] = []
    seen: set[str] = set()
    for r in matched_rules:
        for c in r.citations:
            if c not in seen:
                seen.add(c)
                citations.append(c)
    return EmergencyAssessment(
        level=top.level,
        matched_rules=list(matched_rules),
        suggested_action_i18n_key=top.suggested_action_i18n_key,
        missing_qualifiers=list(missing_qualifiers or []),
        reasoning=reasoning,
        citations=citations,
        suspected_high_risk_categories=[],
    )
