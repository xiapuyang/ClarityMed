"""Schemas for the emergency triage gate.

* :class:`ExtractedSymptoms` — output of the (local) extractor LLM.
  Phase 1 ships the type; Phase 3 wires the extractor agent.
* :class:`MatchedRule` — one rule that fired against extracted
  symptoms. Carries the rule id + level + i18n action key so the
  composer LLM and downstream consumers do not need to re-derive
  wording from free text.
* :class:`EmergencyAssessment` — facade output. ``level`` is the
  worst level across ``matched_rules``; ``routine_noop()`` builds a
  zero-finding default for non-clinical turns and for the ``off``
  sensitivity short-circuit.

Why pydantic ``BaseModel`` rather than ``@dataclass``: the same shape
serializes into audit payloads (``redflag_trigger``) and feeds the
pydantic-ai ``output_validator``. One model, one source of truth.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EmergencyLevel = Literal["critical", "urgent", "moderate", "routine"]
SensitivityName = Literal["strict", "balanced", "lenient", "off"]
Onset = Literal["sudden", "gradual", "unknown"]


class ExtractedSymptoms(BaseModel):
    """Structured shape the extractor LLM emits.

    ``primary_complaint=None`` is the early-exit signal: non-clinical
    input (greetings, meta questions, unrelated chatter). The rule
    engine is never invoked when this is None.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    primary_complaint: str | None = None
    qualifiers: list[str] = Field(default_factory=list)
    onset: Onset = "unknown"
    duration_hours: float | None = None
    severity_self_report: int | None = Field(default=None, ge=0, le=10)
    age: int | None = Field(default=None, ge=0, le=130)
    sex: Literal["F", "M"] | None = None
    key_history: list[str] = Field(default_factory=list)
    associated: list[str] = Field(default_factory=list)


class MatchedRule(BaseModel):
    """One rule that fired against the extracted symptoms."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_id: str = Field(min_length=1)
    level: EmergencyLevel
    suggested_action_i18n_key: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)
    matched_qualifiers: list[str] = Field(default_factory=list)


class EmergencyAssessment(BaseModel):
    """Pre-step gate output, consumed by ``AskService``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    level: EmergencyLevel = "routine"
    matched_rules: list[MatchedRule] = Field(default_factory=list)
    # Highest-severity rule's action key. ``None`` when level == routine.
    suggested_action_i18n_key: str | None = None
    suspected_high_risk_categories: list[str] = Field(default_factory=list)
    # Hints for the main agent to elicit on the next turn. Drives the
    # sparse-input recovery path (plan §"Sparse-input handling").
    missing_qualifiers: list[str] = Field(default_factory=list)
    # Composer LLM output. Empty on routine_noop / off-mode.
    reasoning: str = ""
    citations: list[str] = Field(default_factory=list)

    @classmethod
    def routine_noop(cls) -> "EmergencyAssessment":
        """Default no-finding result.

        Returned for non-clinical input, for ``sensitivity == 'off'``
        short-circuit, and as the Phase 1 placeholder until the
        extractor + rule engine are wired in Phases 2-3.
        """
        return cls()
