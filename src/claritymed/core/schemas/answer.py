"""Final orchestrator output contract (architecture §3 [4] → §6).

``GroundedAnswer`` is what the agent must return. Pydantic AI uses it as
``result_type``; the safety / composition layer reads ``red_flags`` and
``disclaimer`` to decide post-processing; audit logs serialize ``provenance``.

Error convention: callers must **not** let a ``ValidationError`` bubble up to
the CLI / API. The orchestrator wraps schema failures in a high-epistemic
``GroundedAnswer`` so the user always gets a structured response.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from claritymed.core.schemas.uncertainty import UncertaintyResult

Language = Literal["en", "zh"]
RedFlagSeverity = Literal["info", "warn", "emergency"]


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    language: Language
    url: str | None = None
    published_at: date | None = None
    quote: str | None = None


class RedFlag(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_id: str = Field(min_length=1)
    severity: RedFlagSeverity
    message: str = Field(min_length=1)
    language: Language
    # i18n key for the rule's suggested action (e.g.
    # ``emergency.action.call_ems_cardiac``). The pre-step EmergencyTriage
    # populates this so downstream surfaces can render the action verb
    # without re-deriving it from ``message``. Optional because legacy
    # callers (none today) that fill RedFlag from BASD-side severity
    # heuristics would not have an i18n key handy.
    suggested_action_i18n_key: str | None = None


class Disclaimer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    language: Language


class GroundedAnswer(BaseModel):
    """The orchestrator's single output. Pydantic AI ``result_type``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(pattern=r"^[0-9]{14}[0-9A-F]{8}$")
    user_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,32}$")
    language: Language
    text: str = Field(min_length=1)
    citations: list[Citation] = Field(default_factory=list)
    uncertainty: UncertaintyResult
    red_flags: list[RedFlag] = Field(default_factory=list)
    disclaimer: Disclaimer
    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _no_citations_means_uncertain(self) -> "GroundedAnswer":
        """No citations -> uncertainty must be at least 'medium'.

        An answer with zero evidence pretending to be high-confidence is the
        textbook RAG failure mode; we refuse it at the contract layer.
        """
        if not self.citations and self.uncertainty.level == "low":
            raise ValueError(
                "GroundedAnswer with no citations cannot have uncertainty.level='low'"
            )
        return self
