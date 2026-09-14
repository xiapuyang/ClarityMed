"""Case + prediction + metrics schemas for the emergency eval harness.

A :class:`Case` carries the ground-truth label plus either pre-extracted
:class:`~claritymed.core.emergency.ExtractedSymptoms` (so the rule
engine can be evaluated without an LLM) or a list of conversation
``turns`` (so a real extractor can be exercised in the e2e run).

Why pydantic ``BaseModel`` rather than ``@dataclass``: cases are loaded
from YAML and need validation + ``extra="forbid"`` so a typo in a
source file fails loud instead of silently dropping the field.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from claritymed.core.emergency.schemas import EmergencyLevel, ExtractedSymptoms

Source = Literal[
    "public_vignettes",
    "ddxplus_subset",
    "synthetic",
    "meddialog_silver",
]


class Turn(BaseModel):
    """One conversation turn — text mode only."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    text: str = Field(min_length=1)


class Case(BaseModel):
    """One eval case.

    Exactly one of ``symptoms`` (structured mode — rule engine only)
    or ``turns`` (text mode — needs a real extractor) must be present.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    source: Source
    language: Literal["en", "zh"]
    citation: str = Field(default="", description="Public source pointer.")
    notes: str = Field(default="", description="Author notes / scoring rationale.")

    symptoms: ExtractedSymptoms | None = None
    turns: list[Turn] = Field(default_factory=list)

    ground_truth_level: EmergencyLevel
    ground_truth_rule_id: str | None = None
    # An adversarial case mimics a critical presentation but is **not**
    # critical (panic attack, costochondritis, tension HA). Drives the
    # adversarial-FPR metric.
    is_adversarial: bool = False

    @model_validator(mode="after")
    def _one_input_mode(self) -> "Case":
        has_symptoms = self.symptoms is not None
        has_turns = bool(self.turns)
        if has_symptoms == has_turns:
            raise ValueError(
                f"case {self.id!r}: must set exactly one of "
                "`symptoms` (structured mode) or `turns` (text mode)."
            )
        return self


class Prediction(BaseModel):
    """Result of running one case through the gate at one profile."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    profile: Literal["strict", "balanced", "lenient", "off"]
    predicted_level: EmergencyLevel
    matched_rule_ids: list[str] = Field(default_factory=list)
    # ``True`` when the case was skipped because it needed an extractor
    # we did not have. Skipped cases are excluded from metrics but
    # reported in the run summary so the operator sees the coverage gap.
    skipped: bool = False
    skip_reason: str = ""


class ConfusionCell(BaseModel):
    """One (actual, predicted) cell of the confusion matrix."""

    model_config = ConfigDict(extra="forbid")

    actual: EmergencyLevel
    predicted: EmergencyLevel
    count: int


class ProfileMetrics(BaseModel):
    """Aggregated metrics for one profile over the full case set."""

    model_config = ConfigDict(extra="forbid")

    profile: Literal["strict", "balanced", "lenient", "off"]
    total: int
    scored: int  # total minus skipped
    skipped: int
    # Critical = positive class for F-beta.
    critical_recall: float
    critical_precision: float
    f_beta_2: float
    # Of cases marked ``is_adversarial``, fraction predicted urgent+.
    adversarial_fpr: float
    # Of cases with a ``ground_truth_rule_id``, fraction where that
    # rule id appears in ``matched_rule_ids``.
    per_rule_recall: dict[str, float]
    # Alerts / 100 clinical turns. "Alert" = predicted urgent or higher.
    alert_rate_per_100: float
    confusion: list[ConfusionCell]
