"""Patient profile and longitudinal record contracts.

PHI lives here. Per architecture §2-3, ``Patient`` is *deterministically
injected* into the context, never retrieved through the vector store — the
``Allergy`` list is the canonical example: missing an anaphylactic substance
because the embedding ranker did not surface it would kill a patient. Storage
and retrieval policy lives in ``core/stores/profile.py``; this module is the
data contract only.

Solicitation policy: every ``Profile`` field carries a
``json_schema_extra={"solicitation": "proactive" | "passive"}`` tag.
*Proactive* fields (sex, age, weight, height, residence, birthplace)
materially shape clinical reasoning, so the ask-agent may ask for them
when they are missing. *Passive* fields (marital status, children,
occupation) are recorded only when the user volunteers — the agent must
not solicit them, to keep the questionnaire from feeling intrusive.
``PROACTIVE_PROFILE_FIELDS`` / ``PASSIVE_PROFILE_FIELDS`` are derived
from this tag so the policy stays single-sourced.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Sex = Literal["female", "male", "intersex", "unknown"]
AllergySeverity = Literal["mild", "moderate", "severe", "anaphylactic"]
AllergySource = Literal["self_report", "clinical_record"]
MaritalStatus = Literal["single", "partnered", "married", "divorced", "widowed"]
RecordKind = Literal["visit", "lab", "imaging", "note", "self_report"]
Solicitation = Literal["proactive", "passive"]


class Profile(BaseModel):
    """Biometric and biographical basics — one per user.

    We store ``birth_date`` rather than ``age`` because the latter drifts: a
    cached ``age`` is wrong the day after every birthday, while a ``birth_date``
    is stable for life. ``age`` is exposed as a derived property.

    Fields split into two solicitation tiers (see module docstring): *proactive*
    fields are clinically load-bearing and the agent may ask when they are
    missing; *passive* fields are only ever filled when the user mentions them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # --- Proactive: directly shape clinical reasoning ---------------------
    sex: Sex | None = Field(
        default=None, json_schema_extra={"solicitation": "proactive"}
    )
    weight_kg: float | None = Field(
        default=None, gt=0, le=500, json_schema_extra={"solicitation": "proactive"}
    )
    height_cm: float | None = Field(
        default=None, gt=0, le=300, json_schema_extra={"solicitation": "proactive"}
    )
    birth_date: date | None = Field(
        default=None, json_schema_extra={"solicitation": "proactive"}
    )
    residence: str | None = Field(
        default=None,
        max_length=128,
        json_schema_extra={"solicitation": "proactive"},
        description="Current residence (city / region), affects endemic exposure.",
    )
    birthplace: str | None = Field(
        default=None,
        max_length=128,
        json_schema_extra={"solicitation": "proactive"},
        description="Birthplace region, affects early-life exposure history.",
    )

    # --- Passive: only recorded if the user volunteers --------------------
    marital_status: MaritalStatus | None = Field(
        default=None, json_schema_extra={"solicitation": "passive"}
    )
    has_children: bool | None = Field(
        default=None, json_schema_extra={"solicitation": "passive"}
    )
    current_occupation: str | None = Field(
        default=None,
        max_length=128,
        json_schema_extra={"solicitation": "passive"},
    )
    past_occupations: str | None = Field(
        default=None,
        max_length=512,
        json_schema_extra={"solicitation": "passive"},
        description="Free text, comma-separated; relevant to occupational exposure.",
    )

    @field_validator("birth_date")
    @classmethod
    def _birth_date_plausible(cls, v: date | None) -> date | None:
        if v is None:
            return v
        today = date.today()
        if v > today:
            raise ValueError("birth_date cannot be in the future")
        if (today.year - v.year) > 130:
            raise ValueError("birth_date implies age > 130")
        return v

    @property
    def age(self) -> int | None:
        """Years since ``birth_date``. ``None`` if ``birth_date`` is unset."""
        if self.birth_date is None:
            return None
        today = date.today()
        years = today.year - self.birth_date.year
        if (today.month, today.day) < (self.birth_date.month, self.birth_date.day):
            years -= 1
        return years


def _fields_by_solicitation(level: Solicitation) -> frozenset[str]:
    """Derive the {proactive,passive} field partition from Profile's metadata."""
    return frozenset(
        name
        for name, info in Profile.model_fields.items()
        if (info.json_schema_extra or {}).get("solicitation") == level
    )


PROACTIVE_PROFILE_FIELDS: frozenset[str] = _fields_by_solicitation("proactive")
PASSIVE_PROFILE_FIELDS: frozenset[str] = _fields_by_solicitation("passive")


def solicitation_for(field: str) -> Solicitation:
    """Return the solicitation tier of a Profile field. Unknown → ValueError."""
    if field in PROACTIVE_PROFILE_FIELDS:
        return "proactive"
    if field in PASSIVE_PROFILE_FIELDS:
        return "passive"
    raise ValueError(f"unknown profile field: {field!r}")


def _validate_end_date(v: date | None, onset: date | None) -> date | None:
    """Shared validator for (onset_date, end_date) pairs.

    Rule: end_date may be null (ongoing), but if set it cannot be in the
    future, nor precede onset_date. Used by Condition, Allergy, Medication
    so the three durational PHI tables share one validation path.
    """
    if v is None:
        return v
    if v > date.today():
        raise ValueError("end_date cannot be in the future")
    if onset is not None and v < onset:
        raise ValueError("end_date cannot precede onset_date")
    return v


class Allergy(BaseModel):
    """One known allergy.

    ``onset_date`` is when the allergy was discovered / first reacted.
    ``end_date`` is when it was resolved (rare in practice — desensitization
    therapy or outgrown childhood allergies); null means still active.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    substance: str = Field(min_length=1)
    severity: AllergySeverity
    source: AllergySource
    onset_date: date | None = None
    end_date: date | None = None

    @field_validator("end_date")
    @classmethod
    def _end_after_onset(cls, v: date | None, info) -> date | None:
        return _validate_end_date(v, info.data.get("onset_date"))


class Condition(BaseModel):
    """One diagnosed / self-reported condition.

    ``onset_date`` is the start; ``end_date`` is the resolved date. A null
    ``end_date`` means the condition is still ongoing — that is the canonical
    encoding, not a separate ``is_active`` flag. Duration is derived from the
    two dates by the application layer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    display: str = Field(min_length=1)
    code: str | None = None
    onset_date: date | None = None
    end_date: date | None = None

    @field_validator("end_date")
    @classmethod
    def _end_after_onset(cls, v: date | None, info) -> date | None:
        return _validate_end_date(v, info.data.get("onset_date"))


class Medication(BaseModel):
    """One medication the patient is or was taking.

    ``onset_date`` is when the patient started; ``end_date`` is when it
    was discontinued. Null end_date is the canonical "currently taking"
    encoding — the query for "what is the patient on right now" is
    ``WHERE end_date IS NULL``, and the composite index covers it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    display: str = Field(min_length=1)
    code: str | None = None
    dose: str | None = None
    frequency: str | None = None
    onset_date: date | None = None
    end_date: date | None = None

    @field_validator("end_date")
    @classmethod
    def _end_after_onset(cls, v: date | None, info) -> date | None:
        return _validate_end_date(v, info.data.get("onset_date"))


class LongitudinalRecord(BaseModel):
    """One historical observation. Free text lives in ``summary``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    observed_at: datetime
    kind: RecordKind
    summary: str = Field(min_length=1)
    source: str = Field(min_length=1)


class Patient(BaseModel):
    """Patient profile — PHI. Bound to one ``user_id``.

    Composition: ``profile`` holds biometric basics (sex / weight / height /
    birth_date); allergies / conditions / medications are independently growing
    lists. There is no top-level ``age`` or ``sex`` — both derive from
    ``profile`` so the two can never disagree.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    user_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,32}$")
    profile: Profile = Field(default_factory=Profile)
    allergies: list[Allergy] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
