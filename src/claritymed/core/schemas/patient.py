"""Patient profile and longitudinal record contracts.

PHI lives here. Per architecture §2-3, ``Patient`` is *deterministically
injected* into the context, never retrieved through the vector store — the
``Allergy`` list is the canonical example: missing an anaphylactic substance
because the embedding ranker did not surface it would kill a patient. Storage
and retrieval policy lives in ``core/stores/profile.py``; this module is the
data contract only.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Sex = Literal["female", "male", "intersex", "unknown"]
AllergySeverity = Literal["mild", "moderate", "severe", "anaphylactic"]
AllergySource = Literal["self_report", "clinical_record"]
RecordKind = Literal["visit", "lab", "imaging", "note", "self_report"]


class Profile(BaseModel):
    """Biometric basics — one per user.

    We store ``birth_date`` rather than ``age`` because the latter drifts: a
    cached ``age`` is wrong the day after every birthday, while a ``birth_date``
    is stable for life. ``age`` is exposed as a derived property.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sex: Sex | None = None
    weight_kg: float | None = Field(default=None, gt=0, le=500)
    height_cm: float | None = Field(default=None, gt=0, le=300)
    birth_date: date | None = None

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


class Allergy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    substance: str = Field(min_length=1)
    severity: AllergySeverity
    source: AllergySource


class Condition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    display: str = Field(min_length=1)
    code: str | None = None
    onset_date: date | None = None


class Medication(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    display: str = Field(min_length=1)
    code: str | None = None
    dose: str | None = None
    frequency: str | None = None


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
