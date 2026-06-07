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

from pydantic import BaseModel, ConfigDict, Field

Sex = Literal["female", "male", "intersex", "unknown"]
AllergySeverity = Literal["mild", "moderate", "severe", "anaphylactic"]
AllergySource = Literal["self_report", "clinical_record"]
RecordKind = Literal["visit", "lab", "imaging", "note", "self_report"]


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
    """Patient profile — PHI. Bound to one ``user_id``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    user_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,32}$")
    age: int = Field(ge=0, le=130)
    sex: Sex
    allergies: list[Allergy] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
