"""Tests for ``Patient`` and its nested PHI types."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from claritymed.core.schemas import Allergy, Condition, Medication, Patient, Profile


def _patient(**overrides) -> Patient:
    payload = {
        "user_id": "alice",
        "profile": Profile(sex="female", birth_date=date(1984, 1, 1)),
        "allergies": [
            Allergy(substance="penicillin", severity="severe", source="self_report"),
        ],
        "conditions": [Condition(display="Type 2 diabetes")],
        "medications": [
            Medication(display="metformin", dose="500 mg", frequency="bid"),
        ],
    }
    payload.update(overrides)
    return Patient(**payload)


def test_happy_path_construction():
    p = _patient()
    assert p.allergies[0].substance == "penicillin"
    assert p.profile.sex == "female"


def test_empty_profile_legal():
    """A brand-new patient with no biometric data must construct."""
    p = Patient(user_id="alice")
    assert p.profile == Profile()
    assert p.profile.age is None


def test_empty_allergies_legal():
    p = _patient(allergies=[])
    assert p.allergies == []


def test_invalid_user_id_rejected():
    with pytest.raises(ValidationError):
        _patient(user_id="../etc")


def test_extra_fields_blocked():
    """The PHI smuggling defence: a hallucinated ``ssn`` field must fail."""
    with pytest.raises(ValidationError):
        Patient(
            user_id="alice",
            profile=Profile(sex="female"),
            ssn="000-00-0000",  # type: ignore[call-arg]
        )


def test_round_trip_json():
    p = _patient()
    raw = p.model_dump_json()
    restored = Patient.model_validate_json(raw)
    assert restored == p


def test_age_derived_from_profile():
    """No top-level ``age`` — it lives on ``profile`` and is computed."""
    today = date.today()
    p = _patient(
        profile=Profile(
            sex="female", birth_date=date(today.year - 25, today.month, today.day)
        )
    )
    assert p.profile.age == 25
