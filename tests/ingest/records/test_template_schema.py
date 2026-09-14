"""Tests for ``ingest.records.template_schema``.

Two responsibilities to pin:

1. The shape contract — required fields, frozen, extra="forbid".
2. The "no silent defaulting" decisions — ``kind`` is required, profile
   stays a dict (not a Profile model), user_id regex bites.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from claritymed.core.schemas.patient import Allergy, Condition, Medication
from claritymed.ingest.records.template_schema import (
    CaseAttachment,
    CaseEntry,
    FactsBundle,
    MetaConfig,
    UserBundle,
)


# --- MetaConfig --------------------------------------------------------


def test_meta_config_minimal_valid():
    meta = MetaConfig(
        created_at=datetime(2026, 6, 25, tzinfo=timezone.utc),
        user_ids=["alice"],
    )
    assert meta.schema_version == 1
    assert meta.user_ids == ["alice"]
    assert meta.source_hint is None


def test_meta_config_rejects_user_id_with_path_traversal():
    """`USER_ID_RE` is the gate that keeps a template from naming a
    sibling user directory like ``../etc``. Tested at the model layer
    so an invalid `_meta.user_ids` never escapes pydantic."""
    with pytest.raises(ValidationError, match="invalid user_id"):
        MetaConfig(
            created_at=datetime(2026, 6, 25, tzinfo=timezone.utc),
            user_ids=["../etc"],
        )


def test_meta_config_rejects_user_id_with_slash():
    with pytest.raises(ValidationError, match="invalid user_id"):
        MetaConfig(
            created_at=datetime(2026, 6, 25, tzinfo=timezone.utc),
            user_ids=["alice/bob"],
        )


def test_meta_config_rejects_duplicate_user_ids():
    with pytest.raises(ValidationError, match="duplicates"):
        MetaConfig(
            created_at=datetime(2026, 6, 25, tzinfo=timezone.utc),
            user_ids=["alice", "alice"],
        )


def test_meta_config_requires_at_least_one_user_id():
    with pytest.raises(ValidationError):
        MetaConfig(
            created_at=datetime(2026, 6, 25, tzinfo=timezone.utc),
            user_ids=[],
        )


def test_meta_config_forbids_extra_keys():
    with pytest.raises(ValidationError):
        MetaConfig.model_validate(
            {
                "created_at": "2026-06-25T00:00:00Z",
                "user_ids": ["alice"],
                "unexpected_key": "value",
            }
        )


# --- CaseEntry ---------------------------------------------------------


def _valid_case_kwargs(**overrides) -> dict:
    base = dict(
        case_id="notion-abcdef12",
        event_date=date(2024, 1, 15),
        title="Annual checkup",
        kind="exam-report",
    )
    base.update(overrides)
    return base


def test_case_entry_minimal_valid():
    case = CaseEntry(**_valid_case_kwargs())
    assert case.body_md == ""
    assert case.attachments == []
    assert case.tags == []
    assert case.category is None  # loader backfills from _meta.default_category


def test_case_entry_rejects_case_id_with_space():
    """`SLUG_RE` mismatch — caught early so the slug formula in
    case_writer never sees something it can't safely use."""
    with pytest.raises(ValidationError, match="case_id"):
        CaseEntry(**_valid_case_kwargs(case_id="has space"))


def test_case_entry_rejects_case_id_with_path_traversal():
    with pytest.raises(ValidationError, match="case_id"):
        CaseEntry(**_valid_case_kwargs(case_id="../escape"))


def test_case_entry_event_date_coerces_iso_string():
    """YAML round-trips ``date`` as ISO string; pydantic must coerce
    cleanly. Otherwise the loader has to special-case every date field."""
    case = CaseEntry(**_valid_case_kwargs(event_date="2024-01-15"))
    assert case.event_date == date(2024, 1, 15)


def test_case_entry_kind_is_required():
    """Key Decision §`Manifest.kind` is REQUIRED — no default from
    category. This test locks in the no-silent-defaulting decision."""
    kwargs = _valid_case_kwargs()
    kwargs.pop("kind")
    with pytest.raises(ValidationError, match="kind"):
        CaseEntry(**kwargs)


def test_case_entry_forbids_extra_keys():
    """Defense against a skill author who adds ``user_id`` or
    ``_ocr_warnings`` to the case body — the CLI's filename-authoritative
    discipline only holds if extras are rejected at the model layer."""
    with pytest.raises(ValidationError):
        CaseEntry(**_valid_case_kwargs(user_id="alice"))


# --- CaseAttachment ----------------------------------------------------


def test_case_attachment_minimal_valid():
    att = CaseAttachment(
        path="/tmp/draft/photo.jpg",
        original_filename="photo.jpg",
        mime="image/jpeg",
    )
    assert att.path == "/tmp/draft/photo.jpg"


def test_case_attachment_forbids_extra_keys():
    with pytest.raises(ValidationError):
        CaseAttachment(
            path="/tmp/x.pdf",
            original_filename="x.pdf",
            mime="application/pdf",
            sha256="abc",
        )


# --- FactsBundle -------------------------------------------------------


def test_facts_bundle_profile_is_plain_dict_not_hydrated():
    """The fact writer needs `profile` as a *partial* dict so it can
    distinguish "user explicitly set X" from "user said nothing about X".
    Hydrating to ``Profile`` here would backfill every unset field with
    ``None`` and break the partial-merge contract (R12 / Unit 7)."""
    facts = FactsBundle(profile={"sex": "female"})
    assert facts.profile == {"sex": "female"}
    assert "weight_kg" not in facts.profile
    assert "birth_date" not in facts.profile


def test_facts_bundle_allergies_validate_through_patient_model():
    facts = FactsBundle(
        allergies=[
            Allergy(substance="penicillin", severity="severe", source="clinical_record")
        ]
    )
    assert len(facts.allergies) == 1
    assert facts.allergies[0].substance == "penicillin"


def test_facts_bundle_conditions_and_medications_default_empty():
    facts = FactsBundle()
    assert facts.allergies == []
    assert facts.conditions == []
    assert facts.medications == []
    assert facts.profile is None


def test_facts_bundle_medication_rejects_invalid_dates():
    """Inherited from ``Medication`` validators — pinned to confirm
    re-use rather than re-derivation."""
    with pytest.raises(ValidationError):
        FactsBundle(
            medications=[
                Medication(
                    display="metformin",
                    onset_date="2024-02-01",
                    end_date="2024-01-01",  # before onset → reject
                )
            ]
        )


def test_facts_bundle_forbids_extra_keys():
    with pytest.raises(ValidationError):
        FactsBundle.model_validate({"profile": None, "labs": []})


def test_facts_bundle_condition_validates():
    facts = FactsBundle(
        conditions=[Condition(display="Type 2 diabetes", onset_date=date(2020, 5, 1))]
    )
    assert facts.conditions[0].display == "Type 2 diabetes"


# --- UserBundle --------------------------------------------------------


def test_user_bundle_defaults_to_empty_cases_and_facts():
    bundle = UserBundle()
    assert bundle.cases == []
    assert bundle.facts.profile is None


def test_user_bundle_forbids_extra_keys():
    """The defense behind 'user_id is filename-authoritative': a
    maintainer who later adds ``user_id:`` inside a per-user YAML for
    convenience gets a load-time rejection."""
    with pytest.raises(ValidationError):
        UserBundle.model_validate({"user_id": "alice", "cases": [], "facts": {}})


def test_user_bundle_round_trips_minimal_template():
    bundle = UserBundle.model_validate(
        {
            "cases": [
                {
                    "case_id": "notion-abcdef12",
                    "event_date": "2024-01-15",
                    "title": "Annual checkup",
                    "kind": "exam-report",
                    "body_md": "BP 120/80; HR 65.",
                }
            ],
            "facts": {
                "profile": {"sex": "female"},
                "allergies": [
                    {
                        "substance": "penicillin",
                        "severity": "severe",
                        "source": "clinical_record",
                    }
                ],
            },
        }
    )
    assert bundle.cases[0].event_date == date(2024, 1, 15)
    assert bundle.facts.profile == {"sex": "female"}
    assert bundle.facts.allergies[0].substance == "penicillin"
