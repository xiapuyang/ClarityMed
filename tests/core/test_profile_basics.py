"""Tests for the per-user ``profile`` table (biometric basics).

Covers the new singleton row: get/upsert round-trip, cross-user isolation, and
the schema-level rejections (future birth_date, negative weight, invalid sex).
Also covers the solicitation tier partition derived from ``Profile.model_fields``.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from claritymed.core.schemas import (
    PASSIVE_PROFILE_FIELDS,
    PROACTIVE_PROFILE_FIELDS,
    Profile,
    solicitation_for,
)
from claritymed.errors import UserIdMismatch
from claritymed.stores.profile import ProfileStore


def test_get_profile_returns_none_when_unset(trio):
    alice_uid = trio["admin"].user_id
    assert ProfileStore(alice_uid).get_profile() is None


def test_upsert_then_get_round_trip(trio):
    alice_uid = trio["admin"].user_id
    profile = Profile(
        sex="female",
        weight_kg=62.5,
        height_cm=168.0,
        birth_date=date(1990, 4, 12),
    )
    ProfileStore(alice_uid).upsert_profile(profile, owner_user_id=alice_uid)

    fetched = ProfileStore(alice_uid).get_profile()
    assert fetched == profile


def test_upsert_updates_existing_row_not_insert(trio):
    """Singleton invariant: a second upsert overwrites, never duplicates."""
    alice_uid = trio["admin"].user_id
    store = ProfileStore(alice_uid)

    store.upsert_profile(Profile(sex="female", weight_kg=62.5), owner_user_id=alice_uid)
    store.upsert_profile(
        Profile(sex="female", weight_kg=63.1, height_cm=168.0),
        owner_user_id=alice_uid,
    )

    from sqlmodel import Session, select
    from claritymed.stores.profile import ProfileRow

    with Session(store.engine) as session:
        rows = session.exec(
            select(ProfileRow).where(ProfileRow.user_id == alice_uid)
        ).all()
    assert len(rows) == 1
    assert rows[0].weight_kg == 63.1
    assert rows[0].height_cm == 168.0


def test_bob_cannot_see_alice_profile(trio):
    """Isolation: the can't-ever-fail invariant carried over to the profile row."""
    alice_uid = trio["admin"].user_id
    bob_uid = trio["users"][0].user_id
    ProfileStore(alice_uid).upsert_profile(
        Profile(sex="female", weight_kg=62.5), owner_user_id=alice_uid
    )
    assert ProfileStore(bob_uid).get_profile() is None


def test_upsert_owner_mismatch_rejected(trio):
    alice_uid = trio["admin"].user_id
    bob_uid = trio["users"][0].user_id
    with pytest.raises(UserIdMismatch):
        ProfileStore(alice_uid).upsert_profile(
            Profile(sex="female"), owner_user_id=bob_uid
        )


def test_invalid_sex_rejected():
    with pytest.raises(ValidationError):
        Profile(sex="other")  # type: ignore[arg-type]


def test_negative_weight_rejected():
    with pytest.raises(ValidationError):
        Profile(weight_kg=-1)


def test_implausible_height_rejected():
    with pytest.raises(ValidationError):
        Profile(height_cm=400)


def test_future_birth_date_rejected():
    with pytest.raises(ValidationError):
        Profile(birth_date=date(2999, 1, 1))


def test_age_derived_from_birth_date():
    today = date.today()
    p = Profile(birth_date=date(today.year - 30, today.month, today.day))
    assert p.age == 30


def test_age_is_none_when_birth_date_unset():
    assert Profile().age is None


def test_extra_fields_blocked():
    """PHI smuggling defence carries over to the new schema."""
    with pytest.raises(ValidationError):
        Profile(blood_type="O+")  # type: ignore[call-arg]


def test_new_geo_and_biographical_fields_round_trip(trio):
    """Residence, birthplace, marital_status, has_children, occupations."""
    alice_uid = trio["admin"].user_id
    profile = Profile(
        sex="female",
        residence="Shanghai, China",
        birthplace="Kunming, Yunnan",
        marital_status="married",
        has_children=True,
        current_occupation="nurse",
        past_occupations="teacher, farm worker",
    )
    ProfileStore(alice_uid).upsert_profile(profile, owner_user_id=alice_uid)
    fetched = ProfileStore(alice_uid).get_profile()
    assert fetched == profile


def test_marital_status_enum_rejects_unknown_value():
    with pytest.raises(ValidationError):
        Profile(marital_status="complicated")  # type: ignore[arg-type]


def test_residence_length_capped():
    with pytest.raises(ValidationError):
        Profile(residence="x" * 200)


def test_solicitation_partition_covers_every_profile_field():
    """The two derived sets must partition Profile.model_fields exactly.

    If we add a Profile field without tagging it, the partition gap fails
    fast here instead of silently making the prompt-side policy stale.
    """
    tagged = PROACTIVE_PROFILE_FIELDS | PASSIVE_PROFILE_FIELDS
    assert tagged == set(Profile.model_fields)
    assert PROACTIVE_PROFILE_FIELDS.isdisjoint(PASSIVE_PROFILE_FIELDS)


def test_solicitation_tiers_expected_membership():
    """Concrete partition the prompt-policy depends on. If this drifts the
    ask-agent prompt is wrong, so make it loud."""
    assert PROACTIVE_PROFILE_FIELDS == frozenset(
        {"sex", "weight_kg", "height_cm", "birth_date", "residence", "birthplace"}
    )
    assert PASSIVE_PROFILE_FIELDS == frozenset(
        {
            "marital_status",
            "has_children",
            "current_occupation",
            "past_occupations",
        }
    )


def test_solicitation_for_unknown_field_raises():
    with pytest.raises(ValueError, match="unknown profile field"):
        solicitation_for("blood_type")


def test_durational_phi_tables_have_composite_user_end_date_index(trio):
    """Each of allergy/condition/medication must carry the composite index
    that backs both the "currently active" (end_date IS NULL) lookup and the
    "most recent N" range scan. If you add a new durational PHI table,
    extend the expected_indexes set instead of skipping this test."""
    import sqlite3

    from claritymed.stores.paths import user_db_path
    from claritymed.stores.profile import ProfileStore

    alice_uid = trio["admin"].user_id
    ProfileStore(alice_uid)  # ensure the engine + tables exist

    con = sqlite3.connect(user_db_path(alice_uid))
    try:
        present = {
            row[0]
            for row in con.execute("SELECT name FROM sqlite_master WHERE type='index'")
        }
    finally:
        con.close()

    expected = {
        "ix_allergy_user_end_date",
        "ix_condition_user_end_date",
        "ix_medication_user_end_date",
    }
    missing = expected - present
    assert not missing, f"missing indexes: {missing}"


def test_list_conditions_returns_active_first_then_recent_resolved(trio):
    """Sort contract: end_date IS NULL first, then end_date DESC."""
    from claritymed.core.schemas import Condition
    from claritymed.stores.profile import ProfileStore

    alice_uid = trio["admin"].user_id
    s = ProfileStore(alice_uid)
    s.add_condition(
        Condition(display="resolved_old", end_date=date(2018, 1, 1)),
        owner_user_id=alice_uid,
    )
    s.add_condition(
        Condition(display="resolved_recent", end_date=date(2024, 1, 1)),
        owner_user_id=alice_uid,
    )
    s.add_condition(Condition(display="active"), owner_user_id=alice_uid)

    names = [c.display for c in s.list_conditions()]
    assert names[0] == "active"
    assert names[1] == "resolved_recent"
    assert names[2] == "resolved_old"


def test_list_medications_returns_taking_first_then_recent_discontinued(trio):
    from claritymed.core.schemas import Medication
    from claritymed.stores.profile import ProfileStore

    alice_uid = trio["admin"].user_id
    s = ProfileStore(alice_uid)
    s.add_medication(
        Medication(display="old_drug", end_date=date(2020, 1, 1)),
        owner_user_id=alice_uid,
    )
    s.add_medication(
        Medication(display="recent_stopped", end_date=date(2024, 6, 1)),
        owner_user_id=alice_uid,
    )
    s.add_medication(Medication(display="current"), owner_user_id=alice_uid)

    names = [m.display for m in s.list_medications()]
    assert names == ["current", "recent_stopped", "old_drug"]


def test_list_allergies_returns_active_first(trio):
    from claritymed.core.schemas import Allergy
    from claritymed.stores.profile import ProfileStore

    alice_uid = trio["admin"].user_id
    s = ProfileStore(alice_uid)
    s.add_allergy(
        Allergy(
            substance="peanut",
            severity="mild",
            source="self_report",
            end_date=date(2024, 1, 1),
        ),
        owner_user_id=alice_uid,
    )
    s.add_allergy(
        Allergy(
            substance="penicillin",
            severity="severe",
            source="self_report",
        ),
        owner_user_id=alice_uid,
    )
    names = [a.substance for a in s.list_allergies()]
    assert names == ["penicillin", "peanut"]
