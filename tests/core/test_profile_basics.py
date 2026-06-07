"""Tests for the per-user ``profile`` table (biometric basics).

Covers the new singleton row: get/upsert round-trip, cross-user isolation, and
the schema-level rejections (future birth_date, negative weight, invalid sex).
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from claritymed.core.schemas import Profile
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
