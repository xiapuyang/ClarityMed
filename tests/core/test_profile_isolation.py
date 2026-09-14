"""The key safety test: alice cannot see bob's PHI, and vice versa."""

from __future__ import annotations

import pytest

from claritymed.context import MissingContextError
from claritymed.core.schemas import Allergy
from claritymed.errors import UserIdMismatch
from claritymed.stores.profile import ProfileStore


def test_each_user_sees_only_their_own_allergies(trio):
    alice_uid = trio["admin"].user_id
    bob_uid = trio["users"][0].user_id
    ProfileStore(alice_uid).add_allergy(
        Allergy(substance="penicillin", severity="severe", source="self_report"),
        owner_user_id=alice_uid,
    )
    ProfileStore(bob_uid).add_allergy(
        Allergy(substance="latex", severity="mild", source="self_report"),
        owner_user_id=bob_uid,
    )
    alice_view = [a.substance for a in ProfileStore(alice_uid).list_allergies()]
    bob_view = [a.substance for a in ProfileStore(bob_uid).list_allergies()]
    assert alice_view == ["penicillin"]
    assert bob_view == ["latex"]


def test_bob_cannot_see_alice_allergies(trio):
    """Renamed from the docs/plans 'critical safety test'.

    This is the can't-ever-fail invariant: if it goes red, the system is unsafe
    to ship and the regression must be the top priority.
    """
    alice_uid = trio["admin"].user_id
    bob_uid = trio["users"][0].user_id
    ProfileStore(alice_uid).add_allergy(
        Allergy(substance="penicillin", severity="anaphylactic", source="self_report"),
        owner_user_id=alice_uid,
    )
    bob_view = ProfileStore(bob_uid).list_allergies()
    assert bob_view == []


def test_for_current_user_requires_context_var():
    with pytest.raises(MissingContextError):
        ProfileStore.for_current_user()


def test_add_allergy_owner_mismatch_rejected(trio):
    alice_uid = trio["admin"].user_id
    bob_uid = trio["users"][0].user_id
    with pytest.raises(UserIdMismatch):
        ProfileStore(alice_uid).add_allergy(
            Allergy(substance="x", severity="mild", source="self_report"),
            owner_user_id=bob_uid,
        )
