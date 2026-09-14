"""Tests for ``ingest.records.fact_writer``.

Two responsibilities to pin:

1. The **profile merge dance** — only template-provided keys overlay;
   None values don't overwrite; replay-after-crash short-circuits;
   validation errors are localized to the profile fact (allergies still
   apply).
2. **Fact equivalence** — allergy by substance.casefold; condition /
   medication by (display.casefold, onset_date). Dose / frequency
   changes are skipped, not updates.
"""

from __future__ import annotations

from datetime import date

import pytest

from claritymed.core.schemas.patient import Allergy, Condition, Medication
from claritymed.ingest.records.fact_writer import (
    apply_facts,
    compute_fact_row_id,
)
from claritymed.ingest.records.template_schema import FactsBundle


# --- fixtures ---------------------------------------------------------


@pytest.fixture
def redirected_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CLARITYMED_DATA_DIR", str(tmp_path / "data"))
    import importlib

    from claritymed import config as _cfg

    importlib.reload(_cfg)


@pytest.fixture
def test_user(redirected_data_dir):
    from claritymed.stores.account import init_user

    return init_user("test", display_name="Test")


# --- profile merge ----------------------------------------------------


def test_profile_merge_overlays_only_provided_keys(test_user):
    """Existing Profile(sex='female', weight_kg=60); template provides
    only ``birth_date`` → other fields unchanged."""
    from claritymed.core.schemas.patient import Profile
    from claritymed.stores.profile import ProfileStore

    store = ProfileStore("test")
    store.upsert_profile(Profile(sex="female", weight_kg=60.0), owner_user_id="test")

    results = apply_facts(
        "test",
        FactsBundle(profile={"birth_date": "2010-05-21"}),
    )
    assert len(results) == 1
    assert results[0].status == "done"

    updated = store.get_profile()
    assert updated.sex == "female"
    assert updated.weight_kg == 60.0
    assert updated.birth_date == date(2010, 5, 21)


def test_profile_merge_explicit_value_wins(test_user):
    from claritymed.core.schemas.patient import Profile
    from claritymed.stores.profile import ProfileStore

    store = ProfileStore("test")
    store.upsert_profile(Profile(weight_kg=60.0), owner_user_id="test")

    apply_facts("test", FactsBundle(profile={"weight_kg": 62.0}))
    assert store.get_profile().weight_kg == 62.0


def test_profile_merge_none_values_do_not_overwrite(test_user):
    """Template-provided ``None`` is skill noise (default dump rather
    than omitted key). It must not wipe out a real existing value."""
    from claritymed.core.schemas.patient import Profile
    from claritymed.stores.profile import ProfileStore

    store = ProfileStore("test")
    store.upsert_profile(Profile(sex="female", weight_kg=60.0), owner_user_id="test")

    apply_facts(
        "test",
        FactsBundle(profile={"sex": "female", "weight_kg": None}),
    )
    assert store.get_profile().weight_kg == 60.0


def test_profile_merge_creates_row_if_none_exists(test_user):
    from claritymed.stores.profile import ProfileStore

    store = ProfileStore("test")
    assert store.get_profile() is None

    apply_facts("test", FactsBundle(profile={"sex": "male"}))
    profile = store.get_profile()
    assert profile is not None
    assert profile.sex == "male"


def test_profile_merge_validation_error_does_not_block_allergies(test_user):
    """Locks Key Decision §"Profile validation failure handling".

    A bogus birth_date in the template fails ``Profile.model_validate``
    inside the profile branch. The allergies branch must still run —
    isolating one bad fact to one error row instead of aborting the
    whole bundle."""
    from claritymed.stores.profile import ProfileStore

    store = ProfileStore("test")

    results = apply_facts(
        "test",
        FactsBundle(
            profile={"birth_date": "1850-01-01"},  # >130 years → ValidationError
            allergies=[
                Allergy(
                    substance="penicillin",
                    severity="severe",
                    source="clinical_record",
                )
            ],
        ),
    )
    statuses = {r.fact_kind: r.status for r in results}
    assert statuses["profile"] == "error"
    assert statuses["allergy"] == "done"
    assert any(
        r.error_detail and r.error_detail.startswith("profile_validation:")
        for r in results
    )
    assert len(store.list_allergies()) == 1


def test_profile_merge_replay_short_circuits_no_update_time_bump(test_user):
    """Replay-after-crash with identical content must skip the
    upsert (otherwise update_time bumps misleadingly)."""
    from claritymed.core.schemas.patient import Profile
    from claritymed.stores.profile import ProfileStore

    store = ProfileStore("test")
    store.upsert_profile(Profile(sex="female"), owner_user_id="test")
    before = store.get_profile()

    apply_facts("test", FactsBundle(profile={"sex": "female"}))
    after = store.get_profile()
    # Same content, no upsert call — sex still female, no spurious change.
    assert after.sex == before.sex


# --- allergy ---------------------------------------------------------


def test_allergy_happy_path(test_user):
    from claritymed.stores.profile import ProfileStore

    store = ProfileStore("test")
    results = apply_facts(
        "test",
        FactsBundle(
            allergies=[
                Allergy(
                    substance="penicillin",
                    severity="severe",
                    source="clinical_record",
                )
            ]
        ),
    )
    assert results[0].status == "done"
    assert len(store.list_allergies()) == 1


def test_allergy_case_insensitive_dedup(test_user):
    """Existing ``Penicillin``; template ``penicillin`` → skipped."""
    from claritymed.stores.profile import ProfileStore

    store = ProfileStore("test")
    store.add_allergy(
        Allergy(substance="Penicillin", severity="severe", source="clinical_record"),
        owner_user_id="test",
    )

    results = apply_facts(
        "test",
        FactsBundle(
            allergies=[
                Allergy(
                    substance="penicillin",
                    severity="severe",
                    source="clinical_record",
                )
            ]
        ),
    )
    assert results[0].status == "skipped"
    assert results[0].error_detail == "already_exists"
    assert len(store.list_allergies()) == 1


# --- condition --------------------------------------------------------


def test_condition_distinct_onset_dates_are_separate_rows(test_user):
    """Existing T2DM with null onset; template same display with
    onset_date=2024-01-15 → both rows present (distinct equivalence key)."""
    from claritymed.stores.profile import ProfileStore

    store = ProfileStore("test")
    store.add_condition(
        Condition(display="Type 2 diabetes"),
        owner_user_id="test",
    )

    results = apply_facts(
        "test",
        FactsBundle(
            conditions=[
                Condition(display="Type 2 diabetes", onset_date=date(2024, 1, 15))
            ]
        ),
    )
    assert results[0].status == "done"
    assert len(store.list_conditions()) == 2


def test_condition_same_key_skips(test_user):
    from claritymed.stores.profile import ProfileStore

    store = ProfileStore("test")
    store.add_condition(
        Condition(display="Hypertension", onset_date=date(2023, 5, 1)),
        owner_user_id="test",
    )

    results = apply_facts(
        "test",
        FactsBundle(
            conditions=[Condition(display="hypertension", onset_date=date(2023, 5, 1))]
        ),
    )
    assert results[0].status == "skipped"


# --- medication -------------------------------------------------------


def test_medication_dose_change_is_skipped_not_updated(test_user):
    """Locks Scope Boundary: dose / frequency edits via import are
    NOT updates — they're skips. Corrections go through chat tools."""
    from claritymed.stores.profile import ProfileStore

    store = ProfileStore("test")
    store.add_medication(
        Medication(display="metformin", onset_date=date(2024, 1, 20), dose="500mg"),
        owner_user_id="test",
    )

    results = apply_facts(
        "test",
        FactsBundle(
            medications=[
                Medication(
                    display="metformin",
                    onset_date=date(2024, 1, 20),
                    dose="1000mg",
                )
            ]
        ),
    )
    assert results[0].status == "skipped"

    saved = store.list_medications()
    assert len(saved) == 1
    assert saved[0].dose == "500mg"  # untouched


# --- row_id ----------------------------------------------------------


def test_fact_row_id_format():
    rid = compute_fact_row_id("alice", "allergy", ["penicillin"])
    assert rid.startswith("fact:alice:allergy:")
    assert len(rid.rsplit(":", 1)[1]) == 8


def test_fact_row_id_stable_under_replay():
    """Same fact applied twice computes the same row_id → resume works."""
    a = compute_fact_row_id("alice", "allergy", ["penicillin"])
    b = compute_fact_row_id("alice", "allergy", ["penicillin"])
    assert a == b


def test_fact_row_id_json_encoding_robust_to_pipe_chars():
    """JSON-encoded canonical key prevents the ``shellfish | crab`` vs
    ``shellfish`` ``|``-separator collision."""
    a = compute_fact_row_id("alice", "allergy", ["shellfish | crab"])
    b = compute_fact_row_id("alice", "allergy", ["shellfish"])
    assert a != b


def test_fact_row_id_includes_user_in_namespace():
    """Different users with same substance → distinct row_ids."""
    a = compute_fact_row_id("alice", "allergy", ["penicillin"])
    b = compute_fact_row_id("bob", "allergy", ["penicillin"])
    assert a != b


# --- resume + integration --------------------------------------------


def test_resume_mid_batch_no_duplicates(test_user):
    """Write 3 allergies, simulate "the 4th + 5th got dropped" by re-
    applying the full set on resume → live DB has exactly 5, no dupes."""
    from claritymed.stores.profile import ProfileStore

    store = ProfileStore("test")
    allergies = [
        Allergy(substance=s, severity="moderate", source="self_report")
        for s in ("a", "b", "c", "d", "e")
    ]

    apply_facts("test", FactsBundle(allergies=allergies[:3]))
    assert len(store.list_allergies()) == 3

    # Resume: full set re-applied.
    results = apply_facts("test", FactsBundle(allergies=allergies))
    by_status = [r.status for r in results]
    assert by_status.count("skipped") == 3
    assert by_status.count("done") == 2
    assert {a.substance for a in store.list_allergies()} == {"a", "b", "c", "d", "e"}


def test_apply_facts_returns_one_result_per_attempted_fact(test_user):
    results = apply_facts(
        "test",
        FactsBundle(
            profile={"sex": "female"},
            allergies=[Allergy(substance="x", severity="mild", source="self_report")],
            conditions=[Condition(display="y")],
            medications=[Medication(display="z")],
        ),
    )
    kinds = [r.fact_kind for r in results]
    assert kinds == ["profile", "allergy", "condition", "medication"]


def test_apply_facts_empty_bundle_returns_no_results(test_user):
    """Edge case: skill emitted facts: {} → no work, no results, no
    spurious upserts."""
    results = apply_facts("test", FactsBundle())
    assert results == []
