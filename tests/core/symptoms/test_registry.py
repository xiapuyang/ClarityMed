"""Tests for the plugin-side :class:`DatasetRegistry` (KTD-9 resolution)."""

from __future__ import annotations

import pytest

from claritymed.core.symptoms.registry import DatasetRegistry
from claritymed.core.symptoms.schemas import DatasetSpec
from claritymed.errors import UnknownDatasetError


def _spec(id_: str, enabled: bool = True) -> DatasetSpec:
    return DatasetSpec(id=id_, enabled=enabled, model_ids=[f"{id_}_m1"])


# --- list / has / get ----------------------------------------------------


def test_list_enabled_returns_enabled_subset() -> None:
    reg = DatasetRegistry([_spec("a"), _spec("b", enabled=False), _spec("c")])
    assert [d.id for d in reg.list_enabled()] == ["a", "c"]


def test_has_and_get_lookups() -> None:
    reg = DatasetRegistry([_spec("a"), _spec("b")])
    assert reg.has("a") is True
    assert reg.has("missing") is False
    assert reg.get("a").id == "a"
    assert reg.get("missing") is None


# --- KTD-9 precedence -----------------------------------------------------


def test_resolve_with_no_enabled_datasets_returns_none() -> None:
    reg = DatasetRegistry([_spec("a", enabled=False)])
    assert reg.resolve() is None
    assert reg.resolve(hint="a") is None


def test_resolve_explicit_hint_matches() -> None:
    reg = DatasetRegistry([_spec("a"), _spec("b")])
    assert reg.resolve(hint="b").id == "b"


def test_resolve_unknown_hint_falls_through_to_registry_order() -> None:
    """Soft hint per KTD-9 — LLM hallucination doesn't raise."""
    reg = DatasetRegistry([_spec("a"), _spec("b")])
    # No eligibility scores → registry-list order wins → first enabled.
    assert reg.resolve(hint="nonexistent").id == "a"


def test_resolve_eligibility_scores_pick_highest() -> None:
    reg = DatasetRegistry([_spec("a"), _spec("b"), _spec("c")])
    picked = reg.resolve(eligibility_scores={"a": 0.4, "b": 0.9, "c": 0.6})
    assert picked.id == "b"


def test_resolve_eligibility_tie_broken_by_dataset_id() -> None:
    reg = DatasetRegistry([_spec("z"), _spec("a"), _spec("m")])
    picked = reg.resolve(eligibility_scores={"z": 0.5, "a": 0.5, "m": 0.5})
    assert picked.id == "a"  # min(id)


def test_resolve_eligibility_zero_signal_falls_through_to_registry_order() -> None:
    """All-zero scores → no eligibility-driven pick → use registry order."""
    reg = DatasetRegistry([_spec("b"), _spec("a")])
    picked = reg.resolve(eligibility_scores={"a": 0.0, "b": 0.0})
    assert picked.id == "b"  # registry-list order


def test_resolve_hint_wins_over_eligibility() -> None:
    """Explicit hint dominates even when eligibility prefers a different dataset."""
    reg = DatasetRegistry([_spec("a"), _spec("b")])
    picked = reg.resolve(hint="a", eligibility_scores={"a": 0.1, "b": 0.95})
    assert picked.id == "a"


def test_resolve_default_returns_first_enabled() -> None:
    reg = DatasetRegistry([_spec("z"), _spec("a")])
    assert reg.resolve().id == "z"  # operator's preferred order


# --- strict variant -------------------------------------------------------


def test_resolve_or_raise_on_hint_mismatch_returns_match() -> None:
    reg = DatasetRegistry([_spec("ddxplus")])
    assert reg.resolve_or_raise_on_hint_mismatch("ddxplus").id == "ddxplus"


def test_resolve_or_raise_on_hint_mismatch_raises_for_unknown() -> None:
    reg = DatasetRegistry([_spec("a")])
    with pytest.raises(UnknownDatasetError, match="not in registry"):
        reg.resolve_or_raise_on_hint_mismatch("missing")


def test_disabled_dataset_not_visible_to_strict_resolve() -> None:
    reg = DatasetRegistry([_spec("a", enabled=False)])
    with pytest.raises(UnknownDatasetError):
        reg.resolve_or_raise_on_hint_mismatch("a")
