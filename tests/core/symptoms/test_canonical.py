"""Tests for canonical dataset types + adapter registry.

The canonical layer is dataset-agnostic, so these tests use synthetic
evidences/conditions rather than DDXPlus fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest

from claritymed.core.symptoms.datasets import (
    CanonicalCondition,
    CanonicalDataset,
    CanonicalEvidence,
    CanonicalValue,
    LoadedDataset,
    LoadedModel,
    available_adapters,
    build_dataset,
    register_adapter,
    slugify_condition,
    unregister_adapter,
)
from claritymed.core.symptoms.schemas import DatasetSpec, ModelSpec
from claritymed.errors import UnknownDatasetError
from claritymed.ingest.symptoms.typed_basd import build_layout

_DUMMY_SHA = "a" * 64


def _evidences() -> list[CanonicalEvidence]:
    return [
        CanonicalEvidence(
            id="E_91", idx=0, dtype="B", native_question_text={"en": "Fever?"}
        ),
        CanonicalEvidence(
            id="E_55",
            idx=1,
            dtype="C",
            values=[
                CanonicalValue(raw="V_123", local_idx=0),
                CanonicalValue(raw="V_14", local_idx=1),
            ],
        ),
    ]


def _conditions() -> list[CanonicalCondition]:
    return [
        CanonicalCondition(
            id="spontaneous_pneumothorax",
            idx=0,
            severity=2,
            icd10="J93",
            native_name={"en": "Spontaneous pneumothorax"},
        ),
        CanonicalCondition(id="common_cold", idx=1, severity=5),
    ]


def _layout() -> dict:
    return build_layout(
        [
            {"name": "E_91", "dtype": "B", "values": []},
            {"name": "E_55", "dtype": "C", "values": ["V_123", "V_14"]},
        ]
    )


def _canonical() -> CanonicalDataset:
    return CanonicalDataset.build(
        id="synth",
        evidences=_evidences(),
        conditions=_conditions(),
        layout=_layout(),
        severity_vector=np.array([2.0, 5.0]),
    )


# --- slugify --------------------------------------------------------------


def test_slugify_strips_punctuation_and_lowercases() -> None:
    assert slugify_condition("Spontaneous pneumothorax") == "spontaneous_pneumothorax"
    assert (
        slugify_condition("ANCA-associated vasculitis") == "anca_associated_vasculitis"
    )
    assert slugify_condition("Type 2 diabetes (T2D)") == "type_2_diabetes_t2d"


def test_slugify_collapses_runs_and_trims_edges() -> None:
    assert slugify_condition("  Foo --- Bar  ") == "foo_bar"
    assert slugify_condition("___") == "unknown"


# --- CanonicalDataset lookups --------------------------------------------


def test_canonical_evidence_by_id_round_trip() -> None:
    cd = _canonical()
    assert cd.evidence_by_id("E_91").idx == 0
    assert cd.evidence_by_idx(1).id == "E_55"


def test_canonical_condition_by_id_round_trip() -> None:
    cd = _canonical()
    assert cd.condition_by_id("spontaneous_pneumothorax").severity == 2
    assert cd.condition_by_idx(1).id == "common_cold"


def test_canonical_evidence_value_lookup() -> None:
    cd = _canonical()
    ev = cd.evidence_by_id("E_55")
    assert ev.value_by_raw("V_14").local_idx == 1
    assert ev.value_by_raw("V_nope") is None


def test_canonical_build_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicate evidence"):
        CanonicalDataset.build(
            id="bad",
            evidences=[_evidences()[0], _evidences()[0]],
            conditions=_conditions(),
            layout=_layout(),
            severity_vector=np.array([2.0, 5.0]),
        )


def test_canonical_build_rejects_severity_length_mismatch() -> None:
    with pytest.raises(ValueError, match="severity_vector length"):
        CanonicalDataset.build(
            id="bad",
            evidences=_evidences(),
            conditions=_conditions(),
            layout=_layout(),
            severity_vector=np.array([2.0]),
        )


def test_canonical_unknown_lookup_raises_keyerror() -> None:
    cd = _canonical()
    with pytest.raises(KeyError):
        cd.evidence_by_id("does_not_exist")
    with pytest.raises(KeyError):
        cd.condition_by_idx(99)


# --- LoadedDataset model selection ----------------------------------------


def _spec(model_ids: list[str], selection: str = "first") -> DatasetSpec:
    return DatasetSpec(
        id="synth",
        model_ids=model_ids,
        model_selection=selection,  # type: ignore[arg-type]
        maxstep=8,
    )


def _model_spec(mid: str) -> ModelSpec:
    return ModelSpec(
        id=mid,
        algorithm_module="typed_basd",
        weights_subpath=f"synth/{mid}",
        manifest_sha256=_DUMMY_SHA,
    )


def _loaded(model_ids: list[str], selection: str = "first") -> LoadedDataset:
    return LoadedDataset(
        spec=_spec(model_ids, selection=selection),
        canonical=_canonical(),
        models={
            mid: LoadedModel(spec=_model_spec(mid), agent=object(), manifest={})
            for mid in model_ids
        },
    )


def test_loaded_dataset_first_selection_picks_primary() -> None:
    ds = _loaded(["a", "b"], selection="first")
    for _ in range(3):
        assert ds.select_model().spec.id == "a"


def test_loaded_dataset_round_robin_cycles() -> None:
    ds = _loaded(["a", "b", "c"], selection="round_robin")
    picks = [ds.select_model().spec.id for _ in range(6)]
    assert picks == ["a", "b", "c", "a", "b", "c"]


def test_loaded_dataset_model_lookup_raises_on_unknown() -> None:
    ds = _loaded(["a"])
    with pytest.raises(KeyError):
        ds.model("z")


# --- registry dispatch ----------------------------------------------------


class _SyntheticAdapter:
    """Minimal DatasetAdapter for registry tests."""

    dataset_id = "test_synth_ds"

    @classmethod
    def load(cls, spec, model_specs, *, device):  # noqa: ARG003
        return LoadedDataset(
            spec=spec,
            canonical=_canonical(),
            models={
                mid: LoadedModel(spec=ms, agent=object(), manifest={})
                for mid, ms in model_specs.items()
            },
        )


@pytest.fixture
def _registered_adapter():
    register_adapter(_SyntheticAdapter)
    yield _SyntheticAdapter
    unregister_adapter(_SyntheticAdapter.dataset_id)


def test_registry_lists_registered_adapters(_registered_adapter) -> None:
    assert "test_synth_ds" in available_adapters()


def test_registry_build_dispatches_to_adapter(_registered_adapter) -> None:
    spec = DatasetSpec(id="test_synth_ds", model_ids=["m"], maxstep=8)
    m = ModelSpec(
        id="m",
        algorithm_module="typed_basd",
        weights_subpath="x/y",
        manifest_sha256=_DUMMY_SHA,
    )
    loaded = build_dataset(spec, [m], device="cpu")
    assert loaded.spec.id == "test_synth_ds"
    assert "m" in loaded.models


def test_registry_unknown_id_raises_unknown_dataset_error() -> None:
    spec = DatasetSpec(id="never_registered", model_ids=["m"], maxstep=8)
    m = ModelSpec(
        id="m",
        algorithm_module="typed_basd",
        weights_subpath="x/y",
        manifest_sha256=_DUMMY_SHA,
    )
    with pytest.raises(UnknownDatasetError, match="no adapter registered"):
        build_dataset(spec, [m], device="cpu")


def test_registry_missing_model_spec_raises(_registered_adapter) -> None:
    spec = DatasetSpec(id="test_synth_ds", model_ids=["m1", "m2"], maxstep=8)
    only_m1 = ModelSpec(
        id="m1",
        algorithm_module="typed_basd",
        weights_subpath="x/y",
        manifest_sha256=_DUMMY_SHA,
    )
    with pytest.raises(UnknownDatasetError, match="m2"):
        build_dataset(spec, [only_m1], device="cpu")


def test_registry_refuses_to_clobber_existing_id(_registered_adapter) -> None:
    class _Dup:
        dataset_id = "test_synth_ds"

        @classmethod
        def load(cls, spec, model_specs, *, device):  # noqa: ARG003
            raise NotImplementedError

    with pytest.raises(ValueError, match="already registered"):
        register_adapter(_Dup)


def test_registry_register_is_idempotent_for_same_class(_registered_adapter) -> None:
    register_adapter(_SyntheticAdapter)  # second call is a no-op
    assert "test_synth_ds" in available_adapters()
