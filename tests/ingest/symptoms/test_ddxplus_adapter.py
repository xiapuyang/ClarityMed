"""Tests for the DDXPlus → CanonicalDataset adapter.

Stubs ``CLARITYMED_HOME`` to point at tmp_path so the adapter reads
fixture JSONs + computes hashes against fixture weights. Torch model
loading is monkeypatched out — only the canonical-construction +
manifest-chain logic is under test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from claritymed.core.symptoms.datasets.canonical import LoadedDataset
from claritymed.core.symptoms.schemas import DatasetSpec, ModelSpec
from claritymed.ingest.symptoms.ddxplus import adapter as adapter_mod
from claritymed.ingest.symptoms.ddxplus.adapter import DDXPlusAdapter


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect CLARITYMED_HOME so the adapter reads from tmp_path."""
    monkeypatch.setattr(adapter_mod._cfg, "CLARITYMED_HOME", tmp_path)
    return tmp_path


def _write_evidences(data_dir: Path) -> None:
    data = {
        "E_91": {
            "name": "E_91",
            "data_type": "B",
            "question_en": "Do you have a fever?",
            "question_fr": "Avez-vous de la fièvre?",
            "possible-values": [],
        },
        "E_55": {
            "name": "E_55",
            "data_type": "C",
            "question_en": "Where does it hurt?",
            "possible-values": ["V_123", "V_14"],
            "value_meaning": {
                "V_123": {"en": "nowhere", "fr": "nulle part"},
                "V_14": {"en": "iliac wing (right)", "fr": "aile iliaque (D)"},
            },
        },
    }
    (data_dir / "release_evidences.json").write_text(json.dumps(data), encoding="utf-8")


def _write_conditions(data_dir: Path) -> None:
    data = {
        "Acute appendicitis": {
            "condition_name": "Acute appendicitis",
            "cond-name-fr": "Appendicite aiguë",
            "icd10-id": "K35",
            "severity": 1,
        },
        "Common cold": {
            "condition_name": "Common cold",
            "icd10-id": "J00",
            "severity": 5,
        },
    }
    (data_dir / "release_conditions.json").write_text(
        json.dumps(data), encoding="utf-8"
    )


def _stub_weights_and_manifest(weights_dir: Path) -> str:
    """Write fake weights + matching manifest; return the pinned manifest sha."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / "weights.pt"
    weights_payload = b"fake-weights-bytes"
    weights_path.write_bytes(weights_payload)
    weights_sha = hashlib.sha256(weights_payload).hexdigest()
    manifest = {
        "manifest_version": 1,
        "dataset_id": "ddxplus",
        "model_id": "stub",
        "algorithm_module": "typed_basd",
        "training_commit": "deadbeef",
        "sha256": weights_sha,
        "eval": {},
    }
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
    (weights_dir / "manifest.json").write_bytes(manifest_bytes)
    return hashlib.sha256(manifest_bytes).hexdigest()


@pytest.fixture
def patched_torch_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace torch agent-loading with a stub that returns a sentinel object."""

    def _fake_loader(schema, n_dis, weights_path, device):  # noqa: ARG001
        return SimpleNamespace(loaded_from=str(weights_path), n_dis=n_dis)

    monkeypatch.setattr(adapter_mod, "_load_torch_agent", _fake_loader)


def _spec(model_ids=("typed_basd_v1",)) -> DatasetSpec:
    return DatasetSpec(
        id="ddxplus",
        model_ids=list(model_ids),
        maxstep=8,
    )


def _model_spec(model_id: str, manifest_sha: str) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        algorithm_module="typed_basd",
        weights_subpath=f"ddxplus/{model_id}",
        manifest_sha256=manifest_sha,
    )


# --- happy path ------------------------------------------------------------


def test_ddxplus_adapter_dataset_id_attribute() -> None:
    assert DDXPlusAdapter.dataset_id == "ddxplus"


def test_load_builds_canonical_with_slugified_conditions(
    fake_home: Path, patched_torch_loader
) -> None:
    data_dir = fake_home / "data" / "symptoms" / "ddxplus"
    data_dir.mkdir(parents=True)
    _write_evidences(data_dir)
    _write_conditions(data_dir)
    weights_dir = fake_home / "models" / "symptoms" / "ddxplus" / "typed_basd_v1"
    manifest_sha = _stub_weights_and_manifest(weights_dir)

    spec = _spec()
    loaded = DDXPlusAdapter.load(
        spec,
        {"typed_basd_v1": _model_spec("typed_basd_v1", manifest_sha)},
        device="cpu",
    )

    assert isinstance(loaded, LoadedDataset)
    assert loaded.canonical.id == "ddxplus"
    # Conditions slugified.
    ids = {c.id for c in loaded.canonical.conditions}
    assert ids == {"acute_appendicitis", "common_cold"}
    appendicitis = loaded.canonical.condition_by_id("acute_appendicitis")
    assert appendicitis.severity == 1
    assert appendicitis.icd10 == "K35"
    assert appendicitis.native_name["fr"] == "Appendicite aiguë"


def test_load_captures_native_question_text_and_value_meaning(
    fake_home: Path, patched_torch_loader
) -> None:
    data_dir = fake_home / "data" / "symptoms" / "ddxplus"
    data_dir.mkdir(parents=True)
    _write_evidences(data_dir)
    _write_conditions(data_dir)
    weights_dir = fake_home / "models" / "symptoms" / "ddxplus" / "typed_basd_v1"
    manifest_sha = _stub_weights_and_manifest(weights_dir)

    loaded = DDXPlusAdapter.load(
        _spec(),
        {"typed_basd_v1": _model_spec("typed_basd_v1", manifest_sha)},
        device="cpu",
    )

    fever = loaded.canonical.evidence_by_id("E_91")
    assert fever.dtype == "B"
    assert fever.native_question_text["en"] == "Do you have a fever?"
    assert fever.native_question_text["fr"] == "Avez-vous de la fièvre?"

    pain = loaded.canonical.evidence_by_id("E_55")
    assert pain.dtype == "C"
    assert {v.raw for v in pain.values} == {"V_123", "V_14"}
    assert pain.native_value_labels["V_14"]["en"] == "iliac wing (right)"


def test_load_marks_high_specificity_evidences(
    fake_home: Path, patched_torch_loader
) -> None:
    data_dir = fake_home / "data" / "symptoms" / "ddxplus"
    data_dir.mkdir(parents=True)
    _write_evidences(data_dir)
    _write_conditions(data_dir)
    weights_dir = fake_home / "models" / "symptoms" / "ddxplus" / "typed_basd_v1"
    manifest_sha = _stub_weights_and_manifest(weights_dir)

    spec = DatasetSpec(
        id="ddxplus",
        model_ids=["typed_basd_v1"],
        maxstep=8,
        severity_high_specificity_evidence_ids=["E_91"],
    )
    loaded = DDXPlusAdapter.load(
        spec,
        {"typed_basd_v1": _model_spec("typed_basd_v1", manifest_sha)},
        device="cpu",
    )

    assert loaded.canonical.evidence_by_id("E_91").is_high_specificity is True
    assert loaded.canonical.evidence_by_id("E_55").is_high_specificity is False


def test_load_multi_model_loads_each_checkpoint(
    fake_home: Path, patched_torch_loader
) -> None:
    data_dir = fake_home / "data" / "symptoms" / "ddxplus"
    data_dir.mkdir(parents=True)
    _write_evidences(data_dir)
    _write_conditions(data_dir)
    # Two weights subpaths.
    shas: dict[str, str] = {}
    for mid in ("v1", "v2"):
        weights_dir = fake_home / "models" / "symptoms" / "ddxplus" / mid
        shas[mid] = _stub_weights_and_manifest(weights_dir)

    spec = DatasetSpec(
        id="ddxplus",
        model_ids=["v1", "v2"],
        model_selection="round_robin",
        maxstep=8,
    )
    loaded = DDXPlusAdapter.load(
        spec,
        {
            "v1": ModelSpec(
                id="v1",
                algorithm_module="typed_basd",
                weights_subpath="ddxplus/v1",
                manifest_sha256=shas["v1"],
            ),
            "v2": ModelSpec(
                id="v2",
                algorithm_module="typed_basd",
                weights_subpath="ddxplus/v2",
                manifest_sha256=shas["v2"],
            ),
        },
        device="cpu",
    )
    assert set(loaded.models) == {"v1", "v2"}


# --- fail-loud paths -------------------------------------------------------


def test_load_missing_manifest_raises(fake_home: Path, patched_torch_loader) -> None:
    data_dir = fake_home / "data" / "symptoms" / "ddxplus"
    data_dir.mkdir(parents=True)
    _write_evidences(data_dir)
    _write_conditions(data_dir)
    weights_dir = fake_home / "models" / "symptoms" / "ddxplus" / "typed_basd_v1"
    weights_dir.mkdir(parents=True)
    # No manifest.json written.
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        DDXPlusAdapter.load(
            _spec(),
            {"typed_basd_v1": _model_spec("typed_basd_v1", "a" * 64)},
            device="cpu",
        )


def test_load_manifest_sha_mismatch_raises(
    fake_home: Path, patched_torch_loader
) -> None:
    data_dir = fake_home / "data" / "symptoms" / "ddxplus"
    data_dir.mkdir(parents=True)
    _write_evidences(data_dir)
    _write_conditions(data_dir)
    weights_dir = fake_home / "models" / "symptoms" / "ddxplus" / "typed_basd_v1"
    _stub_weights_and_manifest(weights_dir)
    # Pin a wrong manifest sha.
    with pytest.raises(RuntimeError, match="manifest sha256 mismatch"):
        DDXPlusAdapter.load(
            _spec(),
            {"typed_basd_v1": _model_spec("typed_basd_v1", "f" * 64)},
            device="cpu",
        )


def test_load_weights_sha_mismatch_raises(
    fake_home: Path, patched_torch_loader
) -> None:
    data_dir = fake_home / "data" / "symptoms" / "ddxplus"
    data_dir.mkdir(parents=True)
    _write_evidences(data_dir)
    _write_conditions(data_dir)
    weights_dir = fake_home / "models" / "symptoms" / "ddxplus" / "typed_basd_v1"
    manifest_sha = _stub_weights_and_manifest(weights_dir)
    # Mutate the weights file post-manifest so its sha drifts.
    (weights_dir / "weights.pt").write_bytes(b"tampered-bytes")
    with pytest.raises(RuntimeError, match="weights sha256 mismatch"):
        DDXPlusAdapter.load(
            _spec(),
            {"typed_basd_v1": _model_spec("typed_basd_v1", manifest_sha)},
            device="cpu",
        )


def test_load_missing_data_files_raise(fake_home: Path, patched_torch_loader) -> None:
    """Adapter expects data_dir to exist; surfaces FileNotFoundError from loaders."""
    weights_dir = fake_home / "models" / "symptoms" / "ddxplus" / "typed_basd_v1"
    manifest_sha = _stub_weights_and_manifest(weights_dir)
    with pytest.raises(FileNotFoundError, match="release_evidences"):
        DDXPlusAdapter.load(
            _spec(),
            {"typed_basd_v1": _model_spec("typed_basd_v1", manifest_sha)},
            device="cpu",
        )
