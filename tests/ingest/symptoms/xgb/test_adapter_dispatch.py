"""Adapter dispatch tests for the ``algorithm_module: xgb`` load path.

Exercises :class:`DDXPlusAdapter.load` end-to-end with a real xgb
checkpoint (produced by :func:`train_xgb_agent` on a synthetic corpus).
The typed-BASD path is already covered by
``tests/ingest/symptoms/test_ddxplus_adapter.py`` — this file adds the
xgb-specific fail-loud edges the plan calls out:

* algorithm_module ↔ manifest mismatch (config says ``xgb``, manifest
  says ``typed_basd``) → fail loud.
* feature_columns drift between train-time and serve-time schemas →
  fail loud.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("xgboost")

from claritymed.core.symptoms.datasets.canonical import LoadedDataset
from claritymed.core.symptoms.schemas import DatasetSpec, ModelSpec
from claritymed.ingest.symptoms.ddxplus import adapter as adapter_mod
from claritymed.ingest.symptoms.ddxplus.adapter import DDXPlusAdapter
from claritymed.ingest.symptoms.xgb.algorithm import XgbAgent
from claritymed.ingest.symptoms.xgb.encoding import feature_columns_from_schema
from claritymed.ingest.symptoms.xgb.train import _write_manifest, train_xgb_agent


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect CLARITYMED_HOME so the adapter reads from tmp_path."""
    monkeypatch.setattr(adapter_mod._cfg, "CLARITYMED_HOME", tmp_path)
    return tmp_path


def _write_evidences(data_dir: Path) -> None:
    """Two binary evidences — matches the mini schema used by test_train."""
    data = {
        "E_A": {
            "name": "E_A",
            "data_type": "B",
            "question_en": "Symptom A?",
            "possible-values": [],
        },
        "E_B": {
            "name": "E_B",
            "data_type": "B",
            "question_en": "Symptom B?",
            "possible-values": [],
        },
    }
    (data_dir / "release_evidences.json").write_text(json.dumps(data), encoding="utf-8")


def _write_conditions(data_dir: Path) -> None:
    data = {
        "ClassA": {"condition_name": "ClassA", "severity": 3},
        "ClassB": {"condition_name": "ClassB", "severity": 3},
    }
    (data_dir / "release_conditions.json").write_text(
        json.dumps(data), encoding="utf-8"
    )


def _tiny_patients(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    patients: list[dict] = []
    for i in range(n):
        cls = i % 2
        has_a = rng.random() < (0.9 if cls == 0 else 0.1)
        has_b = rng.random() < (0.1 if cls == 0 else 0.9)
        bin_pos: set[int] = set()
        if has_a:
            bin_pos.add(0)
        if has_b:
            bin_pos.add(1)
        init = next(iter(bin_pos)) if bin_pos else 0
        patients.append(
            dict(
                bin_pos=bin_pos,
                cat_val={},
                multi_val={},
                pos=bin_pos.copy(),
                init=init,
                d=cls,
                age=3,
                sex=0,
                diff=np.array([1.0 - cls, float(cls)]),
            )
        )
    return patients


def _write_xgb_checkpoint(
    weights_dir: Path,
    schema: dict,
    diseases_trained: list[str],
    *,
    override_manifest_algorithm: str | None = None,
    override_feature_columns: list[str] | None = None,
) -> str:
    """Train a mini XgbAgent, save weights + manifest, return pinned manifest sha."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    columns, _labels, columns_idx = feature_columns_from_schema(schema)
    train_pats = _tiny_patients(80, seed=1)
    agent = train_xgb_agent(
        schema=schema,
        train_patients=train_pats,
        val_patients=[],
        columns=columns,
        columns_idx=columns_idx,
        n_classes=2,
        max_depth=3,
        n_estimators=20,
        lr=0.3,
        calibration="none",
        mask_policy="full",
        keep_lo=0.3,
        keep_hi=0.7,
        stop_thres=0.9,
        ig_smoothing=0.05,
        seed=0,
    )
    weights_path = weights_dir / "weights.pkl"
    agent.save(weights_path)

    from claritymed.ingest.symptoms.xgb.train import _sha256_file

    weights_sha = _sha256_file(weights_path)
    manifest = {
        "manifest_version": 1,
        "dataset_id": "ddxplus",
        "model_id": "test_xgb",
        "algorithm_module": override_manifest_algorithm or "xgb",
        "algorithm_module_version": 1,
        "training_commit": "deadbeef",
        "sha256": weights_sha,
        "train_params": {},
        "diseases_trained": diseases_trained,
        "training_target": "pathology",
        "feature_columns": override_feature_columns or columns,
        "feature_importance_top_k": [],
        "eval": {"IL": 3.0, "ACC": 95.0, "maxstep": 3},
    }
    manifest_path = weights_dir / "manifest.json"
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    return hashlib.sha256(manifest_bytes).hexdigest()


def _spec() -> DatasetSpec:
    return DatasetSpec(id="ddxplus", model_ids=["xgb_test_v1"])


def _model_spec(
    manifest_sha: str, *, algorithm: str = "xgb", subpath: str = "xgb_test_v1"
) -> ModelSpec:
    return ModelSpec(
        id="xgb_test_v1",
        algorithm_module=algorithm,
        weights_subpath=f"ddxplus/{subpath}",
        manifest_sha256=manifest_sha,
        maxstep=6,
    )


# --- happy path ------------------------------------------------------------


def test_load_xgb_checkpoint_returns_working_agent(fake_home: Path) -> None:
    data_dir = fake_home / "data" / "symptoms" / "ddxplus"
    data_dir.mkdir(parents=True)
    _write_evidences(data_dir)
    _write_conditions(data_dir)
    from claritymed.ingest.symptoms.ddxplus.schema import load_evidence_schema

    schema = load_evidence_schema(data_dir)
    weights_dir = fake_home / "models" / "symptoms" / "ddxplus" / "xgb_test_v1"
    manifest_sha = _write_xgb_checkpoint(
        weights_dir, schema, diseases_trained=["ClassA", "ClassB"]
    )

    loaded = DDXPlusAdapter.load(
        _spec(),
        {"xgb_test_v1": _model_spec(manifest_sha)},
        device="cpu",
    )
    assert isinstance(loaded, LoadedDataset)
    xgb_model = loaded.models["xgb_test_v1"]
    assert isinstance(xgb_model.agent, XgbAgent)
    # The loaded agent must still classify a synthetic patient.
    from claritymed.ingest.symptoms.typed_basd import TypedEnv

    env = TypedEnv(_tiny_patients(4, seed=99), schema, n_dis=2)
    s, _ = env.initialize_state(4)
    argmax, probs = xgb_model.agent.diagnose(s)
    assert argmax.shape == (4,)
    assert probs.shape == (4, 2)


# --- fail-loud paths -------------------------------------------------------


def test_load_manifest_algorithm_mismatch_raises(fake_home: Path) -> None:
    """Config says xgb, manifest says typed_basd → fail loud."""
    data_dir = fake_home / "data" / "symptoms" / "ddxplus"
    data_dir.mkdir(parents=True)
    _write_evidences(data_dir)
    _write_conditions(data_dir)
    from claritymed.ingest.symptoms.ddxplus.schema import load_evidence_schema

    schema = load_evidence_schema(data_dir)
    weights_dir = fake_home / "models" / "symptoms" / "ddxplus" / "xgb_test_v1"
    manifest_sha = _write_xgb_checkpoint(
        weights_dir,
        schema,
        diseases_trained=["ClassA", "ClassB"],
        override_manifest_algorithm="typed_basd",
    )
    with pytest.raises(RuntimeError, match="algorithm_module mismatch"):
        DDXPlusAdapter.load(
            _spec(),
            {"xgb_test_v1": _model_spec(manifest_sha)},
            device="cpu",
        )


def test_load_feature_columns_mismatch_raises(fake_home: Path) -> None:
    """A manifest with a stale feature_columns list fails loud."""
    data_dir = fake_home / "data" / "symptoms" / "ddxplus"
    data_dir.mkdir(parents=True)
    _write_evidences(data_dir)
    _write_conditions(data_dir)
    from claritymed.ingest.symptoms.ddxplus.schema import load_evidence_schema

    schema = load_evidence_schema(data_dir)
    weights_dir = fake_home / "models" / "symptoms" / "ddxplus" / "xgb_test_v1"
    manifest_sha = _write_xgb_checkpoint(
        weights_dir,
        schema,
        diseases_trained=["ClassA", "ClassB"],
        # Manifest declares columns that don't include the live schema's E_B.
        override_feature_columns=["E_A", "E_STALE"],
    )
    with pytest.raises(RuntimeError, match="feature_columns mismatch"):
        DDXPlusAdapter.load(
            _spec(),
            {"xgb_test_v1": _model_spec(manifest_sha)},
            device="cpu",
        )


def test_load_missing_weights_pkl_raises(fake_home: Path) -> None:
    """Manifest present + weights.pkl missing → FileNotFoundError with filename."""
    data_dir = fake_home / "data" / "symptoms" / "ddxplus"
    data_dir.mkdir(parents=True)
    _write_evidences(data_dir)
    _write_conditions(data_dir)
    from claritymed.ingest.symptoms.ddxplus.schema import load_evidence_schema

    schema = load_evidence_schema(data_dir)
    weights_dir = fake_home / "models" / "symptoms" / "ddxplus" / "xgb_test_v1"
    manifest_sha = _write_xgb_checkpoint(
        weights_dir, schema, diseases_trained=["ClassA", "ClassB"]
    )
    (weights_dir / "weights.pkl").unlink()  # remove the checkpoint

    with pytest.raises(FileNotFoundError, match=r"weights\.pkl missing"):
        DDXPlusAdapter.load(
            _spec(),
            {"xgb_test_v1": _model_spec(manifest_sha)},
            device="cpu",
        )


def test_write_manifest_writes_expected_shape(tmp_path: Path) -> None:
    """Guard on _write_manifest output shape — used by adapter tests."""
    # Reuse the training helper; feature-list assertion prevents accidental
    # regressions when someone re-orders the manifest dict.
    schema_evs = [
        {"name": "E_A", "dtype": "B", "values": []},
        {"name": "E_B", "dtype": "B", "values": []},
    ]
    from claritymed.ingest.symptoms.typed_basd import build_layout

    schema = build_layout(schema_evs)
    columns, _labels, _index = feature_columns_from_schema(schema)
    from claritymed.ingest.symptoms.typed_basd import EvalMetrics

    metrics = EvalMetrics(
        IL=3.0,
        ACC=95.0,
        GTPA=90.0,
        DDR=90.0,
        DDP=90.0,
        DDF1=90.0,
        DSR=float("nan"),
        n_severe=0,
    )
    manifest_path = _write_manifest(
        tmp_path,
        model_id="test",
        weights_sha="a" * 64,
        train_params={"seed": 0},
        diseases_trained=["ClassA", "ClassB"],
        columns=columns,
        feature_importance_top_k=[],
        metrics=metrics,
        maxstep=3,
    )
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text())
    assert payload["algorithm_module"] == "xgb"
    assert payload["feature_columns"] == columns
