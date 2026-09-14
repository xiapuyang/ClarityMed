"""Tests for the cross-dataset drift bench driver.

Component-level tests cover the bits we can isolate from the heavy
torch + dataset path:

* Registry filtering + dotted-path resolution
* Ground-truth label-space remap
* Manifest tuned-threshold extraction
* Softmax with temperature
* Output cell serialization
* CLI dry-run smoke

The end-to-end model-load + forward integration is intentionally
deferred to Unit 3's real-artifact run — synthesizing a torch ResNet/
EfficientNet fixture would add several hundred lines of test infra for
diminishing return, since the actual bench is hand-validated on real
breast US artifacts before commit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.benchmarks.cross_dataset_drift import registry, run as bench


# --- registry ------------------------------------------------------------


def test_entries_for_pair_returns_matching_entries() -> None:
    """``breast_us`` should resolve to the 4 entries defined in the registry."""
    entries = registry.entries_for_pair("breast_us")
    assert len(entries) == 4
    assert all(e.pair_id == "breast_us" for e in entries)
    # Two self-eval (busi→busi, kaggle→kaggle) + two cross (busi→kaggle, kaggle→busi).
    train_eval_pairs = {(e.train_dataset_id, e.eval_dataset_id) for e in entries}
    assert ("busi", "busi") in train_eval_pairs
    assert ("breast_us_kaggle", "breast_us_kaggle") in train_eval_pairs
    assert ("busi", "breast_us_kaggle") in train_eval_pairs
    assert ("breast_us_kaggle", "busi") in train_eval_pairs


def test_entries_for_pair_unknown_raises_with_known_list() -> None:
    """Typo'd pair id should surface the known pair ids — operators are
    almost always one underscore vs hyphen away from the right answer.
    """
    with pytest.raises(ValueError, match="breast_us"):
        registry.entries_for_pair("breast-us")


# --- dataset spec dotted-path resolution --------------------------------


def test_resolve_dataset_spec_happy_path() -> None:
    """Valid ``module:ATTR`` returns the spec instance."""
    spec = bench._resolve_dataset_spec(
        "claritymed.ingest.vision.busi.dataset_spec:BUSI_DATASET"
    )
    assert spec.disease_id == "breast_cancer_ultrasound"
    assert spec.accepted_modality == "ultrasound"


def test_resolve_dataset_spec_missing_colon_fails_loud() -> None:
    """Forgot the colon — fail before importlib runs."""
    with pytest.raises(SystemExit, match="':'"):
        bench._resolve_dataset_spec("claritymed.ingest.vision.busi.dataset_spec")


def test_resolve_dataset_spec_bad_module_fails_loud() -> None:
    """Importable namespace check — typo in module path."""
    with pytest.raises(SystemExit, match="failed to import"):
        bench._resolve_dataset_spec("claritymed.does.not.exist:THING")


def test_resolve_dataset_spec_missing_attr_fails_loud() -> None:
    """Module exists but doesn't export the named attribute."""
    with pytest.raises(SystemExit, match="no attribute"):
        bench._resolve_dataset_spec(
            "claritymed.ingest.vision.busi.dataset_spec:NOT_AN_EXPORT"
        )


# --- artifact dir resolution --------------------------------------------


def test_resolve_artifact_dir_missing_dir_fails_loud(tmp_path: Path) -> None:
    """Pointing at a nonexistent dir fails with a clear message."""
    with pytest.raises(SystemExit, match="not found"):
        bench._resolve_artifact_dir("does/not/exist", vision_root=tmp_path)


def test_resolve_artifact_dir_missing_manifest_fails_loud(tmp_path: Path) -> None:
    """Dir exists but no manifest.json — caught before torch.load is attempted."""
    (tmp_path / "fake_model").mkdir()
    (tmp_path / "fake_model" / "weights.pt").write_text("not real weights")
    with pytest.raises(SystemExit, match="missing manifest.json"):
        bench._resolve_artifact_dir("fake_model", vision_root=tmp_path)


def test_resolve_artifact_dir_missing_weights_fails_loud(tmp_path: Path) -> None:
    """Dir + manifest present but no weights.pt — caught early."""
    (tmp_path / "fake_model").mkdir()
    (tmp_path / "fake_model" / "manifest.json").write_text("{}")
    with pytest.raises(SystemExit, match="missing weights.pt"):
        bench._resolve_artifact_dir("fake_model", vision_root=tmp_path)


# --- gt label-space remap -----------------------------------------------


def test_remap_gt_no_fallback_when_eval_labels_subset_of_model_labels() -> None:
    """breast_us_kaggle (2-class) eval set being scored against BUSI model
    (3-class): every eval label exists in model vocab → no fallback notes.
    """
    remapped, notes = bench._remap_gt_to_model_space(
        eval_labels=("benign", "malignant"),
        model_labels=("benign", "malignant", "normal"),
        gt_eval_space=[0, 1, 0, 1, 0],
        positive_labels=frozenset({"malignant"}),
    )
    # benign and malignant are at the same indices in both spaces here.
    assert remapped == [0, 1, 0, 1, 0]
    assert notes == ""


def test_remap_gt_folds_eval_only_labels_into_fallback_negative() -> None:
    """BUSI (3-class) eval set scored against breast_us_kaggle (2-class)
    model: BUSI's ``normal`` is missing from model vocab — must fall
    into a non-positive model label.
    """
    remapped, notes = bench._remap_gt_to_model_space(
        eval_labels=("benign", "malignant", "normal"),
        model_labels=("benign", "malignant"),
        gt_eval_space=[0, 1, 2, 2, 0],
        positive_labels=frozenset({"malignant"}),
    )
    # ``normal`` (idx 2 in eval) → fallback negative = ``benign`` (idx 0 in model).
    assert remapped == [0, 1, 0, 0, 0]
    assert "normal" in notes
    assert "benign" in notes
    assert "2 sample(s)" in notes


def test_remap_gt_raises_when_no_negative_label_in_model() -> None:
    """Pathological case: model_labels = positive_labels means there's
    no negative class for binary collapse. Fail loud — silently
    treating everything as positive would produce sensitivity=1 with
    zero meaning.
    """
    with pytest.raises(SystemExit, match="no non-positive label"):
        bench._remap_gt_to_model_space(
            eval_labels=("malignant",),
            model_labels=("malignant",),
            gt_eval_space=[0],
            positive_labels=frozenset({"malignant"}),
        )


# --- tuned threshold extraction -----------------------------------------


def _minimal_manifest(**overrides):
    """Build a minimal valid Manifest for threshold tests."""
    from claritymed.core.vision.schemas import Manifest

    payload = {
        "model_id": "test_model",
        "model_version": "v1.0.0",
        "framework": "pytorch",
        "accepted_modality": "ultrasound",
        "sha256_weights": "a" * 64,
        "task": "classification",
        "labels": ["benign", "malignant"],
        "labels_meta": {
            "benign": {"description": "Benign lesion."},
            "malignant": {"description": "Malignant lesion."},
        },
        "backbone": "resnet50",
        "tuned_inference": None,
    }
    payload.update(overrides)
    return Manifest.model_validate(payload)


def test_resolve_tuned_threshold_none_when_no_tuned_inference() -> None:
    """Untuned manifest → None (bench falls back to natural threshold)."""
    manifest = _minimal_manifest()
    assert (
        bench._resolve_tuned_threshold(
            manifest=manifest, positive_labels=frozenset({"malignant"})
        )
        is None
    )


def test_resolve_tuned_threshold_none_when_positive_label_not_tuned() -> None:
    """Manifest has thresholds but not for the requested positive label."""
    manifest = _minimal_manifest(
        tuned_inference={
            "classification_thresholds": {"benign": 0.5},
        }
    )
    assert (
        bench._resolve_tuned_threshold(
            manifest=manifest, positive_labels=frozenset({"malignant"})
        )
        is None
    )


def test_resolve_tuned_threshold_returns_label_threshold() -> None:
    """Happy path: single positive label, threshold present."""
    manifest = _minimal_manifest(
        tuned_inference={
            "classification_thresholds": {"malignant": 0.42},
        }
    )
    assert (
        bench._resolve_tuned_threshold(
            manifest=manifest, positive_labels=frozenset({"malignant"})
        )
        == 0.42
    )


def test_resolve_tuned_threshold_takes_min_for_multi_positive() -> None:
    """Multi-positive collapse uses the loosest threshold across the
    positive labels — sum-of-probs ≥ threshold makes most sense at the
    smallest cutoff.
    """
    manifest = _minimal_manifest(
        labels=["benign", "malignant", "suspicious"],
        labels_meta={
            "benign": {"description": "Benign lesion."},
            "malignant": {"description": "Malignant lesion."},
            "suspicious": {"description": "Suspicious lesion."},
        },
        tuned_inference={
            "classification_thresholds": {"malignant": 0.6, "suspicious": 0.4},
        },
    )
    assert (
        bench._resolve_tuned_threshold(
            manifest=manifest,
            positive_labels=frozenset({"malignant", "suspicious"}),
        )
        == 0.4
    )


# --- softmax ------------------------------------------------------------


def test_softmax_unit_temperature_sums_to_one() -> None:
    """Default temperature=1.0 produces standard softmax."""
    logits = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]], dtype=np.float64)
    probs = bench._softmax(logits, temperature=1.0)
    np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], atol=1e-9)
    assert probs[0, 2] > probs[0, 1] > probs[0, 0]


def test_softmax_high_temperature_flattens() -> None:
    """High temperature pushes probabilities toward uniform."""
    logits = np.array([[10.0, 0.0]], dtype=np.float64)
    sharp = bench._softmax(logits, temperature=0.5)
    flat = bench._softmax(logits, temperature=5.0)
    # Sharp should be more confident on class 0 than flat.
    assert sharp[0, 0] > flat[0, 0]
    # Flat should still favor class 0 but less extremely.
    assert flat[0, 0] > 0.5


# --- output serialization -----------------------------------------------


def test_bench_cell_to_json_dict_round_trip() -> None:
    """``BenchCell.to_json_dict`` must produce JSON-serializable values
    (no frozensets, no pydantic models) and preserve key fields.
    """
    import json

    from claritymed.core.vision.eval_metrics import BinaryMetrics

    cell = bench.BenchCell(
        pair_id="breast_us",
        model_id="breast_busi_unet_v1",
        model_artifact_dir="breast_cancer_ultrasound/breast_busi_unet_v1",
        train_dataset_id="busi",
        eval_dataset_id="busi",
        threshold_kind="natural",
        threshold_value=None,
        positive_labels=("malignant",),
        metrics=BinaryMetrics(
            sensitivity=0.85,
            specificity=0.92,
            accuracy=0.89,
            auc=0.94,
            n_total=100,
            n_positive=30,
        ),
        notes="",
    )
    payload = cell.to_json_dict()
    # Round-trip through json.dumps to verify serializability.
    assert json.loads(json.dumps(payload))["sensitivity"] == 0.85
    assert payload["positive_labels"] == ["malignant"]
    assert payload["threshold_value"] is None


# --- dry-run smoke ------------------------------------------------------


def test_dry_run_writes_skeleton_without_loading_models(tmp_path: Path) -> None:
    """``--dry-run`` exits 0 and writes an empty cells list — no torch
    import in the critical path, no artifact resolution required.
    """
    exit_code = bench.main(
        [
            "--pair",
            "breast_us",
            "--dry-run",
            "--vision-root",
            str(tmp_path / "vision"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert exit_code == 0
    # Dry-run writes a ``*.dry_run.json`` companion to make it obvious the
    # output is not real bench data.
    dry_run_files = list((tmp_path / "out").glob("*.dry_run.json"))
    assert len(dry_run_files) == 1
    import json

    assert json.loads(dry_run_files[0].read_text()) == []


# --- markdown rendering -------------------------------------------------


def test_markdown_renders_n_a_for_none_metrics(tmp_path: Path) -> None:
    """Degenerate cells (None specificity/AUC) render as ``N/A``, not 0.0."""
    from claritymed.core.vision.eval_metrics import BinaryMetrics

    cell = bench.BenchCell(
        pair_id="test",
        model_id="m1",
        model_artifact_dir="x",
        train_dataset_id="d1",
        eval_dataset_id="d1",
        threshold_kind="natural",
        threshold_value=None,
        positive_labels=("malignant",),
        metrics=BinaryMetrics(
            sensitivity=1.0,
            specificity=None,
            accuracy=1.0,
            auc=None,
            n_total=10,
            n_positive=10,
        ),
        notes="",
    )
    out = tmp_path / "x.md"
    bench._write_markdown([cell], out, pair_id="test")
    body = out.read_text()
    assert "N/A" in body
    assert "100.0%" in body  # sensitivity rendered as percent
