"""Framework-level tests for :mod:`yolo_forge.framework`.

These cover everything that doesn't need a real Ultralytics import —
the pure helpers (``_check_thresholds`` / ``_regression_check``), the
``run_prepare`` plumbing (which just calls a spec callable), and the
``run_search`` no-HPO fallback. A real train/eval/deploy smoke
exercising Ultralytics is out of scope for CI; run it manually with
``uv sync --extra yolo-forge`` then invoke the CLI's ``pipeline``
subcommand with ``--quick --skip-search``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from claritymed.ingest.vision.yolo_forge import common, framework
from claritymed.ingest.vision.yolo_forge.spec import (
    DetectionDatasetSpec,
    DetectionSplits,
    YoloModelSpec,
    YoloTrainHparams,
)


def _stub_spec(*, eval_thresholds=None) -> YoloModelSpec:
    ds = DetectionDatasetSpec(
        dataset_id="stub_ds",
        disease_id="stub",
        class_names=("c",),
        prepare_fn=lambda: DetectionSplits(
            data_yaml_path=Path("/tmp/data.yaml"),
            train_count=10,
            val_count=2,
            test_count=2,
        ),
    )
    return YoloModelSpec(
        dataset=ds,
        model_id="stub_model_v1",
        model_version="v1",
        base_weights="yolov8n.pt",
        eval_thresholds=eval_thresholds or {},
    )


# --- prepare -----------------------------------------------------------


def test_run_prepare_invokes_dataset_prepare_fn_and_returns_splits() -> None:
    spec = _stub_spec()
    out = framework.run_prepare(spec)
    assert isinstance(out, DetectionSplits)
    assert out.train_count == 10


def test_run_prepare_raises_when_split_is_empty(tmp_path: Path) -> None:
    ds = DetectionDatasetSpec(
        dataset_id="broken",
        disease_id="x",
        class_names=("c",),
        prepare_fn=lambda: DetectionSplits(
            data_yaml_path=tmp_path / "data.yaml",
            train_count=0,  # broken upstream → must fail loud
            val_count=1,
            test_count=1,
        ),
    )
    spec = YoloModelSpec(
        dataset=ds,
        model_id="m",
        model_version="v1",
        base_weights="yolov8n.pt",
    )
    with pytest.raises(RuntimeError, match="empty"):
        framework.run_prepare(spec)


# --- search no-op fallback ---------------------------------------------


def test_run_search_no_hparam_space_returns_spec_defaults() -> None:
    """Empty hparam_space ⇒ skip Optuna entirely, return spec defaults."""
    spec = _stub_spec()  # default empty hparam_space
    splits = spec.dataset.prepare_fn()
    out = framework.run_search(spec, splits)
    # Defaults round-trip through asdict — comparing dicts directly is
    # fine since YoloTrainHparams is a frozen dataclass.
    assert out == asdict(YoloTrainHparams())


# --- device resolution -------------------------------------------------


def test_resolve_device_passes_explicit_value_through() -> None:
    """A pinned device wins over auto-detect, even if MPS is available."""
    assert framework._resolve_device("cpu") == "cpu"
    assert framework._resolve_device("cuda:0") == "cuda:0"


def test_resolve_device_prefers_mps_over_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Apple silicon we want MPS, not Ultralytics' CPU default."""
    import torch

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert framework._resolve_device(None) == "mps"


def test_resolve_device_prefers_cuda_when_no_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert framework._resolve_device(None) == "cuda"


def test_resolve_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert framework._resolve_device(None) == "cpu"


# --- threshold + regression gate helpers -------------------------------


def test_check_thresholds_passes_when_all_met() -> None:
    failures = framework._check_thresholds(
        {"mAP50": 0.50, "image_recall": 0.85},
        {"mAP50": 0.30, "image_recall": 0.80},
    )
    assert failures == []


def test_check_thresholds_reports_each_failure() -> None:
    failures = framework._check_thresholds(
        {"mAP50": 0.20, "image_recall": 0.60},
        {"mAP50": 0.30, "image_recall": 0.80},
    )
    assert sorted(f[0] for f in failures) == ["image_recall", "mAP50"]


def test_check_thresholds_unknown_metric_fails_loud() -> None:
    with pytest.raises(SystemExit, match="unknown metric"):
        framework._check_thresholds({"mAP50": 0.5}, {"bogus": 0.1})


def test_regression_check_no_history_returns_empty(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(common._cfg, "CLARITYMED_HOME", tmp_path)
    spec = _stub_spec(eval_thresholds={"mAP50": 0.3})
    assert framework._regression_check(spec, {"mAP50": 0.5}) == []


def test_regression_check_flags_one_percent_drop(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(common._cfg, "CLARITYMED_HOME", tmp_path)
    spec = _stub_spec(eval_thresholds={"mAP50": 0.3, "image_recall": 0.8})
    log = common.latest_jsonl_path(spec.dataset.dataset_id)
    common.append_entry(
        log,
        {
            "model_id": spec.model_id,
            "metrics": {"mAP50": 0.50, "image_recall": 0.85},
        },
    )
    failures = framework._regression_check(
        spec,
        {"mAP50": 0.40, "image_recall": 0.85},  # mAP50 dropped 10pp
    )
    assert [f[0] for f in failures] == ["mAP50"]


# --- deploy failure modes ----------------------------------------------


def test_run_deploy_missing_eval_metrics_fails_loud(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(common._cfg, "CLARITYMED_HOME", tmp_path)
    spec = _stub_spec()
    staging = common.staging_dir(
        dataset_id=spec.dataset.dataset_id, model_id=spec.model_id
    )
    staging.mkdir(parents=True)
    with pytest.raises(SystemExit, match="no eval_metrics.json"):
        framework.run_deploy(spec, staging=staging)


def test_run_deploy_threshold_failure_blocks_promotion(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(common._cfg, "CLARITYMED_HOME", tmp_path)
    spec = _stub_spec(eval_thresholds={"mAP50": 0.5})
    staging = common.staging_dir(
        dataset_id=spec.dataset.dataset_id, model_id=spec.model_id
    )
    staging.mkdir(parents=True)
    (staging / "eval_metrics.json").write_text(json.dumps({"metrics": {"mAP50": 0.20}}))
    with pytest.raises(SystemExit, match="deploy gate failed"):
        framework.run_deploy(spec, staging=staging)


def test_run_deploy_promotes_weights_and_appends_audit(
    monkeypatch, tmp_path: Path
) -> None:
    """Happy path: thresholds met → weights copied + LATEST.jsonl appended."""
    monkeypatch.setattr(common._cfg, "CLARITYMED_HOME", tmp_path)
    spec = _stub_spec(eval_thresholds={"mAP50": 0.3})
    staging = common.staging_dir(
        dataset_id=spec.dataset.dataset_id, model_id=spec.model_id
    )
    (staging / "train" / "weights").mkdir(parents=True)
    (staging / "train" / "weights" / "best.pt").write_bytes(b"\x00fake-weights")
    (staging / "eval_metrics.json").write_text(
        json.dumps(
            {
                "metrics": {"mAP50": 0.50, "image_recall": 0.9},
                "tuned_inference_params": {"conf": 0.27, "iou": 0.5},
            }
        )
    )

    entry = framework.run_deploy(spec, staging=staging)
    promoted = (
        common.disease_root(spec.dataset.dataset_id)
        / f"{spec.model_id}_{spec.model_version}.pt"
    )
    assert promoted.is_file()
    assert entry["weights_path"] == str(promoted)
    assert entry["tuned_inference_params"] == {"conf": 0.27, "iou": 0.5}
    log = common.latest_jsonl_path(spec.dataset.dataset_id)
    last = common.read_last_entry(log, model_id=spec.model_id)
    assert last["metrics"]["mAP50"] == 0.50


# --- ultralytics smoke (skipped unless extra installed) ----------------


@pytest.mark.skipif(
    not pytest.importorskip("ultralytics", reason="yolo-forge extra not installed"),
    reason="",
)
def test_ultralytics_yolo_class_importable() -> None:
    """Lightweight check that the extra wires up — full train is operator-driven.

    Heavier smoke (1-epoch train against the synthetic RSNA fixture)
    is intentionally not in CI: ultralytics pulls ~600MB and the
    weights download adds another ~6MB on first run, which makes the
    test brittle in offline / CI environments.
    """
    from ultralytics import YOLO  # noqa: F401
