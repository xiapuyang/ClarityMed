"""MLflow integration tests for yolo_forge.

Two contracts:

1. **mlflow missing at execution → fail loud.** ``mlflow_phase_run``
   raises ``SystemExit`` with a remediation hint when ``import mlflow``
   fails inside the context. Silent no-op would let training appear
   to succeed with zero metrics persisted; we want the operator to
   know immediately.
2. **mlflow present** — runs land in the right experiment with the
   right tags so the UI can pivot across pipelines.

The tests stub ``_mlflow_run`` (the underlying contextmanager) to
avoid hitting the real SQLite tracking DB during CI.
"""

from __future__ import annotations

import csv
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from claritymed.ingest.vision.yolo_forge import common, framework
from claritymed.ingest.vision.yolo_forge.spec import (
    DetectionDatasetSpec,
    DetectionSplits,
    YoloModelSpec,
)


def _stub_spec() -> YoloModelSpec:
    ds = DetectionDatasetSpec(
        dataset_id="stub_detection",
        disease_id="stub_disease",
        class_names=("c",),
        prepare_fn=lambda: DetectionSplits(
            data_yaml_path=Path("/tmp/x.yaml"),
            train_count=1,
            val_count=1,
            test_count=1,
        ),
    )
    return YoloModelSpec(
        dataset=ds,
        model_id="m_v1",
        model_version="v1",
        base_weights="yolov8n.pt",
    )


# --- mlflow missing → fail loud at execution -----------------------------


def test_mlflow_phase_run_fails_loud_when_mlflow_missing(monkeypatch) -> None:
    """`import mlflow` failing inside the contextmanager → SystemExit with hint."""
    monkeypatch.setitem(sys.modules, "mlflow", None)  # forces ImportError
    spec = _stub_spec()
    with pytest.raises(SystemExit, match="mlflow not installed"):
        with common.mlflow_phase_run(spec=spec, phase="test", task_id="t1"):
            pass


def test_log_metrics_outside_active_run_is_silent_noop(monkeypatch) -> None:
    """forge.log_metrics keeps its try/except no-op for code paths outside a run.

    The pipeline never calls log_metrics outside mlflow_phase_run, so this
    silent fallback is mostly a defense-in-depth detail — but worth pinning
    so the shim contract is explicit.
    """
    monkeypatch.setitem(sys.modules, "mlflow", None)
    common.log_metrics({"any/key": 1.23})  # must not raise


# --- mlflow present → invoked with the right args ------------------------


def test_mlflow_phase_run_invokes_mlflow_run_with_expected_tags(monkeypatch) -> None:
    """The shim must populate `pipeline=yolo_forge` + lineage tags."""
    captured: dict = {}

    @contextmanager
    def fake_mlflow_run(*, feature, dataset_id, run_name, run_type, params, tags):
        captured.update(
            feature=feature,
            dataset_id=dataset_id,
            run_name=run_name,
            run_type=run_type,
            params=params,
            tags=dict(tags) if tags else {},
        )
        yield object()

    monkeypatch.setattr(common, "_mlflow_run", fake_mlflow_run)
    monkeypatch.setitem(sys.modules, "mlflow", object())  # truthy → import succeeds
    spec = _stub_spec()
    with common.mlflow_phase_run(
        spec=spec, phase="train", task_id="abc-123", params={"lr": 0.01}
    ):
        pass

    assert captured["feature"] == "vision"
    # disease_id (not dataset_id) is used for the experiment so forge +
    # yolo_forge share the same MLflow experiment per disease.
    assert captured["dataset_id"] == spec.dataset.disease_id
    assert captured["run_type"] == "train"
    assert captured["params"] == {"lr": 0.01}
    tags = captured["tags"]
    assert tags["pipeline"] == "yolo_forge"
    assert tags["claritymed.task_id"] == "abc-123"
    assert tags["model_id"] == spec.model_id
    assert tags["model_version"] == spec.model_version
    assert tags["dataset_id"] == spec.dataset.dataset_id
    assert tags["task"] == "detection"


# --- results.csv replay -------------------------------------------------


def test_log_results_csv_replays_each_row_as_step(tmp_path: Path, monkeypatch) -> None:
    """`_log_results_csv_to_mlflow` must turn each CSV row into one log_metrics call."""
    csv_path = tmp_path / "results.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train/box_loss", "val/box_loss", "metrics/mAP50"])
        writer.writerow([0, 0.50, 0.45, 0.10])
        writer.writerow([1, 0.30, 0.25, 0.35])

    calls: list[tuple[dict, int | None]] = []

    def fake_log(metrics, step=None):
        calls.append((dict(metrics), step))

    monkeypatch.setattr(framework, "log_metrics", fake_log)
    framework._log_results_csv_to_mlflow(csv_path)

    assert len(calls) == 2
    metrics_0, step_0 = calls[0]
    assert step_0 == 0
    assert metrics_0["train/train/box_loss"] == 0.50
    assert metrics_0["train/metrics/mAP50"] == 0.10
    metrics_1, step_1 = calls[1]
    assert step_1 == 1
    assert metrics_1["train/val/box_loss"] == 0.25


def test_log_results_csv_missing_file_is_noop(tmp_path: Path, monkeypatch) -> None:
    """Silent skip — Ultralytics output layout shouldn't crash the framework."""
    calls: list = []
    monkeypatch.setattr(
        framework, "log_metrics", lambda *a, **kw: calls.append((a, kw))
    )
    framework._log_results_csv_to_mlflow(tmp_path / "does_not_exist.csv")
    assert calls == []


# --- phase functions thread task_id ------------------------------------


def test_run_search_skip_path_does_not_call_mlflow(monkeypatch) -> None:
    """Empty hparam_space short-circuits before opening any mlflow run."""
    spec = _stub_spec()  # default empty hparam_space
    splits = spec.dataset.prepare_fn()
    calls: list = []

    @contextmanager
    def fail_if_called(**kwargs):
        calls.append(kwargs)
        yield None

    monkeypatch.setattr(framework, "mlflow_phase_run", fail_if_called)
    out = framework.run_search(spec, splits)
    # No mlflow_run opened — we bailed before the context.
    assert calls == []
    assert "epochs" in out  # spec defaults returned


def test_run_deploy_threads_task_id_into_audit_entry(
    monkeypatch, tmp_path: Path
) -> None:
    """Deploy entry must carry the task_id we passed in."""
    import json

    monkeypatch.setattr(common._cfg, "CLARITYMED_HOME", tmp_path)
    spec = _stub_spec()
    object.__setattr__(
        spec, "eval_thresholds", {"mAP50": 0.3}
    )  # frozen dataclass workaround
    staging = common.staging_dir(
        dataset_id=spec.dataset.dataset_id, model_id=spec.model_id
    )
    (staging / "train" / "weights").mkdir(parents=True)
    (staging / "train" / "weights" / "best.pt").write_bytes(b"\x00")
    (staging / "eval_metrics.json").write_text(
        json.dumps({"metrics": {"mAP50": 0.50, "image_recall": 0.9}})
    )

    entry = framework.run_deploy(spec, staging=staging, task_id="explicit-task-id")
    assert entry["task_id"] == "explicit-task-id"
    last = common.read_last_entry(
        common.latest_jsonl_path(spec.dataset.dataset_id), model_id=spec.model_id
    )
    assert last["task_id"] == "explicit-task-id"


def test_run_deploy_picks_up_task_id_from_eval_blob(
    monkeypatch, tmp_path: Path
) -> None:
    """When no explicit task_id is passed, use the tune/eval phase's recorded id."""
    import json

    monkeypatch.setattr(common._cfg, "CLARITYMED_HOME", tmp_path)
    spec = _stub_spec()
    object.__setattr__(spec, "eval_thresholds", {"mAP50": 0.3})
    staging = common.staging_dir(
        dataset_id=spec.dataset.dataset_id, model_id=spec.model_id
    )
    (staging / "train" / "weights").mkdir(parents=True)
    (staging / "train" / "weights" / "best.pt").write_bytes(b"\x00")
    (staging / "eval_metrics.json").write_text(
        json.dumps({"metrics": {"mAP50": 0.50}, "task_id": "from-tune-phase"})
    )

    entry = framework.run_deploy(spec, staging=staging)
    assert entry["task_id"] == "from-tune-phase"
