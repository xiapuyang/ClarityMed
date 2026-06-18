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


# --- mlflow key sanitiser ----------------------------------------------


def test_sanitize_mlflow_key_strips_ultralytics_paren_suffix() -> None:
    """``metrics/precision(B)`` → ``metrics/precisionB`` (parens dropped)."""
    assert framework._sanitize_mlflow_key("metrics/precision(B)") == (
        "metrics/precisionB"
    )
    assert framework._sanitize_mlflow_key("metrics/mAP50-95(B)") == (
        "metrics/mAP50-95B"
    )


def test_sanitize_mlflow_key_preserves_allowed_chars() -> None:
    """Allowed MLflow chars (alnum + ``_-./: ``) pass through untouched."""
    assert framework._sanitize_mlflow_key("train/box_loss") == "train/box_loss"
    assert framework._sanitize_mlflow_key("lr/pg0") == "lr/pg0"
    assert framework._sanitize_mlflow_key("a.b-c_d:e f/g") == "a.b-c_d:e f/g"


# --- live per-epoch callback --------------------------------------------


def test_mlflow_epoch_callback_logs_trainer_metrics_live(monkeypatch) -> None:
    """Callback must forward ``trainer.metrics`` to ``log_metrics`` each epoch."""

    class _StubModel:
        def __init__(self):
            self.registered: dict[str, object] = {}

        def add_callback(self, event, fn):
            self.registered[event] = fn

    class _StubTrainer:
        def __init__(self, epoch, metrics):
            self.epoch = epoch
            self.metrics = metrics

    calls: list[tuple[dict, int | None]] = []
    monkeypatch.setattr(
        framework, "log_metrics", lambda m, step=None: calls.append((dict(m), step))
    )

    model = _StubModel()
    framework._attach_mlflow_epoch_callback(model)
    assert "on_fit_epoch_end" in model.registered

    callback = model.registered["on_fit_epoch_end"]
    callback(
        _StubTrainer(epoch=0, metrics={"metrics/mAP50(B)": 0.42, "train/box_loss": 1.3})
    )
    callback(
        _StubTrainer(epoch=1, metrics={"metrics/mAP50(B)": 0.51, "train/box_loss": 1.0})
    )

    # Epoch 0 → step 1 (Ultralytics 1-indexes results.csv).
    assert calls[0][1] == 1
    # Parentheses stripped: ``metrics/mAP50(B)`` → ``metrics/mAP50B``.
    # MLflow rejects ``()`` and rejects the whole batch on one bad key,
    # so the sanitiser is the difference between "everything logged"
    # and "epoch silently dropped".
    assert calls[0][0]["train/metrics/mAP50B"] == 0.42
    assert "train/metrics/mAP50(B)" not in calls[0][0]
    assert calls[1][1] == 2
    assert calls[1][0]["train/train/box_loss"] == 1.0


def test_mlflow_epoch_callback_swallows_exceptions(monkeypatch) -> None:
    """A callback bug must never crash the multi-hour training run."""

    class _StubModel:
        def __init__(self):
            self.fn = None

        def add_callback(self, event, fn):
            self.fn = fn

    def _boom(*_a, **_kw):
        raise RuntimeError("mlflow exploded")

    monkeypatch.setattr(framework, "log_metrics", _boom)

    model = _StubModel()
    framework._attach_mlflow_epoch_callback(model)

    class _StubTrainer:
        epoch = 0
        metrics = {"metrics/mAP50(B)": 0.5}

    # Must not raise — Ultralytics' on-disk results.csv stays as the
    # authoritative metric log when MLflow logging glitches.
    model.fn(_StubTrainer())


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
