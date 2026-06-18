"""Smoke tests for :mod:`claritymed.ingest.vision.yolo_forge.spec`.

The spec types are frozen dataclasses with one small invariant
(:meth:`DetectionSplits.assert_non_empty`) and a derived property
(:attr:`DetectionDatasetSpec.num_classes`). Both get covered here so
a future refactor that drops them surfaces a failing test rather than
a silent regression.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from claritymed.ingest.vision.yolo_forge.spec import (
    DetectionDatasetSpec,
    DetectionSplits,
    YoloModelSpec,
    YoloTrainHparams,
)


def _dummy_spec() -> DetectionDatasetSpec:
    return DetectionDatasetSpec(
        dataset_id="fake",
        disease_id="fake_disease",
        class_names=("a", "b", "c"),
        prepare_fn=lambda: DetectionSplits(
            data_yaml_path=Path("/nonexistent.yaml"),
            train_count=1,
            val_count=1,
            test_count=1,
        ),
    )


def test_detection_dataset_spec_num_classes() -> None:
    spec = _dummy_spec()
    assert spec.num_classes == 3


@pytest.mark.parametrize("empty_split", ["train", "val", "test"])
def test_detection_splits_assert_non_empty_raises_on_zero(empty_split: str) -> None:
    counts = {"train_count": 5, "val_count": 5, "test_count": 5}
    counts[f"{empty_split}_count"] = 0
    splits = DetectionSplits(data_yaml_path=Path("/tmp/x.yaml"), **counts)
    with pytest.raises(RuntimeError, match=f"split {empty_split!r} is empty"):
        splits.assert_non_empty()


def test_detection_splits_assert_non_empty_happy_path() -> None:
    splits = DetectionSplits(
        data_yaml_path=Path("/tmp/x.yaml"),
        train_count=1,
        val_count=1,
        test_count=1,
    )
    splits.assert_non_empty()  # no raise


def test_yolo_model_spec_default_train_hparams() -> None:
    """Defaults stay frozen — bumping them is opt-in per spec, not implicit."""
    spec = YoloModelSpec(
        dataset=_dummy_spec(),
        model_id="m",
        model_version="v1",
        base_weights="yolov8n.pt",
    )
    assert isinstance(spec.train_hparams, YoloTrainHparams)
    assert spec.train_hparams.epochs == 50
    assert spec.train_hparams.imgsz == 640
    assert spec.train_hparams.optimizer == "SGD"
    # No HPO unless the spec author opts in.
    assert spec.hparam_space == {}
    assert spec.inference_space == {}
    assert spec.eval_thresholds == {}


def test_yolo_model_spec_suggest_hparams_walks_search_space() -> None:
    """suggest_hparams must invoke each SearchSpace's suggest(trial, name)."""
    from claritymed.ingest.vision.yolo_forge.spec import LogUniform, Uniform

    spec = YoloModelSpec(
        dataset=_dummy_spec(),
        model_id="m",
        model_version="v1",
        base_weights="yolov8n.pt",
        hparam_space={
            "lr0": LogUniform(1e-4, 1e-1),
            "momentum": Uniform(0.8, 0.99),
        },
        inference_space={"conf": Uniform(0.1, 0.5)},
    )

    class StubTrial:
        def __init__(self) -> None:
            self.calls: list[tuple[str, type]] = []

        def suggest_float(self, name, low, high, log=False):
            self.calls.append((name, "float"))
            return (low + high) / 2.0

    trial = StubTrial()
    out = spec.suggest_hparams(trial)
    assert set(out) == {"lr0", "momentum"}
    inf_out = spec.suggest_inference_params(trial)
    assert set(inf_out) == {"conf"}
