"""Tests for ``DiseaseVisionModel`` Protocol + ``PredictionSet`` / ``QualityReport``."""

from __future__ import annotations

import warnings

import pytest
from pydantic import ValidationError

from claritymed.core.schemas import (
    DiseaseVisionModel,
    ModelMetadata,
    PredictionSet,
    QualityReport,
)


class _DummyDermModel:
    """Structural impl of ``DiseaseVisionModel`` for ``isinstance`` test."""

    def __init__(self) -> None:
        self.metadata = ModelMetadata(
            disease="rash",
            modality="skin",
            training_dist="Fitzpatrick17k",
            weights_path="rash/v1/",
            version="v1",
            skin_tone_coverage={
                "I": 0.9,
                "II": 0.9,
                "III": 0.8,
                "IV": 0.7,
                "V": 0.5,
                "VI": 0.4,
            },
        )

    def predict(self, image):  # noqa: ARG002
        return [0.6, 0.4]

    def calibrate(self, raw_scores):  # noqa: ARG002
        return [0.55, 0.45]

    def conformal_set(self, probs, alpha):  # noqa: ARG002
        return PredictionSet(
            classes=["eczema", "acne"], probs=[0.55, 0.45], alpha=alpha, set_size=2
        )

    def ood_score(self, image):  # noqa: ARG002
        return 0.1

    def quality_gate(self, image):  # noqa: ARG002
        return QualityReport(passes=True, reasons=[], skin_tone="III")


def test_dummy_satisfies_protocol():
    assert isinstance(_DummyDermModel(), DiseaseVisionModel)


def test_prediction_set_shapes_must_match():
    with pytest.raises(ValidationError):
        PredictionSet(classes=["a", "b"], probs=[0.6], alpha=0.1, set_size=1)


def test_prediction_set_size_capped_by_class_count():
    with pytest.raises(ValidationError):
        PredictionSet(classes=["a", "b"], probs=[0.6, 0.4], alpha=0.1, set_size=3)


def test_quality_report_failure_requires_reason():
    with pytest.raises(ValidationError):
        QualityReport(passes=False, reasons=[])


def test_skin_model_missing_coverage_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ModelMetadata(
            disease="rash",
            modality="skin",
            training_dist="SD-198",
            weights_path="rash/v1/",
            version="v1",
        )
    assert any("skin_tone_coverage" in str(w.message) for w in caught)


def test_chest_xray_model_no_warning_without_coverage():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ModelMetadata(
            disease="pneumonia",
            modality="chest_xray",
            training_dist="ChestX-ray14",
            weights_path="chestxray14/v1/",
            version="v1",
        )
    assert all("skin_tone_coverage" not in str(w.message) for w in caught)
