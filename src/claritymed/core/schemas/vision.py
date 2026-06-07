"""Vision tool contract (architecture §7).

``DiseaseVisionModel`` is a structural ``Protocol`` (not an ``ABC``) so a
third-party vision model can satisfy the contract without importing this
package. ``runtime_checkable`` enables ``isinstance`` checks for the
registry's hot-load path.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

Modality = Literal["skin", "chest_xray"]
SkinToneFitzpatrick = Literal["I", "II", "III", "IV", "V", "VI"]


class PredictionSet(BaseModel):
    """Conformal prediction set: classes + their calibrated probabilities."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    classes: list[str] = Field(min_length=1)
    probs: list[float] = Field(min_length=1)
    alpha: float = Field(gt=0.0, lt=1.0)
    set_size: int = Field(ge=0)

    @model_validator(mode="after")
    def _shapes_match(self) -> "PredictionSet":
        if len(self.classes) != len(self.probs):
            raise ValueError(
                f"classes ({len(self.classes)}) and probs "
                f"({len(self.probs)}) must have equal length"
            )
        if self.set_size > len(self.classes):
            raise ValueError(
                f"set_size {self.set_size} > number of classes {len(self.classes)}"
            )
        return self


class QualityReport(BaseModel):
    """Per-image quality gate result. Failures must explain why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    passes: bool
    reasons: list[str] = Field(default_factory=list)
    skin_tone: SkinToneFitzpatrick | None = None

    @model_validator(mode="after")
    def _failure_requires_reason(self) -> "QualityReport":
        if not self.passes and not self.reasons:
            raise ValueError("quality_report.passes=False must list reasons")
        return self


class ModelMetadata(BaseModel):
    """Static description of one registered per-disease vision model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    disease: str = Field(min_length=1)
    modality: Modality
    training_dist: str = Field(min_length=1)
    weights_path: str = Field(min_length=1)
    version: str = Field(min_length=1)
    skin_tone_coverage: dict[SkinToneFitzpatrick, float] | None = None

    @model_validator(mode="after")
    def _skin_modality_should_declare_coverage(self) -> "ModelMetadata":
        if self.modality == "skin" and self.skin_tone_coverage is None:
            warnings.warn(
                f"skin model '{self.disease}' missing skin_tone_coverage — "
                "Fitzpatrick stratification required before clinical use",
                stacklevel=2,
            )
        return self


@runtime_checkable
class DiseaseVisionModel(Protocol):
    """Per-disease vision model contract (architecture §7.1)."""

    metadata: ModelMetadata

    def predict(self, image: Any) -> Any: ...

    def calibrate(self, raw_scores: Any) -> Any: ...

    def conformal_set(self, probs: Any, alpha: float) -> PredictionSet: ...

    def ood_score(self, image: Any) -> float: ...

    def quality_gate(self, image: Any) -> QualityReport: ...
