"""Per-source uncertainty primitive (architecture §8).

``UncertaintyResult`` is the load-bearing data structure of the project: every
pipeline edge — retrieval, lab interpretation, vision prediction, fusion —
emits one tagged with its *type* (aleatoric vs epistemic) and *source*. The
policy layer reads these to decide whether to ask for more input, hand off to
a human, or answer.

The paper's central claim (G4) is *structured uncertainty propagation*, so we
refuse to collapse multi-source uncertainty into a single scalar at this layer.
Fusion lives in ``core/uncertainty/fusion.py`` (later plan), not here.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

UncertaintyType = Literal["aleatoric", "epistemic", "mixed"]
UncertaintySource = Literal["self_report", "measured", "retrieval", "vision", "fusion"]
UncertaintyLevel = Literal["low", "medium", "high"]


class UncertaintyResult(BaseModel):
    """One uncertainty observation tagged by type and source.

    ``level`` is the patient-facing tier ("calibrated band"); ``score`` is the
    optional continuous value used for ECE / Brier reporting. ``reasons`` must
    contain at least one human-readable explanation — refusing empty reasons is
    how we avoid a confident-sounding "high" without justification.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: UncertaintyType
    source: UncertaintySource
    level: UncertaintyLevel
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    reasons: list[str] = Field(min_length=1)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def fuse(cls, others: list["UncertaintyResult"]) -> "UncertaintyResult":
        """Combine multiple source-tagged results into a single answer-level one.

        Not implemented in the foundation plan — the fusion algorithm (late
        fusion with reliability weighting, "contradictions raise uncertainty,
        weakest link caps") is its own plan. This stub exists so downstream
        code can wire the call site early.
        """
        raise NotImplementedError("fusion algorithm lives in a separate plan")
