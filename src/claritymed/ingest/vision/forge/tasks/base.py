"""Abstract :class:`Task` — the polymorphic surface forge dispatches over.

A Task encodes everything that differs between cls / cls+seg / detection
training pipelines:

* Architecture (model factory).
* Loss formula.
* Per-batch / per-epoch evaluation (which metrics matter, how they're
  computed).
* The on-disk cache shape used for the tune phase (cls keeps only
  cls_logits; cls+seg also caches seg probabilities + GT masks).
* Inference-time params surfaced into the manifest.

Per-phase floors + composite weights live on the Task instance so two
diseases sharing a task type can carry different medical thresholds.

Smoke paths
-----------
Forge supports a ``--smoke`` mode that exercises the full pipeline
wiring (manifest write, deploy gate, LATEST.jsonl append) on synthetic
numbers without torch. Each concrete Task implements
:meth:`smoke_breakdown` and :meth:`smoke_tuned_params` so the smoke
breakdowns are *feasible by construction* — the wire test never gets
gated out by a real metric being absent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from claritymed.core.vision.schemas import TunedInferenceParams
from claritymed.ingest.vision.forge.scoring import PhaseFloors

TaskName = Literal["classification", "classification+segmentation", "detection"]
PhaseLabel = Literal["search", "train", "tune"]


@dataclass(frozen=True)
class FloorBundle:
    """The three per-phase floor dicts a Task carries.

    Kept as one frozen struct so a ModelSpec can build it once and
    reuse via :meth:`Task.phase_floors` — avoids a dict-of-dicts hop on
    every score call.
    """

    search: dict[str, float]
    train: dict[str, float]
    deploy: dict[str, float]  # tune phase shares this — last selection step

    def for_phase(self, phase: PhaseLabel) -> PhaseFloors:
        if phase == "search":
            return PhaseFloors(floors=dict(self.search), label="search")
        if phase == "train":
            return PhaseFloors(floors=dict(self.train), label="train")
        if phase == "tune":
            return PhaseFloors(floors=dict(self.deploy), label="tune")
        raise ValueError(f"unknown phase {phase!r}")


class Task(ABC):
    """Abstract base for every task forge knows how to drive.

    Subclasses set the configuration fields in ``__init__`` and
    implement the abstract behaviour methods. The framework reaches
    these via duck-typed attribute access — the ABC just pins the
    contract.
    """

    # --- configuration (set by subclass __init__) ------------------------

    name: TaskName
    critical_labels: tuple[str, ...]
    composite_weights: dict[str, float]
    floors: FloorBundle
    supports_tta: bool = True
    supports_saliency: bool = False

    # --- floor accessors -------------------------------------------------

    def phase_floors(self, phase: PhaseLabel) -> PhaseFloors:
        """Return :class:`PhaseFloors` for the given phase."""
        return self.floors.for_phase(phase)

    def deploy_floors_map(self) -> dict[str, float]:
        """Return the deploy-phase floor map (used by the deploy gate)."""
        return dict(self.floors.deploy)

    # --- spec validation -----------------------------------------------

    @property
    @abstractmethod
    def breakdown_metric_keys(self) -> frozenset[str]:
        """Metric keys this task's ``evaluate`` surfaces in the breakdown.

        Excludes the derived ``composite`` field — that's computed by
        the task itself from these keys, so referencing it from
        ``floors`` or ``composite_weights`` would be a self-reference.
        ``_validate_metric_keys`` uses this set to fail-fast on a typo
        in any ``ModelSpec``.
        """

    def _validate_metric_keys(self) -> None:
        """Raise if ``floors`` or ``composite_weights`` reference unknown keys.

        Called from concrete-task ``__init__`` so a typo (``"recal"``
        instead of ``"recall"``, ``"accuray"`` instead of ``"accuracy"``)
        crashes the spec module at import time. Without this check,
        ``feasibility_aware_score`` would silently treat the missing
        key as ``0.0`` — infeasible-forever for a bad floor, or zero
        contribution for a bad composite weight — both invisible.
        """
        allowed = self.breakdown_metric_keys
        for phase in ("search", "train", "deploy"):
            phase_floors = getattr(self.floors, phase)
            unknown = sorted(set(phase_floors) - allowed)
            if unknown:
                raise ValueError(
                    f"{type(self).__name__}: floors.{phase} references "
                    f"unknown metric(s) {unknown} not in breakdown shape "
                    f"{sorted(allowed)}. Likely a typo in the ModelSpec."
                )
        unknown = sorted(set(self.composite_weights) - allowed)
        if unknown:
            raise ValueError(
                f"{type(self).__name__}: composite_weights references "
                f"unknown metric(s) {unknown} not in breakdown shape "
                f"{sorted(allowed)}. Likely a typo in the ModelSpec."
            )

    # --- architecture + batch interface ----------------------------------

    @abstractmethod
    def build_model(self, *, backbone: str, num_classes: int, pretrained: bool) -> Any:
        """Construct the model. Returns a ``torch.nn.Module``.

        ``pretrained=True`` should be passed only from real training
        (not smoke or inference) so the encoder loads ImageNet weights
        on first run.
        """

    @abstractmethod
    def unpack_batch(self, batch) -> tuple[Any, dict[str, Any]]:
        """Split a DataLoader batch into ``(inputs, targets_dict)``.

        cls: returns ``(imgs, {"labels": labels})``.
        cls+seg: returns ``(imgs, {"labels": labels, "masks": masks})``.

        Framework code stays task-agnostic — it forwards ``inputs``
        through the model and passes ``targets_dict`` to
        :meth:`compute_loss` / :meth:`evaluate_outputs`.
        """

    @abstractmethod
    def compute_loss(self, outputs, targets: dict[str, Any], hp: dict[str, Any]) -> Any:
        """Return a scalar torch loss tensor.

        ``hp`` is the per-trial hyperparameter dict — Tasks pull
        loss-side knobs (e.g. ``seg_loss_weight``) directly from it.
        """

    # --- evaluation ------------------------------------------------------

    @abstractmethod
    def evaluate(
        self,
        model,
        loader,
        device,
        *,
        labels: tuple[str, ...],
        hp: dict[str, Any],
    ) -> tuple[float, dict[str, float]]:
        """Compute mean eval loss + the metric breakdown over ``loader``.

        The breakdown dict must include every key referenced by
        :attr:`composite_weights` and :attr:`floors`, plus a
        ``"composite"`` entry (for human readability in curves +
        eval_metrics.json).
        """

    # --- tune phase: cache + cache-eval ----------------------------------

    @abstractmethod
    def cache_outputs(
        self,
        model,
        loader,
        device,
    ):
        """Run the model once over ``loader`` and cache outputs.

        Returns an opaque, task-specific cache object that
        :meth:`evaluate_cache` consumes. Cls task: caches plain + TTA
        cls_logits and labels. Cls+seg task: additionally caches seg
        probabilities + GT masks.
        """

    @abstractmethod
    def evaluate_cache(
        self,
        cache,
        params: dict[str, Any],
        *,
        labels: tuple[str, ...],
    ) -> dict[str, float]:
        """Evaluate one inference-param set against the cached outputs.

        Pure numpy / no model forward — drives the cheap inner loop of
        the tune phase's Optuna study.
        """

    # --- tune output ---------------------------------------------------

    @abstractmethod
    def build_tuned_inference_block(
        self,
        params: dict[str, Any],
        *,
        labels: tuple[str, ...],
    ) -> TunedInferenceParams:
        """Translate raw tune params into a manifest ``TunedInferenceParams``.

        Handles ``classification_thresholds`` per-label expansion
        (e.g. broadcast a single ``cancer_threshold`` over multiple
        critical labels) so callers never construct the schema type
        directly.
        """

    # --- smoke paths ---------------------------------------------------

    @abstractmethod
    def smoke_breakdown(self) -> dict[str, float]:
        """Synthetic feasible breakdown for ``--smoke`` runs.

        Must clear every deploy floor by construction so the smoke
        pipeline's deploy gate accepts it without depending on a real
        forward pass.
        """

    @abstractmethod
    def smoke_tuned_params(self) -> dict[str, Any]:
        """Synthetic tune params for ``--smoke`` runs."""


__all__ = ["FloorBundle", "PhaseLabel", "Task", "TaskName"]
