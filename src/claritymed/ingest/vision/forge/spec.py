"""Forge-side specs: how to describe a (dataset, model-variant) pair.

Two spec types compose:

* :class:`DatasetSpec` — everything dataset-level: identity
  (``disease_id``), label set + metadata, modality, download wrapper,
  split-builder factory. **One per dataset**: BUSI has one, chest CT
  has one, future skin-cancer has one.
* :class:`ModelSpec` — everything model-level for one architecture
  trained against a dataset: identity (``model_id`` + version),
  framework, :class:`~claritymed.ingest.vision.forge.tasks.base.Task`,
  hparam-search space, inference-tune space. **N per dataset**: BUSI
  could ship a U-Net + a future YOLO variant; each gets its own
  ``ModelSpec`` under ``ingest/vision/busi/models/``.

The framework (:mod:`claritymed.ingest.vision.forge.framework`)
consumes one :class:`ModelSpec` per CLI invocation and reaches
``model_spec.dataset`` for label / download / split info as needed.
Datasets never reach into models — the dependency goes one way.

Search-space DSL: :class:`Categorical` / :class:`LogUniform` /
:class:`Uniform`. Each has a ``suggest(trial, name)`` method so the
caller doesn't have to remember which Optuna call corresponds to which
distribution; the spec object knows.

Naming note: this :class:`ModelSpec` is the **training-time** spec.
The runtime-side ``claritymed.core.vision.schemas.ModelSpec`` is a
different (config-shaped) type. Import explicitly when both are in
scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol

from claritymed.core.medical_clip.schemas import Modality
from claritymed.core.vision.schemas import (
    CancerStatus,
    ClinicalAction,
    LabelMeta,
    ModelFramework,
)

if TYPE_CHECKING:
    from claritymed.ingest.vision.forge.tasks.base import Task


# --- search-space DSL ----------------------------------------------------


class SearchSpace(Protocol):
    """Anything that knows how to suggest its own value from an Optuna trial."""

    def suggest(self, trial, name: str) -> Any: ...


@dataclass(frozen=True)
class Categorical:
    """Discrete choice. Carries the value list so logs can pretty-print it."""

    choices: tuple[Any, ...]

    def suggest(self, trial, name: str):
        return trial.suggest_categorical(name, list(self.choices))


@dataclass(frozen=True)
class LogUniform:
    """Log-uniform float in ``[low, high]``. Use for LR-like params."""

    low: float
    high: float

    def suggest(self, trial, name: str) -> float:
        return trial.suggest_float(name, self.low, self.high, log=True)


@dataclass(frozen=True)
class Uniform:
    """Linear-uniform float in ``[low, high]``."""

    low: float
    high: float

    def suggest(self, trial, name: str) -> float:
        return trial.suggest_float(name, self.low, self.high)


@dataclass(frozen=True)
class BoolChoice:
    """Boolean toggle. Thin wrapper around ``Categorical([False, True])`` so
    inference-param specs read more naturally (``tta_default=BoolChoice()``).
    """

    def suggest(self, trial, name: str) -> bool:
        return trial.suggest_categorical(name, [False, True])


# --- DatasetSpec ---------------------------------------------------------


@dataclass(frozen=True)
class Splits:
    """Train / val / test torch Datasets returned by ``build_splits``."""

    train: Any
    val: Any
    test: Any


@dataclass(frozen=True)
class DatasetSpec:
    """Identity + labels + IO for one dataset.

    Per-field rationale:

    * ``disease_id`` — keyed in ``configs/vision.yaml::diseases[i].id``
      and in the on-disk artifact root
      ``~/.claritymed/models/vision/<disease_id>/``.
    * ``accepted_modality`` — KTD-V3 hard gate on incoming images.
    * ``labels`` — class index order; the model's softmax matches this.
      Changing order silently breaks every persisted manifest.
    * ``labels_meta`` — per-label description + cancer_status +
      clinical_action; surfaced verbatim in the manifest.
    * ``cancer_class`` — flips the cancer post-processing block; when
      ``True``, ``cancer_status_mapping`` + ``clinical_action_mapping``
      are derived from ``labels_meta`` automatically.
    * ``download_slug`` / ``dataset_subdir`` — Kaggle CLI driver inputs
      and the post-unzip subdir name to verify completeness.
    * ``build_splits`` — callable returning a :class:`Splits`. Takes no
      args (reaches into ``download``-resolved paths internally) so the
      framework can call it identically for every dataset.
    """

    disease_id: str
    accepted_modality: Modality
    labels: tuple[str, ...]
    labels_meta: dict[str, LabelMeta]
    cancer_class: bool
    download_slug: str
    dataset_subdir: str
    build_splits: Callable[[], Splits]

    def label_index(self, label: str) -> int:
        """Resolve a class name to its index. Raises ``ValueError`` on miss."""
        return self.labels.index(label)

    def cancer_status_mapping(self) -> dict[str, CancerStatus] | None:
        """Derive ``{label: cancer_status}`` from ``labels_meta``.

        Returns ``None`` when ``cancer_class=False`` so the manifest
        builder can pass it straight through to the schema.
        """
        if not self.cancer_class:
            return None
        return {label: self.labels_meta[label].cancer_status for label in self.labels}

    def clinical_action_mapping(self) -> dict[str, ClinicalAction] | None:
        """Derive ``{label: clinical_action}`` from ``labels_meta``."""
        if not self.cancer_class:
            return None
        return {label: self.labels_meta[label].clinical_action for label in self.labels}


# --- ModelSpec -----------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """One model variant trained against one dataset.

    Two ModelSpecs for the same ``DatasetSpec`` (e.g. a U-Net and a
    YOLO for BUSI) live in separate files under ``<dataset>/models/``.
    They produce independent artifact directories
    (``<disease_id>/<model_id>__<tag>/``) and share the dataset's
    ``LATEST.jsonl`` — the regression gate filters rows by ``model_id``
    so the comparison stays apples-to-apples.

    Fields:

    * ``dataset`` — the :class:`DatasetSpec` this model trains against.
    * ``model_id`` / ``model_version`` — surface in the manifest and in
      ``configs/vision.yaml::models``. ``model_id`` is the on-disk
      artifact-dir prefix.
    * ``framework`` — runtime framework key (``"pytorch"`` / ``"onnx"``
      / ``"ultralytics"``). The adapter dispatch on the serving side
      keys off this + ``manifest.task``.
    * ``task`` — the :class:`Task` implementation: loss + model factory
      + metric breakdown + per-phase floors + composite weights live on
      it. Two ModelSpecs for the same dataset can pick different
      tasks (a U-Net does cls+seg, a YOLO does detection).
    * ``hparam_space`` — ``{param_name: SearchSpace}``. Search phase
      walks this dict per trial.
    * ``inference_space`` — same shape, walked in the tune phase.
    """

    dataset: DatasetSpec
    model_id: str
    model_version: str
    framework: ModelFramework
    task: Task
    hparam_space: dict[str, SearchSpace] = field(default_factory=dict)
    inference_space: dict[str, SearchSpace] = field(default_factory=dict)

    def suggest_hparams(self, trial) -> dict[str, Any]:
        """Walk ``hparam_space`` and return one Optuna-suggested set."""
        return {
            name: space.suggest(trial, name)
            for name, space in self.hparam_space.items()
        }

    def suggest_inference_params(self, trial) -> dict[str, Any]:
        """Walk ``inference_space`` and return one Optuna-suggested set."""
        return {
            name: space.suggest(trial, name)
            for name, space in self.inference_space.items()
        }


__all__ = [
    "BoolChoice",
    "Categorical",
    "DatasetSpec",
    "LogUniform",
    "ModelSpec",
    "SearchSpace",
    "Splits",
    "Uniform",
]
