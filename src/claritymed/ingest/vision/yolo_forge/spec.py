"""yolo_forge specs: how to describe a (detection dataset, YOLO model) pair.

Mirrors :mod:`claritymed.ingest.vision.forge.spec` but for detection:

* :class:`DetectionDatasetSpec` — identity + class names + IO callable
  that materialises the raw archive into YOLO format on disk.
* :class:`YoloModelSpec` — identity + base weights + train + eval
  knobs for one Ultralytics architecture trained against one dataset.
* :class:`YoloTrainHparams` — the surface of ``YOLO.train(...)`` we
  expose. We default everything sane for medical chest-x-ray bbox so
  CLI invocations stay short; spec authors override only what differs.
* :class:`DetectionSplits` — the shape ``prepare_fn`` returns. Carries
  the ``data_yaml_path`` the train phase feeds straight to Ultralytics
  plus per-split sample counts (for fail-loud "did discover find
  anything?" checks).

Forge's ``ModelSpec`` carries a ``Task`` instance (cls vs cls+seg);
yolo_forge skips that abstraction — every model here is detection, so
the task layer would be one branch wide. If future detection backends
(torchvision RetinaNet, mmdetection) land we can promote that to a
``DetectionBackend`` enum then.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Reuse forge's task-agnostic search-space DSL — no point re-deriving
# Categorical / LogUniform / Uniform / BoolChoice on this side. The
# ``SearchSpace`` Protocol lives there too. Re-exported via ``__all__``
# so spec authors can write a single ``from yolo_forge.spec import ...``
# without hopping over to forge.spec.
from claritymed.ingest.vision.forge.spec import (
    BoolChoice,
    Categorical,
    LogUniform,
    SearchSpace,
    Uniform,
)


# --- train hparams --------------------------------------------------------


@dataclass(frozen=True)
class YoloTrainHparams:
    """Subset of ``ultralytics.YOLO.train(...)`` knobs we expose.

    Defaults target a single-class chest x-ray bbox task on a single
    Apple-silicon GPU; spec authors override per architecture. The
    field names match Ultralytics' kwargs verbatim so framework code
    can ``train(**asdict(hparams))`` without translation.
    """

    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    optimizer: str = "SGD"
    patience: int = 20
    # Loss-component weights (Ultralytics defaults). Surfaced as fields
    # so detection-recall-sensitive specs can pin or HPO-search them
    # alongside lr0/lrf without having to bypass the dataclass.
    box: float = 7.5
    cls: float = 0.5
    # ``None`` ⇒ framework picks the best available accelerator
    # (``mps`` > ``cuda`` > ``cpu``). Ultralytics' own auto-detect
    # prefers CPU on Apple silicon, which makes training 10-30× slower
    # than necessary; we override that default. Spec authors can pin a
    # specific device (e.g. ``"cpu"`` for a CI smoke trial).
    device: str | None = None
    # Augmentation knobs that matter for medical-image bbox: keep
    # geometric augmentations conservative (flip yes, mosaic yes, but
    # no perspective warp — would distort anatomy). Ultralytics
    # defaults are car-and-pedestrian-tuned and too aggressive.
    fliplr: float = 0.5
    flipud: float = 0.0
    mosaic: float = 1.0
    mixup: float = 0.0
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0


# --- detection dataset spec ----------------------------------------------


@dataclass(frozen=True)
class DetectionSplits:
    """What ``prepare_fn`` returns once the YOLO-format dataset is on disk.

    ``data_yaml_path`` is the file the train phase passes to
    ``YOLO.train(data=...)``. Counts are reported separately so the
    framework can fail loud if a split came out empty (e.g., upstream
    archive shrank).
    """

    data_yaml_path: Path
    train_count: int
    val_count: int
    test_count: int

    def assert_non_empty(self) -> None:
        """Raise ``RuntimeError`` if any split has zero samples."""
        for split, n in (
            ("train", self.train_count),
            ("val", self.val_count),
            ("test", self.test_count),
        ):
            if n <= 0:
                raise RuntimeError(
                    f"detection split {split!r} is empty (count={n}). "
                    f"prepare_fn returned a malformed dataset."
                )


@dataclass(frozen=True)
class DetectionDatasetSpec:
    """Identity + IO for one detection dataset.

    Per-field rationale:

    * ``dataset_id`` — keys the artifact root
      ``~/.claritymed/models/vision/<dataset_id>/``. Pick a name that
      reads as detection (e.g., ``rsna_pneumonia_detection``) so it
      can't be confused with the classification artifact root of the
      same upstream archive.
    * ``disease_id`` — mirrors the same field on the classification
      ``DatasetSpec``. Lets cross-task tooling pair "classification
      vs detection on the same disease" without name fragility.
    * ``class_names`` — order matters: written verbatim into
      ``data.yaml::names`` and indexed into in every label .txt file.
      Changing order silently mislabels every cached label file.
    * ``prepare_fn`` — takes no args (resolves paths internally) and
      returns :class:`DetectionSplits`. Must be idempotent — invoking
      it twice with the cached PNGs already on disk must not redo any
      DICOM conversion.
    """

    dataset_id: str
    disease_id: str
    class_names: tuple[str, ...]
    prepare_fn: Callable[[], DetectionSplits]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


# --- model spec -----------------------------------------------------------


@dataclass(frozen=True)
class YoloModelSpec:
    """One Ultralytics YOLO variant trained against one detection dataset.

    Fields:

    * ``dataset`` — the :class:`DetectionDatasetSpec` to train against.
    * ``model_id`` — on-disk artifact-dir prefix (e.g.
      ``rsna_pneumonia_yolov8n_v1``). Must be unique across all model
      specs that share ``dataset.dataset_id``.
    * ``model_version`` — semantic version string surfaced in the
      manifest; bump when hparams or architecture change.
    * ``base_weights`` — Ultralytics weight identifier (e.g.
      ``"yolov8n.pt"``). Ultralytics downloads on first use and caches
      in its own pip-installable location.
    * ``train_hparams`` — :class:`YoloTrainHparams`. Spec authors
      override only what differs from defaults; ``search`` phase tunes
      a subset by walking :attr:`hparam_space` per trial.
    * ``hparam_space`` — ``{name: SearchSpace}``. The search phase
      Optuna-walks this dict each trial; suggested values overlay
      :attr:`train_hparams` defaults (so unsearched dims stay at the
      spec's value). Empty dict ⇒ no search phase work.
    * ``inference_space`` — ``{name: SearchSpace}``. Walked by the
      tune phase; suggested values feed ``model.val(**kwargs)`` (e.g.
      ``conf``, ``iou``) on the val split. Empty dict ⇒ no tune work.
    * ``eval_thresholds`` — deploy-phase gates as ``{metric_name:
      min_value}``. Standard keys: ``"mAP50"``, ``"mAP50-95"``,
      ``"image_recall"``, ``"image_precision"``. Train + eval phases
      compute all four; deploy gate iterates this dict.
    """

    dataset: DetectionDatasetSpec
    model_id: str
    model_version: str
    base_weights: str
    train_hparams: YoloTrainHparams = field(default_factory=YoloTrainHparams)
    hparam_space: dict[str, SearchSpace] = field(default_factory=dict)
    inference_space: dict[str, SearchSpace] = field(default_factory=dict)
    eval_thresholds: dict[str, float] = field(default_factory=dict)

    def suggest_hparams(self, trial) -> dict[str, Any]:
        """Walk ``hparam_space`` once; return one Optuna-suggested set."""
        return {
            name: space.suggest(trial, name)
            for name, space in self.hparam_space.items()
        }

    def suggest_inference_params(self, trial) -> dict[str, Any]:
        """Walk ``inference_space`` once; return one Optuna-suggested set."""
        return {
            name: space.suggest(trial, name)
            for name, space in self.inference_space.items()
        }


__all__ = [
    "BoolChoice",
    "Categorical",
    "DetectionDatasetSpec",
    "DetectionSplits",
    "LogUniform",
    "SearchSpace",
    "Uniform",
    "YoloModelSpec",
    "YoloTrainHparams",
]
