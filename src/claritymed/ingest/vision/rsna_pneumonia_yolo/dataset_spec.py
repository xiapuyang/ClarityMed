"""``RSNA_PNEUMONIA_YOLO_DATASET`` — :class:`DetectionDatasetSpec` instance.

Single class (``pneumonia``). ``disease_id`` matches the classification
adapter so a future cross-task bench can pair them. ``dataset_id``
suffixes ``_detection`` to keep the artifact roots distinct from the
classification pipeline under ``~/.claritymed/models/vision/``.
"""

from __future__ import annotations

from claritymed.ingest.vision.rsna_pneumonia_yolo.dataset import (
    prepare_rsna_pneumonia_yolo,
)
from claritymed.ingest.vision.yolo_forge.spec import (
    DetectionDatasetSpec,
    DetectionSplits,
)

RSNA_PNEUMONIA_YOLO_CLASSES: tuple[str, ...] = ("pneumonia",)


def _prepare() -> DetectionSplits:
    """Thin closure so the spec exposes a no-arg callable like forge expects."""
    return prepare_rsna_pneumonia_yolo(class_names=RSNA_PNEUMONIA_YOLO_CLASSES)


RSNA_PNEUMONIA_YOLO_DATASET = DetectionDatasetSpec(
    dataset_id="rsna_pneumonia_detection",
    disease_id="chest_xray_pneumonia",
    class_names=RSNA_PNEUMONIA_YOLO_CLASSES,
    prepare_fn=_prepare,
)

__all__ = ["RSNA_PNEUMONIA_YOLO_DATASET", "RSNA_PNEUMONIA_YOLO_CLASSES"]
