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

# Negative (no-bbox) image sub-sampling at prep time. RSNA's natural
# neg:pos is 3.44:1, which dilutes positive signal and adds 3.4× of
# essentially-blank forward/backward per epoch. Three canonical modes:
#
#   None  → keep_all  (the historical default; 3.44:1)
#   0.0   → drop      (positives only; fastest)
#   1.0   → balanced  (cap negatives at #positives per split)
#
# Default is ``1.0`` (balanced) — best ROI on this dataset. Override
# here for A/B (e.g. flip to ``None`` to reproduce the pre-knob runs).
# Switching modes lands in a different prepared dir so cached
# images/labels for other modes remain valid.
NEGATIVE_RATIO: float | None = 1.0


def _prepare() -> DetectionSplits:
    """Thin closure so the spec exposes a no-arg callable like forge expects."""
    return prepare_rsna_pneumonia_yolo(
        class_names=RSNA_PNEUMONIA_YOLO_CLASSES,
        negative_ratio=NEGATIVE_RATIO,
    )


RSNA_PNEUMONIA_YOLO_DATASET = DetectionDatasetSpec(
    dataset_id="rsna_pneumonia_detection",
    disease_id="chest_xray_pneumonia",
    class_names=RSNA_PNEUMONIA_YOLO_CLASSES,
    prepare_fn=_prepare,
)

__all__ = ["RSNA_PNEUMONIA_YOLO_DATASET", "RSNA_PNEUMONIA_YOLO_CLASSES"]
