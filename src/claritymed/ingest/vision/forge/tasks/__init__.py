"""Task implementations: classification, classification+segmentation, …

Each module under this package defines one concrete subclass of
:class:`base.Task`. The :class:`~claritymed.ingest.vision.forge.spec.ModelSpec`
holds a Task *instance* (not a class) so per-model floors / composite
weights / critical labels can vary while sharing the same task type.
"""

from claritymed.ingest.vision.forge.tasks.base import Task
from claritymed.ingest.vision.forge.tasks.classification import ClassificationTask
from claritymed.ingest.vision.forge.tasks.cls_segmentation import (
    ClassificationSegmentationTask,
)

__all__ = ["Task", "ClassificationTask", "ClassificationSegmentationTask"]
