"""Forge — task-polymorphic training framework for medical vision models.

Forge is the shared infrastructure that drives every dataset's
hparam-search → train → tune → deploy pipeline. Each dataset only
ships its data loader (`dataset.py`), download wrapper, a
:class:`spec.DatasetSpec`, and one :class:`spec.ModelSpec` per
architecture under ``<dataset>/models/``. Adding a new dataset = two
files + a CLI flag; no pipeline code to write.

Module layout:

* :mod:`common` — sha256 / device / JSONL / YAML patch / softmax /
  Optuna helpers. Dataset- and task-agnostic.
* :mod:`spec` — :class:`DatasetSpec`, :class:`ModelSpec`, the
  search-space DSL (:class:`Categorical` / :class:`LogUniform` /
  :class:`Uniform`).
* :mod:`scoring` — :class:`PhaseFloors` (dict-of-floors), generic
  :func:`feasibility_aware_score` / :func:`gate_or_raise`. Each
  :class:`Task` plugs in its own ``composite_weights`` and per-phase
  floors.
* :mod:`tasks.base` — :class:`Task` ABC. Concrete tasks in
  :mod:`tasks.classification` and :mod:`tasks.cls_segmentation`.
* :mod:`framework` — :func:`run_hparam` / :func:`run_train` /
  :func:`run_tune` / :func:`run_deploy` / :func:`run_pipeline`. All
  five accept a :class:`ModelSpec`; the Task abstraction does the
  heavy lifting.
* :mod:`cli` — ``claritymed-vision-forge`` entry point. Resolves
  ``--model claritymed.ingest.vision.busi.models.unet_resnet50:UNET_RESNET50``
  via ``importlib``.

The runtime side (``servers/vision/adapters/forge_torch.py``) reads
``manifest.task`` and picks the matching adapter so the same forge
output works for cls-only and cls+seg checkpoints alike.
"""

from claritymed.ingest.vision.forge.spec import (
    DatasetSpec,
    ModelSpec,
    Categorical,
    LogUniform,
    Uniform,
)

__all__ = [
    "DatasetSpec",
    "ModelSpec",
    "Categorical",
    "LogUniform",
    "Uniform",
]
