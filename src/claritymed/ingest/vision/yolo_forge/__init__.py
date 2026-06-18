"""yolo_forge — Ultralytics-backed detection pipeline framework.

Sibling to :mod:`claritymed.ingest.vision.forge` (the classification /
segmentation framework). Shares the on-disk artifact layout
(``~/.claritymed/models/vision/<dataset_id>/run/<model_id>_<ts>/``) and
``LATEST.jsonl`` audit conventions, but the phases differ:

* ``prepare`` — convert raw dataset → YOLO format on disk (data.yaml +
  per-image label .txt files + image symlinks). Idempotent.
* ``train`` — ``ultralytics.YOLO(...).train(data=<yaml>, ...)`` with
  the spec's train hparams; artifacts redirected to the staging dir.
* ``eval`` — ``model.val()`` for mAP plus an image-level recall metric
  (any-box ≥ conf → positive) so a clinical fail-safe stays comparable
  to the classification pipeline.
* ``deploy`` — gate on the spec's eval thresholds + regression check
  against the last entry in ``LATEST.jsonl`` for this model_id.

The framework knows nothing about any concrete dataset — every
dataset ships its own ``DetectionDatasetSpec`` (with a ``prepare_fn``
that knows how to materialise raw → YOLO format) and one or more
``YoloModelSpec`` instances. Per-dataset modules live as siblings of
this package (parallel to how the classification ``forge`` relates to
``rsna_pneumonia`` / ``busi`` / etc.); RSNA Pneumonia detection is
:mod:`claritymed.ingest.vision.rsna_pneumonia_yolo`.

Install the optional dependency stack (ultralytics + torch + pydicom)
with ``uv sync --extra yolo-forge`` before invoking the CLI.
"""
