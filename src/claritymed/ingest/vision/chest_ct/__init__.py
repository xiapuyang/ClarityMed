"""Chest CT dataset — 4-class lung-cancer classification.

Classifier-only (no segmentation masks shipped with the dataset).
Classes: ``adenocarcinoma``, ``large_cell_carcinoma``, ``normal``,
``squamous_cell_carcinoma``.

Modules:

* ``download`` — Kaggle CLI wrapper. Pulls
  ``mohamedhanyyy/chest-ctscan-images`` into
  ``~/.claritymed/data/vision/chest_ct/``. Same credentials story as
  BUSI (``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` env vars or
  ``~/.kaggle/kaggle.json``).
* ``dataset`` — folder-walker + stratified split + Torch ``Dataset``.
  The upstream archive's class-folder names embed staging metadata
  (e.g. ``adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib``); the loader
  normalises every variant to one of the four canonical labels in
  :data:`dataset.CHEST_CT_LABELS`.
* ``dataset_spec`` — :class:`~claritymed.ingest.vision.forge.spec.DatasetSpec`
  instance ``CHEST_CT_DATASET`` carrying labels, label-metadata,
  modality, download slug, and the split-builder factory.
* ``models/resnet50_v1`` — :class:`ModelSpec` for the ResNet-50 /
  EfficientNet classifier search; pick this up via the forge CLI:

  .. code-block:: bash

      claritymed-vision-forge pipeline \\
          --model claritymed.ingest.vision.chest_ct.models.resnet50_v1:RESNET50_V1

The :class:`~claritymed.core.vision.schemas.Manifest` already supports
``task="classification"`` without a segmentation block, so the same
forge framework drives both BUSI (cls+seg) and chest CT (cls-only)
end-to-end.
"""
