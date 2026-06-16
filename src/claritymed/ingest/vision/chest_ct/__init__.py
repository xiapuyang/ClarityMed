"""Chest CT dataset — 4-class lung-cancer classification.

Classifier-only (no segmentation masks shipped with the dataset).
Classes: ``adenocarcinoma``, ``large_cell_carcinoma``, ``normal``,
``squamous_cell_carcinoma``.

Modules:

* ``download`` — Kaggle CLI wrapper. Pulls
  ``mohamedhanyyy/chest-ctscan-images`` into
  ``~/.claritymed/data/vision/chest_ct/``. Same credentials as BUSI
  (``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` env vars or
  ``~/.kaggle/kaggle.json``).
* ``dataset`` — folder-walker + stratified split + Torch ``Dataset``.
  The upstream archive's class-folder names embed staging metadata
  (e.g. ``adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib``); the loader
  normalises every variant to one of the four canonical labels in
  :data:`dataset.CHEST_CT_LABELS`.

Training / tuning / deploy pipelines are not yet ported. The shape is
identical to ``ingest/vision/busi/`` minus the segmentation head — when
that work is picked up, lift ``busi/train.py`` + ``busi/hparam.py`` +
``busi/tune.py`` + ``busi/deploy.py`` as starting templates and:

1. Drop the U-Net + dice loss; use a ResNet-50 / EfficientNet-B3
   classification head.
2. Update ``LABELS_META`` in ``train.py`` for the four CT classes.
3. Update phase floors in ``scoring.py`` — the BUSI floors lean on
   the ``malignant_recall`` metric; for a 4-class CT classifier the
   equivalent is "any cancer recall" (union of the three malignant
   classes vs ``normal``).

The :class:`~claritymed.core.vision.schemas.Manifest` already supports
``task="classification"`` without a segmentation block, so no schema
work is required.
"""
