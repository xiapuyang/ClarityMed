"""Skin lesion dermoscopy dataset — 9-class ISIC classification.

Classifier-only (the upstream ISIC archive ships no lesion masks). Classes
follow the ISIC 9-class taxonomy used by the
``sharanharsoor/skin-cancer-detection`` reference notebook:

* ``actinic_keratosis``
* ``basal_cell_carcinoma``
* ``dermatofibroma``
* ``melanoma``
* ``nevus``
* ``pigmented_benign_keratosis``
* ``seborrheic_keratosis``
* ``squamous_cell_carcinoma``
* ``vascular_lesion``

Modules:

* ``download`` — Kaggle CLI wrapper. Pulls
  ``nodoubttome/skin-cancer9-classesisic`` into
  ``~/.claritymed/data/vision/skin_lesion/``. Same credentials story as
  BUSI / chest CT (``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` env vars or
  ``~/.kaggle/kaggle.json``).
* ``dataset`` — folder-walker + stratified split + Torch ``Dataset``. The
  upstream archive ships images under ``Skin cancer ISIC The
  International Skin Imaging Collaboration/{Train,Test}/<label>/``; the
  loader normalises folder names (lower-case, spaces and dots → ``_``)
  to the canonical labels and pools across both subsplits before our
  hash-stratified split picks them. Same rationale as chest_ct: the
  upstream split is tilted toward demo balance and we want statistical
  representativeness in val + test.
* ``dataset_spec`` — :class:`~claritymed.ingest.vision.forge.spec.DatasetSpec`
  instance ``SKIN_LESION_DATASET`` carrying labels, label-metadata,
  ``dermoscopy`` modality, download slug, and the split-builder factory.
* ``models/resnet50_v1`` — :class:`ModelSpec` for the ResNet-50 /
  EfficientNet classifier search. Pick it up via the forge CLI:

  .. code-block:: bash

      claritymed-vision-forge pipeline \\
          --model claritymed.ingest.vision.skin_lesion.models.resnet50_v1:RESNET50_V1
"""
