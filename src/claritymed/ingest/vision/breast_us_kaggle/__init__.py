"""Breast-ultrasound Kaggle dataset — 2-class alternative variant.

A second ultrasound dataset for the existing
``breast_cancer_ultrasound`` disease. Source: Kaggle
``vuppalaadithyasairam/ultrasound-breast-images-for-breast-cancer``,
~9000 images, augmented (rotation + sharpening), only ``benign`` and
``malignant`` (no ``normal`` class).

Why a second dataset for the same disease: the BUSI module trains a
3-class model (benign / malignant / normal); this Kaggle archive
trades the normal class for substantially more cancer-positive
training data. Operators can promote it as a fallback model in
``configs/vision.yaml::diseases[breast_cancer_ultrasound].flow`` when
they want better positive-class recall at the cost of the normal
distinction.

Modules:

* ``download`` — Kaggle CLI wrapper.
* ``dataset`` — folder-walker + stratified split + Torch ``Dataset``.
* ``dataset_spec`` — :class:`DatasetSpec` instance
  ``BREAST_US_KAGGLE_DATASET``. ``disease_id`` matches BUSI's
  (``breast_cancer_ultrasound``) so it serves the same vision tool.
* ``models/resnet50_v1`` — :class:`ModelSpec` for the ResNet-50 /
  EfficientNet classifier search. Forge CLI:

  .. code-block:: bash

      claritymed-vision-forge pipeline \\
          --model claritymed.ingest.vision.breast_us_kaggle.models.resnet50_v1:RESNET50_V1
"""
