"""Lung histopathology dataset — 3-class LC25000 lung subset classification.

Classifier-only (the upstream archive ships no segmentation masks).
Classes follow the LC25000 lung taxonomy (Borkowski et al., 2019):

* ``adenocarcinoma`` — malignant
* ``normal`` — healthy lung tissue baseline
* ``squamous_cell_carcinoma`` — malignant

The upstream archive uses short folder names (``lung_aca`` /
``lung_n`` / ``lung_scc``); the loader expands them to the canonical
underscore-snake labels so manifests and i18n bundles stay readable
— same naming convention as the ``lung_cancer_chest_ct`` disease.

This module shares the Kaggle download with
:mod:`claritymed.ingest.vision.colon_histopath` via
:mod:`claritymed.ingest.vision.lung_colon_histopath`; the on-disk path
is shared. The discovery here walks ``lung_image_sets/`` only — colon
training data is intentionally invisible to this model because lung
and colon biopsies are clinically independent.

Modules:

* ``dataset`` — folder walker + stratified split + Torch ``Dataset``.
* ``dataset_spec`` — :class:`~claritymed.ingest.vision.forge.spec.DatasetSpec`
  instance ``LUNG_HISTOPATH_DATASET`` carrying labels, label-metadata,
  ``histopathology`` modality, the shared download slug, and the
  split-builder factory.
* ``models/resnet50_v1`` — :class:`ModelSpec` for the ResNet-50 /
  EfficientNet classifier search. Pick it up via the forge CLI:

  .. code-block:: bash

      claritymed-vision-forge pipeline \\
          --model claritymed.ingest.vision.lung_histopath.models.resnet50_v1:RESNET50_V1
"""
