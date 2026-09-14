"""Colon histopathology dataset — 2-class LC25000 colon subset classification.

Classifier-only (the upstream archive ships no segmentation masks).
Classes follow the LC25000 colon taxonomy (Borkowski et al., 2019):

* ``adenocarcinoma`` — malignant
* ``normal`` — healthy colon tissue baseline

The upstream archive uses short folder names (``colon_aca`` /
``colon_n``); the loader expands them to the canonical labels so
manifests and i18n bundles stay readable — same naming convention as
the chest_ct module's labels.

This module shares the Kaggle download with
:mod:`claritymed.ingest.vision.lung_histopath` via
:mod:`claritymed.ingest.vision.lung_colon_histopath`; the on-disk path
is shared. The discovery here walks ``colon_image_sets/`` only — lung
training data is intentionally invisible to this model because lung
and colon biopsies are clinically independent.

Modules:

* ``dataset`` — folder walker + stratified split + Torch ``Dataset``.
* ``dataset_spec`` — :class:`~claritymed.ingest.vision.forge.spec.DatasetSpec`
  instance ``COLON_HISTOPATH_DATASET`` carrying labels, label-metadata,
  ``histopathology`` modality, the shared download slug, and the
  split-builder factory.
* ``models/resnet50_v1`` — :class:`ModelSpec` for the ResNet-50 /
  EfficientNet classifier search. Pick it up via the forge CLI:

  .. code-block:: bash

      claritymed-vision-forge pipeline \\
          --model claritymed.ingest.vision.colon_histopath.models.resnet50_v1:RESNET50_V1
"""
