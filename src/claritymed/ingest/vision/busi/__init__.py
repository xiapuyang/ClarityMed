"""BUSI dataset — breast-ultrasound classification + segmentation.

Modules:

* ``download`` — Kaggle CLI wrapper. Pulls
  ``aryashah2k/breast-ultrasound-images-dataset`` (``Dataset_BUSI_with_GT``)
  into ``~/.claritymed/data/vision/busi/``. Requires Kaggle credentials
  (``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` env vars or
  ``~/.kaggle/kaggle.json``).
* ``dataset`` — ``BUSIDataset(torch.utils.data.Dataset)``: train/val/test
  split deterministic by patient id; returns ``(image, mask, label)``
  tuples. Stratified to keep the rare ``normal`` class in every split.
* ``dataset_spec`` — :class:`~claritymed.ingest.vision.forge.spec.DatasetSpec`
  instance ``BUSI_DATASET`` carrying labels, label-metadata, modality,
  download slug, and the split-builder factory.
* ``models/unet_resnet50`` — :class:`ModelSpec` for the U-Net + cls
  head architecture; pick this up via the forge CLI:

  .. code-block:: bash

      claritymed-vision-forge pipeline \\
          --model claritymed.ingest.vision.busi.models.unet_resnet50:UNET_RESNET50

Hparam search / training / tune / deploy live in
``claritymed.ingest.vision.forge`` and are dataset-agnostic — the
spec carries every per-dataset knob.
"""
