"""BUSI dataset — breast-ultrasound image classification + segmentation.

Modules:

* ``download`` — Kaggle CLI wrapper. Pulls
  ``aryashah2k/breast-ultrasound-images-dataset`` (Dataset_BUSI_with_GT)
  into ``~/.claritymed/data/vision/busi/``. Requires Kaggle credentials
  (``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` env vars or
  ``~/.kaggle/kaggle.json``).
* ``dataset`` — ``BUSIDataset(torch.utils.data.Dataset)``: train/val/test
  split deterministic by patient id; returns ``(image, mask, label)``
  tuples. Stratified to keep the rare ``normal`` class in every split.
* ``hparam`` — Optuna search over backbone + LR + segmentation-loss
  weight; persists trials to the shared
  ``CLARITYMED_HOME/tracking/optuna.db`` (study name disambiguates).
* ``train`` — production training. Outputs ``weights.pt`` +
  ``manifest.json`` (with sha256s, eval metrics,
  ``cancer_status_mapping``, ``clinical_action_mapping``,
  ``labels_meta``).
* ``tune`` — inference-time param sweep (classification threshold +
  TTA on/off) against the held-out test split.

See ``docs/vision-model-workflow.md`` for the end-to-end recipe.
"""
