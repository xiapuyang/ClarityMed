"""RSNA Pneumonia Detection Challenge — Kaggle download wrapper.

Pulls ``rsna-pneumonia-detection-challenge`` (a Kaggle **competition**,
not a regular dataset, so the CLI invocation differs) into
``~/.claritymed/data/vision/rsna_pneumonia/``.

The dataset ships ~30k adult chest X-ray DICOMs labeled NORMAL /
PNEUMONIA (with bounding boxes for pneumonia cases). We use the
image-level binary ``Target`` label from ``stage_2_train_labels.csv``
and ignore the bbox data — the cross-dataset drift bench scores
binary classification only.

Why this dataset: it's the cleanest available adult-population
counterpart to Kermany's pediatric pneumonia dataset. Same disease,
same modality (frontal chest X-ray), same binary task — but a wildly
different acquisition population (adult vs pediatric, multi-center vs
single-hospital, portable AP vs standing PA). That's exactly the
distribution shift the drift bench is designed to quantify.

Credentials: the Kaggle CLI must be configured (``KAGGLE_USERNAME`` +
``KAGGLE_KEY`` or ``~/.kaggle/kaggle.json``) **and** the operator must
have accepted the competition rules at
https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/rules
before the download will succeed.
"""
