"""Chest X-Ray Pneumonia (Kermany et al. 2018) — Kaggle download wrapper.

This module pulls Paul Mooney's mirror of Kermany 2018 on Kaggle
(``paultimothymooney/chest-xray-pneumonia``, 5,856 pediatric chest
radiographs labeled NORMAL / PNEUMONIA) into
``~/.claritymed/data/vision/chest_xray_pneumonia/``.

Unlike the lung+colon histopath module, no per-disease detector lives
here (yet). The dataset's primary use today is **modality coverage**:
xray was the only modality with no labeled image source on disk, so
the bench / e2e couldn't verify BiomedCLIP behaves on radiographs.
A pneumonia-classifier sibling module can plug in later by walking
the same on-disk layout — ``chest_xray/{train,test,val}/{NORMAL,PNEUMONIA}/``.

Same credentials story as the other Kaggle modules: ``KAGGLE_USERNAME``
+ ``KAGGLE_KEY`` env vars or ``~/.kaggle/kaggle.json``.
"""
