"""Standalone benchmark — modality classifier A/B.

Compares modality classifiers (BiomedCLIP via medical-clip-server vs
an omlx-backed multimodal LLM) on a labeled image set drawn from the
already-downloaded vision training datasets. ResNet-class classifiers
plug in through the same ``ModalityClassifier`` Protocol.

See ``run.py`` for the harness, ``samples.py`` for the dataset-to-modality
mapping, and ``classifiers.py`` for the Protocol + shipped backends.
"""
