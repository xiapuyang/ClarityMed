"""YOLO architecture variants trained against RSNA Pneumonia detection.

Each ``*_v<N>.py`` ships one :class:`YoloModelSpec` instance. Bumping
the version suffix when hparams or architecture change keeps the
artifact directory + ``LATEST.jsonl`` history immutable.
"""
