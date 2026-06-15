"""Vision tool primitives — disease detection from medical images.

The ``vision`` module owns the LLM-facing tool contract (``RawDetection``
/ ``LLMDetectionPayload``), the registry-level config models
(``VisionConfig`` / ``DiseaseSpec`` / ``ModelSpec`` / ``ServerSpec``),
and the on-disk manifest model (``Manifest``).

The ``Modality`` Literal is *not* defined here — it lives in
``claritymed.core.medical_clip.schemas`` so the modality vocabulary
shared with the BiomedCLIP server stays in one place.
"""
