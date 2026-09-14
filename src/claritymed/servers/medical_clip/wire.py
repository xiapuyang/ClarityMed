"""Re-export shim — wire models live in ``claritymed.core.medical_clip.schemas``.

Kept here so the FastAPI app module (``servers/medical_clip/app.py``, Unit 2)
and its tests can import wire models from the server package without
reaching into ``core``. Mirrors the ``servers/symptoms/wire.py`` pattern.

The error envelope is shared with the vision server (``core/vision/wire``) —
no medical-clip-specific error shape is necessary in v1; the same
``code`` / ``message`` / ``details`` triple covers both servers.
"""

from claritymed.core.medical_clip.schemas import (
    HealthResponse as HealthResponse,
    ImagePayload as ImagePayload,
    ModalityRequest as ModalityRequest,
    ModalityResponse as ModalityResponse,
    ModalityScore as ModalityScore,
)
from claritymed.core.vision.wire import (
    ErrorDetail as ErrorDetail,
    ErrorResponse as ErrorResponse,
)
