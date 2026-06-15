"""Re-export shim — wire models live in ``claritymed.core.vision.wire``.

Kept here so the FastAPI app module (``servers/vision/app.py``, Unit 4) and
its tests can ``from claritymed.servers.vision.wire import ...`` without
reaching across the package boundary into ``core``. Mirrors the
``servers/symptoms/wire.py`` pattern.
"""

from claritymed.core.vision.wire import (
    CatalogModel as CatalogModel,
    CatalogResponse as CatalogResponse,
    DetectOptions as DetectOptions,
    DetectRequest as DetectRequest,
    DetectResponse as DetectResponse,
    ErrorDetail as ErrorDetail,
    ErrorResponse as ErrorResponse,
    HealthLoadedModel as HealthLoadedModel,
    HealthResponse as HealthResponse,
)
