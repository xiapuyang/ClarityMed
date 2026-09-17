"""Async HTTP client for the vision server.

One client per configured ``ServerSpec`` — the registry constructs them
at orchestrator boot and hands them to the tool body. Per CLAUDE.md's
service-layer rule, no other module talks ``httpx`` to the vision
server directly; if a future replacement (gRPC, batching wrapper,
multi-host gateway) lands, only this module changes.

Fail-loud contract mirrors ``SymptomsServerClient`` and
``MedicalClipClient``:

* connection error, timeout, or 5xx → ``VisionServerUnreachableError``
* 4xx (unknown_disease, modality_mismatch, image_decode_failed) →
  propagated as ``httpx.HTTPStatusError`` so the caller can inspect
  ``response.json()["error"]["code"]`` and branch.

Timeouts come from the plan: connect=5s, read=25s (slightly above
``tool.total_budget_ms=20000`` so a single attempt can complete or fail
fast within the per-tool-call budget), write=10s.
"""

from __future__ import annotations

import base64
import hashlib
import logging
from typing import Any

import httpx

from claritymed.core.medical_clip.schemas import ImagePayload
from claritymed.core.vision.wire import (
    CatalogResponse,
    DetectOptions,
    DetectRequest,
    DetectResponse,
    HealthResponse,
)
from claritymed.errors import VisionServerUnreachableError

logger = logging.getLogger(__name__)

DEFAULT_CONNECT_TIMEOUT_S = 5.0
DEFAULT_READ_TIMEOUT_S = 25.0
DEFAULT_WRITE_TIMEOUT_S = 10.0


class VisionHttpClient:
    """Async HTTP client for one vision-server instance.

    Each ``ServerSpec`` in ``configs/vision.yaml::servers`` gets its own
    client. The registry constructs them at orchestrator boot; the tool
    body retrieves one via ``VisionRegistry.route()``.

    Tests inject a fake ``httpx.MockTransport`` via ``transport`` so the
    client exercises the wire path without binding a port.
    """

    def __init__(
        self,
        base_url: str,
        *,
        connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
        read_timeout_s: float = DEFAULT_READ_TIMEOUT_S,
        write_timeout_s: float = DEFAULT_WRITE_TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        timeout = httpx.Timeout(
            connect=connect_timeout_s,
            read=read_timeout_s,
            write=write_timeout_s,
            pool=write_timeout_s,
        )
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
            trust_env=False,
        )
        self._base_url = base_url

    @property
    def base_url(self) -> str:
        return self._base_url

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> VisionHttpClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        await self.aclose()

    # --- endpoints --------------------------------------------------------

    async def health(self) -> HealthResponse:
        """``GET /health`` — readiness snapshot."""
        data = await self._request_json("GET", "/health")
        return HealthResponse.model_validate(data)

    async def catalog(self) -> CatalogResponse:
        """``GET /v1/catalog`` — per-model truth, consumed at boot by the registry."""
        data = await self._request_json("GET", "/v1/catalog")
        return CatalogResponse.model_validate(data)

    async def detect(
        self,
        *,
        request_id: str,
        disease_id: str,
        model_id: str | None,
        image_bytes: bytes,
        language: str = "en",
        options: DetectOptions | None = None,
        sha256: str | None = None,
    ) -> DetectResponse:
        """``POST /v1/detect`` — run one image through one model.

        Args:
            request_id: Forwarded as ``X-Request-ID`` and echoed in the
                response. The tool body threads the orchestrator's
                ``request_id_ctx`` value through.
            disease_id: Resolved by :class:`VisionRegistry` upstream;
                the server cross-checks against its catalog and 404s on
                miss.
            model_id: ``None`` lets the server fall back to the disease's
                ``primary_model_id`` (mirrors ``DetectRequest.model_id``
                optionality).
            image_bytes: Raw image bytes; the client base64-encodes them
                on the wire and computes/threads the sha256 so the
                server can refuse stale or tampered uploads.
            language: ``"en"`` / ``"zh"`` — the server uses this to
                localize ``warnings`` (KTD-V6 / Unit 6 future work).
            options: ``DetectOptions`` knob bundle. Defaults to
                ``return_segmentation=True``, no saliency, no TTA.
            sha256: Pre-computed digest; defaults to
                ``hashlib.sha256(image_bytes).hexdigest()`` so callers
                that already store the digest avoid re-hashing.

        Raises:
            VisionServerUnreachableError: Connection failure, timeout,
                or 5xx response. The tool body's fallback flow catches
                this and tries the next model in
                ``disease.effective_flow`` (or returns
                ``NoUsableResultError`` when the budget runs out).
            httpx.HTTPStatusError: 4xx — ``modality_mismatch`` /
                ``unknown_disease`` / ``unknown_model`` /
                ``image_decode_failed`` / ``image_hash_mismatch``. The
                caller inspects ``response.json()["error"]["code"]``.
        """
        digest = sha256 or hashlib.sha256(image_bytes).hexdigest()
        body = DetectRequest(
            request_id=request_id,
            disease_id=disease_id,
            model_id=model_id,
            image=ImagePayload(
                sha256=digest,
                data_b64=base64.b64encode(image_bytes).decode("ascii"),
            ),
            language=language,
            options=options or DetectOptions(),
        ).model_dump()
        data = await self._request_json(
            "POST", "/v1/detect", json=body, request_id=request_id
        )
        return DetectResponse.model_validate(data)

    # --- internals --------------------------------------------------------

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        request_id: str | None = None,
        **kwargs: Any,
    ) -> dict:
        if request_id:
            headers = kwargs.pop("headers", {})
            headers["X-Request-ID"] = request_id
            kwargs["headers"] = headers
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            raise VisionServerUnreachableError(
                f"vision server unreachable at {self._base_url}: {exc!s}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise VisionServerUnreachableError(
                f"vision server timeout at {self._base_url}{path}: {exc!s}"
            ) from exc
        except httpx.HTTPError as exc:
            raise VisionServerUnreachableError(
                f"vision transport error at {self._base_url}{path}: {exc!s}"
            ) from exc
        if response.status_code >= 500:
            raise VisionServerUnreachableError(
                f"vision server {response.status_code} at {path}: {response.text[:200]}"
            )
        if response.status_code >= 400:
            # 4xx are expected application errors (modality mismatch,
            # unknown disease, image decode) — propagate so the caller
            # branches on the error code in the body.
            response.raise_for_status()
        return response.json()


__all__ = ["VisionHttpClient"]
