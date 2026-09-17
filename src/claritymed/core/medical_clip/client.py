"""Async HTTP client for the medical-clip server.

Per CLAUDE.md's service-layer rule, every consumer that needs zero-shot
modality classification goes through this client — the attachment
ingest worker (Unit 3) and the vision plugin's askuserquestion path
(Unit 7) never construct ``httpx.AsyncClient`` directly. If a future
replacement (a self-trained ResNet18 classifier, a different runtime)
lands, only this module changes.

Fail-loud contract mirrors :class:`~claritymed.core.symptoms.client.SymptomsServerClient`:
connection errors, timeouts, and 5xx responses raise typed
:class:`~claritymed.errors.MedicalClipUnreachableError`. 4xx responses
(image_decode_failed, image_hash_mismatch) propagate as
:class:`httpx.HTTPStatusError` so callers can inspect ``status_code`` /
``code`` and branch — the OCR worker tags ``modality=unknown`` and
continues, the vision plugin emits askuserquestion.
"""

from __future__ import annotations

import base64
import hashlib
import logging
from typing import Any

import httpx

from claritymed.core.medical_clip.schemas import (
    HealthResponse,
    ImagePayload,
    ModalityRequest,
    ModalityResponse,
)
from claritymed.errors import MedicalClipUnreachableError

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:8086"
DEFAULT_CONNECT_TIMEOUT_S = 5.0
DEFAULT_READ_TIMEOUT_S = 30.0
DEFAULT_WRITE_TIMEOUT_S = 10.0


class MedicalClipClient:
    """Client for the loopback-bound medical-clip FastAPI server.

    Construct once per worker (OCR ingest holds a singleton). The
    underlying ``httpx.AsyncClient`` pools connections, so repeated
    classify calls during a batch attachment ingest skip TCP setup.

    Tests inject a fake :class:`httpx.MockTransport` via ``transport``
    so the client exercises against a router that returns canned
    responses without touching the network.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
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

    async def __aenter__(self) -> MedicalClipClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        await self.aclose()

    # --- endpoints --------------------------------------------------------

    async def health(self) -> HealthResponse:
        """``GET /health`` — engine readiness snapshot."""
        data = await self._request_json("GET", "/health")
        return HealthResponse.model_validate(data)

    async def classify_modality(
        self,
        image_bytes: bytes,
        *,
        request_id: str,
        sha256: str | None = None,
    ) -> ModalityResponse:
        """``POST /v1/classify_modality`` — score one image's modality.

        Args:
            image_bytes: Raw image bytes (JPEG/PNG/etc). The client
                base64-encodes them on the wire; the server re-hashes
                + cross-checks against ``sha256`` to refuse stale or
                tampered uploads.
            request_id: Per-call trace id; forwarded as ``X-Request-ID``
                and echoed in the response.
            sha256: Pre-computed sha256 hex. Defaults to ``hashlib.sha256``
                of ``image_bytes`` so callers that already store the
                digest on the blob avoid re-hashing.

        Raises:
            MedicalClipUnreachableError: Connection failure, timeout,
                or 5xx response. Caller decides fallback —
                ``modality=unknown`` for the OCR worker; askuserquestion
                for the vision plugin.
            httpx.HTTPStatusError: 4xx response (image_decode_failed,
                image_hash_mismatch). The caller inspects ``response.json()``
                for the ``code`` and branches.
        """
        digest = sha256 or hashlib.sha256(image_bytes).hexdigest()
        body = ModalityRequest(
            request_id=request_id,
            image=ImagePayload(
                sha256=digest,
                data_b64=base64.b64encode(image_bytes).decode("ascii"),
            ),
        ).model_dump()
        data = await self._request_json(
            "POST", "/v1/classify_modality", json=body, request_id=request_id
        )
        return ModalityResponse.model_validate(data)

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
            raise MedicalClipUnreachableError(
                f"medical-clip server unreachable at {self._base_url}: {exc!s}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise MedicalClipUnreachableError(
                f"medical-clip server timeout at {self._base_url}{path}: {exc!s}"
            ) from exc
        except httpx.HTTPError as exc:
            raise MedicalClipUnreachableError(
                f"medical-clip transport error at {self._base_url}{path}: {exc!s}"
            ) from exc
        if response.status_code >= 500:
            raise MedicalClipUnreachableError(
                f"medical-clip server {response.status_code} at {path}: "
                f"{response.text[:200]}"
            )
        if response.status_code >= 400:
            # 400 image_decode_failed / image_hash_mismatch are *expected*
            # application errors — propagate as httpx.HTTPStatusError so
            # the OCR worker can branch on the `error.code` field.
            response.raise_for_status()
        return response.json()
