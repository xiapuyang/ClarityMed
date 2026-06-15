"""Async HTTP client for the symptoms server.

Per CLAUDE.md's service-layer rule, all symptoms-server network I/O
goes through this class — the plugin and the tool body never construct
``httpx.AsyncClient`` directly. If a future replacement (a different
inference backend, a gRPC variant) lands, only this module changes.

Fail-loud contract mirrors :class:`~claritymed.core.rag.embedding.bge_m3.BgeM3HttpEmbedder`:
connection errors, timeouts, and non-2xx HTTP responses raise typed
:class:`~claritymed.errors.SymptomsServerUnreachableError`. The plugin
catches this once at the tool-body top and degrades to free-text.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from claritymed.errors import SymptomsServerUnreachableError
from claritymed.servers.symptoms.wire import (
    CancelResponse,
    HealthResponse,
    StartSessionRequest,
    StartSessionResponse,
    TurnRequest,
    TurnResponse,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:8084"
DEFAULT_CONNECT_TIMEOUT_S = 5.0
DEFAULT_READ_TIMEOUT_S = 30.0
DEFAULT_WRITE_TIMEOUT_S = 10.0


class SymptomsServerClient:
    """Client for the loopback-bound symptoms FastAPI server.

    Construct once per agent run (the plugin holds a singleton). The
    underlying ``httpx.AsyncClient`` pools connections, so repeated
    calls within a sub-session avoid TLS / TCP overhead.

    Tests inject a fake :class:`httpx.MockTransport` via ``transport``
    so the client exercises against a router that returns canned
    responses without ever touching the network.
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
        )
        self._base_url = base_url

    @property
    def base_url(self) -> str:
        return self._base_url

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "SymptomsServerClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        await self.aclose()

    # --- endpoints --------------------------------------------------------

    async def health(self) -> HealthResponse:
        """``GET /health`` — readiness + loaded datasets/models snapshot."""
        data = await self._get_json("/health")
        return HealthResponse.model_validate(data)

    async def start_session(
        self,
        dataset_id: str,
        complaint: str,
        profile: dict[str, Any],
        *,
        language: str = "en",
        symptom_summary: str | None = None,
        request_id: str | None = None,
    ) -> StartSessionResponse:
        """``POST /v1/datasets/<id>/sessions`` — open a new sub-session.

        ``symptom_summary`` is the LLM-distilled clinical chief
        complaint (1-2 sentences, EN) — fed to the server's
        init-symptom matcher to pre-reveal turn-0 evidence. ``None``
        is fine: the server falls back to ``complaint``.
        ``request_id`` is forwarded as ``X-Request-ID`` so server logs
        can be correlated with orchestrator audit events.
        """
        payload = StartSessionRequest(
            complaint=complaint,
            profile=profile,  # type: ignore[arg-type] — pydantic coerces
            language=language,  # type: ignore[arg-type]
            symptom_summary=symptom_summary,
        )
        data = await self._post_json(
            f"/v1/datasets/{dataset_id}/sessions",
            payload.model_dump(),
            request_id=request_id,
        )
        return StartSessionResponse.model_validate(data)

    async def turn(
        self,
        dataset_id: str,
        session_id: str,
        answer: Any,
        *,
        answer_value: str | list[str] | None = None,
        language: str = "en",
        request_id: str | None = None,
    ) -> TurnResponse:
        """``POST /v1/datasets/<id>/sessions/<sid>/turn`` — apply an answer."""
        payload = TurnRequest(
            answer=answer,
            answer_value=answer_value,
            language=language,  # type: ignore[arg-type]
        )
        data = await self._post_json(
            f"/v1/datasets/{dataset_id}/sessions/{session_id}/turn",
            payload.model_dump(),
            request_id=request_id,
        )
        return TurnResponse.model_validate(data)

    async def cancel(
        self,
        dataset_id: str,
        session_id: str,
        *,
        request_id: str | None = None,
    ) -> CancelResponse:
        """``DELETE /v1/datasets/<id>/sessions/<sid>`` — cancel mid-loop."""
        data = await self._request_json(
            "DELETE",
            f"/v1/datasets/{dataset_id}/sessions/{session_id}",
            request_id=request_id,
        )
        return CancelResponse.model_validate(data)

    # --- internals --------------------------------------------------------

    async def _get_json(self, path: str) -> dict:
        return await self._request_json("GET", path)

    async def _post_json(
        self, path: str, body: dict, *, request_id: str | None = None
    ) -> dict:
        return await self._request_json("POST", path, json=body, request_id=request_id)

    async def _request_json(
        self, method: str, path: str, *, request_id: str | None = None, **kwargs: Any
    ) -> dict:
        if request_id:
            headers = kwargs.pop("headers", {})
            headers["X-Request-ID"] = request_id
            kwargs["headers"] = headers
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            raise SymptomsServerUnreachableError(
                f"symptoms server unreachable at {self._base_url}: {exc!s}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise SymptomsServerUnreachableError(
                f"symptoms server timeout at {self._base_url}{path}: {exc!s}"
            ) from exc
        except httpx.HTTPError as exc:
            raise SymptomsServerUnreachableError(
                f"symptoms server transport error at {self._base_url}{path}: {exc!s}"
            ) from exc
        if response.status_code >= 500:
            raise SymptomsServerUnreachableError(
                f"symptoms server {response.status_code} at {path}: "
                f"{response.text[:200]}"
            )
        if response.status_code >= 400:
            # 404 (unknown session, unknown dataset) and 422 (bad answer)
            # are *expected* application-level errors the plugin handles
            # — propagate as httpx.HTTPStatusError so the plugin can
            # inspect status_code and branch.
            response.raise_for_status()
        return response.json()
