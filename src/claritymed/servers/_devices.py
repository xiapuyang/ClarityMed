"""Shared utilities for the RAG servers."""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from claritymed.core.device import resolve_device

if TYPE_CHECKING:
    from fastapi import FastAPI

# Passed as log_config to uvicorn.run() in every server so that uvicorn's own
# loggers (which default to propagate=False) also emit %(asctime)s timestamps.
LOG_CONFIG: dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stderr",
        },
    },
    "root": {"handlers": ["default"], "level": "INFO"},
    "loggers": {
        "uvicorn": {"propagate": True},
        "uvicorn.error": {"propagate": True},
        "uvicorn.access": {"propagate": True},
    },
}


def default_device() -> str:
    """Return ``"mps"`` / ``"cuda"`` / ``"cpu"`` based on torch's view."""
    return resolve_device("auto")


def add_logging_middleware(app: "FastAPI", *, server_logger: logging.Logger) -> None:
    """Register a request/response INFO-log middleware on a FastAPI app.

    Logs one line on request arrival and one on response dispatch (both at
    INFO). Skips ``GET /health`` to avoid polling spam.

    Request-ID policy: use the ``X-Request-ID`` header from the client when
    present; otherwise generate a short server-side id (``srv-<8 hex chars>``).
    The resolved id is echoed in the response ``X-Request-ID`` header so the
    caller can correlate its logs with the server's.
    """
    from fastapi import Request

    @app.middleware("http")
    async def _log_requests(request: Request, call_next: Any) -> Any:
        # Use the client-supplied header when present; generate a server-side
        # fallback otherwise. Route handlers that read request_id from the body
        # (vision, medical_clip) will overwrite the response header with the
        # body value — we read back from the response header after call_next so
        # the response log always matches what was actually sent to the client.
        req_id = request.headers.get("X-Request-ID") or f"srv-{uuid.uuid4().hex[:8]}"
        path = request.url.path
        is_health = path == "/health" and request.method == "GET"
        if not is_health:
            server_logger.info("→ %s %s req_id=%s", request.method, path, req_id)
        t0 = time.monotonic()
        response = await call_next(request)
        elapsed_ms = (time.monotonic() - t0) * 1000
        # Prefer what the route handler already set (body-derived request_id for
        # vision/medical_clip); fall back to our header-derived/generated value.
        resp_req_id = response.headers.get("X-Request-ID") or req_id
        response.headers["X-Request-ID"] = resp_req_id
        if not is_health:
            server_logger.info(
                "← %s %s req_id=%s status=%d elapsed_ms=%.0f",
                request.method,
                path,
                resp_req_id,
                response.status_code,
                elapsed_ms,
            )
        return response
