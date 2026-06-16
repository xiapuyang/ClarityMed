"""Shared utilities for the RAG servers."""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING, Any, Iterable

from claritymed.core.device import resolve_device

if TYPE_CHECKING:
    from fastapi import FastAPI

# Passed as log_config to uvicorn.run() in every server so server logs
# match the main app's ``app.log`` shape (timestamp+ms, [LEVEL],
# [request_id][user_id], relpath:lineno, message). The factory key
# ``()`` lets dictConfig instantiate :class:`ClarityMedFormatter`,
# which reads the request_id / user_id / language ContextVars on every
# emit — the logging middleware sets request_id_ctx per request so
# server log lines pick up the same id the orchestrator audit log used.
LOG_CONFIG: dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "claritymed.core.observability.logging.ClarityMedFormatter",
            # APP_FMT carries the [request_id][user_id] + relpath:lineno
            # shape — omit ``datefmt`` so Python's default produces the
            # millisecond suffix (``2026-06-16 00:19:32,807``) that
            # matches ``app.log``.
            "fmt": (
                "%(asctime)s [%(levelname)s] [%(request_id)s][%(user_id)s] "
                "%(relpath)s:%(lineno)d - %(message)s"
            ),
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

# Cap on logged body characters. Applied AFTER redaction of binary
# fields, so a 50KB base64 image redacted to ~30 chars frees the cap to
# show the rest of the JSON (request_id, disease_id, model_id, etc.) in
# full instead of a useless prefix slice of the base64 blob.
DEFAULT_MAX_BODY_CHARS = 2000

# JSON string fields that always carry base64-ish blobs across the
# server fleet. Listed here so every server gets the same redaction
# behavior without each call site having to repeat the list. Add new
# entries when a server introduces another binary-bearing field name.
DEFAULT_REDACT_FIELDS: tuple[str, ...] = (
    "data_b64",
    "mask_png_b64",
    "saliency_b64",
    "image_b64",
)

# How much of a redacted field's value to keep — enough to eyeball the
# format (PNG header, JPEG header, etc.) without flooding the log.
_REDACT_KEEP_CHARS = 16


def default_device() -> str:
    """Return ``"mps"`` / ``"cuda"`` / ``"cpu"`` based on torch's view."""
    return resolve_device("auto")


def _redact_binary_fields(text: str, fields: Iterable[str]) -> str:
    """Replace each ``"<field>":"<value>"`` with a short ``<value>…<+N more>``.

    Operates on the JSON-as-text directly because the bodies we log are
    already JSON strings; a real json.loads/dumps round-trip would
    reorder keys and lose grep-friendliness. The regex is conservative:
    it matches a quoted string value with no embedded quotes (the b64
    alphabet has none) so it won't accidentally chew through escaped
    quotes inside arbitrary payloads.
    """
    for field in fields:
        pattern = re.compile(rf'("{re.escape(field)}"\s*:\s*")([^"]+)(")')

        def _shrink(match: re.Match[str]) -> str:
            prefix, value, suffix = match.group(1), match.group(2), match.group(3)
            if len(value) <= _REDACT_KEEP_CHARS:
                return match.group(0)
            return (
                f"{prefix}{value[:_REDACT_KEEP_CHARS]}"
                f"…<+{len(value) - _REDACT_KEEP_CHARS} more chars>{suffix}"
            )

        text = pattern.sub(_shrink, text)
    return text


def _summarize_body(
    data: bytes,
    max_chars: int,
    redact_fields: Iterable[str] = DEFAULT_REDACT_FIELDS,
) -> str:
    """Format a body for one-line INFO logging.

    Order of operations: decode UTF-8 → collapse newlines → redact any
    binary fields named in ``redact_fields`` → length-truncate. The
    redaction step runs first so the length cap only kicks in for
    bodies that are genuinely large for non-binary reasons; a 50KB
    base64 image redacts to ~30 chars and the rest of the JSON
    survives.

    Bytes that fail UTF-8 decode (typically multipart/protobuf
    payloads) are summarized as their length only — rendering raw
    bytes in a log line is noise.
    """
    if not data:
        return "<empty>"
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return f"<binary {len(data)} bytes>"
    # Collapse newlines so the log line stays grep-friendly. Tabs stay
    # — they show up rarely in JSON and help readability when they do.
    text = text.replace("\n", "\\n").replace("\r", "")
    text = _redact_binary_fields(text, redact_fields)
    if len(text) > max_chars:
        return f"{text[:max_chars]}…<+{len(text) - max_chars} more chars>"
    return text


def add_logging_middleware(
    app: "FastAPI",
    *,
    server_logger: logging.Logger,
    log_body: bool = True,
    max_body_chars: int = DEFAULT_MAX_BODY_CHARS,
    redact_fields: Iterable[str] = DEFAULT_REDACT_FIELDS,
) -> None:
    """Register a request/response INFO-log middleware on a FastAPI app.

    Logs one line on request arrival and one on response dispatch (both
    at INFO). Skips ``GET /health`` to avoid polling spam.

    Request-ID policy: use the ``X-Request-ID`` header from the client
    when present; otherwise generate a fresh 22-char id via
    :func:`claritymed.context.new_request_id` so server-generated ids
    share the same shape as the orchestrator's. The resolved id is
    echoed in the response ``X-Request-ID`` header so the caller can
    correlate its logs with the server's.

    Context propagation: the resolved request_id (and ``X-User-Id`` /
    ``X-Language`` headers when present, with sensible defaults) is
    pushed onto the three ContextVars via
    :func:`~claritymed.context.apply_context` for the duration of the
    request. The formatter installed by :data:`LOG_CONFIG` reads those
    vars on every emit, so handler-level logs (e.g. vision-server's
    "loaded vision model id=…") automatically pick up the same
    ``[request_id][user_id]`` prefix that ``app.log`` uses.

    Body logging (``log_body=True``, default): the request body is
    drained once and re-attached to the request so downstream handlers
    can still read it; the response body is drained and the buffered
    bytes are returned as a fresh ``Response`` so the logged content
    matches what the client saw. Bodies are truncated to
    ``max_body_chars`` and non-UTF-8 bytes are summarised as
    ``<binary N bytes>`` to keep log lines grep-friendly.

    Set ``log_body=False`` only if a server's body shape is genuinely
    unloggable (e.g. very large multipart streams) — privacy is not the
    knob this gates; PHI is already controlled by the loopback
    constraint on PHI-bearing servers.
    """
    from fastapi import Request
    from starlette.responses import Response

    from claritymed.context import (
        apply_context,
        is_valid_request_id,
        new_request_id,
        reset_context,
    )

    @app.middleware("http")
    async def _log_requests(request: Request, call_next: Any) -> Any:
        # Resolve the request id once so the arrival, departure, and
        # response header all carry the same value. Reject malformed
        # client-supplied ids the same way the orchestrator's API
        # middleware does — a forged header should not poison the trace.
        raw_rid = request.headers.get("X-Request-ID", "").strip()
        req_id = (
            raw_rid if raw_rid and is_valid_request_id(raw_rid) else new_request_id()
        )
        user_id = request.headers.get("X-User-Id", "default")
        language = request.headers.get("X-Language", "-")
        path = request.url.path
        is_health = path == "/health" and request.method == "GET"

        ctx_tokens = apply_context(req_id, user_id, language)
        try:
            return await _dispatch(
                request=request,
                call_next=call_next,
                req_id=req_id,
                path=path,
                is_health=is_health,
            )
        finally:
            reset_context(ctx_tokens)

    async def _dispatch(
        *,
        request: "Request",
        call_next: Any,
        req_id: str,
        path: str,
        is_health: bool,
    ) -> Any:
        """Body capture + arrival/departure logging.

        Split from ``_log_requests`` so the ``apply_context`` /
        ``reset_context`` envelope stays at one indentation level — the
        body machinery dominates the function otherwise.
        """
        # Drain and re-attach the request body so handlers can still
        # read it. ``request.body()`` caches the bytes on the Request,
        # but BaseHTTPMiddleware wraps the underlying receive() channel
        # — handlers that call ``request.body()`` again would otherwise
        # block on a drained stream. Wiring ``_receive`` makes the body
        # replayable for the inner ASGI app.
        req_body_repr: str | None = None
        if log_body and not is_health:
            req_body = await request.body()
            req_body_repr = _summarize_body(req_body, max_body_chars, redact_fields)

            async def _receive():
                return {
                    "type": "http.request",
                    "body": req_body,
                    "more_body": False,
                }

            request._receive = _receive  # type: ignore[attr-defined]

        if not is_health:
            # ``req_id`` is also visible via the ContextVar-driven log
            # prefix, but keeping it inline mirrors the ``rid=…`` style
            # used in ``app.log`` messages and keeps existing greps that
            # filter on ``req_id=`` working.
            if req_body_repr is not None:
                server_logger.info(
                    "→ %s %s req_id=%s body=%s",
                    request.method,
                    path,
                    req_id,
                    req_body_repr,
                )
            else:
                server_logger.info("→ %s %s req_id=%s", request.method, path, req_id)

        t0 = time.monotonic()
        response = await call_next(request)
        elapsed_ms = (time.monotonic() - t0) * 1000
        # Prefer what the route handler already set (body-derived
        # request_id for vision/medical_clip); fall back to our
        # header-derived/generated value.
        resp_req_id = response.headers.get("X-Request-ID") or req_id

        # Drain the streaming response so we can both log it and hand
        # the bytes back to the client. JSONResponse already buffers,
        # so this is cheap; for true streaming endpoints it materialises
        # the stream which is the price of log-body coverage.
        resp_body_repr: str | None = None
        if log_body and not is_health:
            resp_body = b""
            async for chunk in response.body_iterator:
                resp_body += chunk
            resp_body_repr = _summarize_body(resp_body, max_body_chars, redact_fields)
            # Replace the consumed streaming response with a buffered
            # one carrying the same status, headers, and content type.
            # Strip ``content-length`` from the propagated header set;
            # the new Response recomputes it from the buffered bytes.
            propagated_headers = {
                k: v
                for k, v in response.headers.items()
                if k.lower() != "content-length"
            }
            response = Response(
                content=resp_body,
                status_code=response.status_code,
                headers=propagated_headers,
                media_type=response.media_type,
            )

        response.headers["X-Request-ID"] = resp_req_id
        if not is_health:
            if resp_body_repr is not None:
                server_logger.info(
                    "← %s %s req_id=%s status=%d elapsed_ms=%.0f body=%s",
                    request.method,
                    path,
                    resp_req_id,
                    response.status_code,
                    elapsed_ms,
                    resp_body_repr,
                )
            else:
                server_logger.info(
                    "← %s %s req_id=%s status=%d elapsed_ms=%.0f",
                    request.method,
                    path,
                    resp_req_id,
                    response.status_code,
                    elapsed_ms,
                )
        return response
