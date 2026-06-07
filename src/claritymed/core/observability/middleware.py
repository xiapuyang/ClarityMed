"""Starlette middleware that injects the three request ContextVars.

Reads ``X-Request-ID`` / ``X-User-Id`` / ``X-Language`` headers; falls back to
``new_request_id()`` / ``"default"`` / ``app.yaml.default_lang``. Validates an
incoming ``X-Request-ID`` with ``is_valid_request_id`` so a malformed header
cannot poison the audit trail — rejected ids are replaced and noted in the
``request_start`` event payload.

The real router wiring (FastAPI app + routes) lives in the api plan; this is
the skeleton the api plan will mount.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from claritymed import config as _cfg
from claritymed.context import (
    apply_context,
    is_valid_request_id,
    new_request_id,
    reset_context,
)
from claritymed.core.observability.audit import audit_event
from claritymed.core.observability.logging import get_access_logger

DEFAULT_USER_ID = "default"
LANGUAGE_HEADERS = ("x-language",)


def _pick_language(request: Request) -> str:
    for header in LANGUAGE_HEADERS:
        raw = request.headers.get(header)
        if raw:
            lang = raw.strip().lower()
            if lang in ("en", "zh"):
                return lang
    return _cfg.default_lang()


def _pick_request_id(request: Request) -> tuple[str, bool]:
    """Return (id, was_rejected). ``was_rejected`` is True if a header was
    present but failed validation — the id we return is then a fresh one."""
    raw = request.headers.get("x-request-id", "").strip()
    if not raw:
        return new_request_id(), False
    if is_valid_request_id(raw):
        return raw, False
    return new_request_id(), True


def _should_skip(request: Request) -> bool:
    skip_prefixes = (
        _cfg.load_yaml("app.yaml").get("access_log", {}).get("skip_prefixes", [])
    )
    return any(request.url.path.startswith(p) for p in skip_prefixes)


class ContextMiddleware(BaseHTTPMiddleware):
    """Apply the three ContextVars for the duration of the request."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self.access_logger = get_access_logger()

    async def dispatch(self, request: Request, call_next) -> Response:
        if _should_skip(request):
            return await call_next(request)

        rid, rejected = _pick_request_id(request)
        uid = request.headers.get("x-user-id", DEFAULT_USER_ID)
        lang = _pick_language(request)
        tokens = apply_context(rid, uid, lang)

        try:
            audit_event(
                "request_start",
                payload={
                    "entry": "api",
                    "method": request.method,
                    "path": request.url.path,
                    "x_request_id_rejected": rejected,
                },
            )
            self.access_logger.info("START %s %s", request.method, request.url.path)
            try:
                response = await call_next(request)
            except Exception:
                audit_event("request_end", payload={"status": "exception"})
                self.access_logger.exception(
                    "ERROR %s %s", request.method, request.url.path
                )
                raise

            response.headers["X-Request-ID"] = rid
            audit_event("request_end", payload={"status": response.status_code})
            self.access_logger.info(
                "EXIT %s %s status=%s",
                request.method,
                request.url.path,
                response.status_code,
            )
            return response
        finally:
            reset_context(tokens)
