"""``CsrfMiddleware`` — double-submit cookie pattern with rotate-on-safe-method.

Rotation strategy: a fresh ``csrf_token`` cookie is set on **every**
safe-method response (GET / HEAD / OPTIONS). The plan considered
HMAC-binding the csrf to the JWT ``sub`` but settled on
per-safe-method rotation because:

1. It defeats pre-login session fixation (Lax cross-site navigation can
   pre-seed a cookie, but per-response rotation invalidates it before
   any state-changing request can carry it).
2. The cost is one ``secrets.token_hex`` call per safe response, which
   is negligible.

Validation on state-changing methods: read ``X-CSRF-Token`` header,
compare byte-for-byte to the ``csrf_token`` cookie. Mismatch / missing
→ 403 + ``audit_event("web.csrf.blocked")``.

The exempt allowlist (:data:`EXEMPT_PATHS`) covers anonymous-by-design
endpoints: ``/auth/login`` and ``/auth/logout``. There is no session
to fix at the login step, and requiring a freshly-rotated csrf for
logout would create a chicken-and-egg with expired sessions. Adding
to this allowlist requires explicit code review — no wildcards.
"""

from __future__ import annotations

import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from claritymed.core.observability.audit import audit_event

COOKIE_CSRF_TOKEN = "csrf_token"
HEADER_CSRF_TOKEN = "X-CSRF-Token"
CSRF_TOKEN_BYTES = 32

SAFE_METHODS: frozenset[str] = frozenset({"GET", "HEAD", "OPTIONS"})

# Explicit allowlist. Adding entries requires an explicit code review;
# never expand by wildcard. ``/auth/login`` and ``/auth/logout`` are
# anonymous-by-design (no session to fix) — the rotate-on-safe-method
# behaviour establishes the cookie/header pair for subsequent
# authenticated mutations.
EXEMPT_PATHS: frozenset[str] = frozenset(
    {
        "/auth/login",
        "/auth/logout",
    }
)


def _new_csrf_value() -> str:
    """Cryptographically-random token, hex-encoded for cookie-safe ASCII."""
    return secrets.token_hex(CSRF_TOKEN_BYTES)


class CsrfMiddleware(BaseHTTPMiddleware):
    """Double-submit CSRF with rotation on every safe-method response."""

    def __init__(self, app: ASGIApp, secure_cookie: bool = True) -> None:
        super().__init__(app)
        self._secure = secure_cookie

    def _set_cookie(self, response: Response, value: str) -> None:
        response.set_cookie(
            COOKIE_CSRF_TOKEN,
            value,
            samesite="strict",
            path="/",
            httponly=False,
            secure=self._secure,
        )

    async def dispatch(self, request: Request, call_next) -> Response:
        method = request.method.upper()

        if method in SAFE_METHODS:
            response = await call_next(request)
            self._set_cookie(response, _new_csrf_value())
            return response

        if request.url.path in EXEMPT_PATHS:
            response = await call_next(request)
            # Even on exempt paths, refresh the csrf cookie on the
            # response so the next authenticated mutation has a fresh
            # pair to send.
            self._set_cookie(response, _new_csrf_value())
            return response

        cookie_token = request.cookies.get(COOKIE_CSRF_TOKEN)
        header_token = request.headers.get(HEADER_CSRF_TOKEN)
        if (
            not cookie_token
            or not header_token
            or not secrets.compare_digest(cookie_token, header_token)
        ):
            audit_event(
                "web.csrf.blocked",
                payload={
                    "path": request.url.path,
                    "method": method,
                },
            )
            return JSONResponse(
                {"detail": "CSRF validation failed"},
                status_code=403,
            )

        response = await call_next(request)
        # Rotate after successful mutation too — keeps a single token
        # from being reused across many mutations.
        self._set_cookie(response, _new_csrf_value())
        return response
