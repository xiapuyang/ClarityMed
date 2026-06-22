"""``WebContextMiddleware`` — JWT-cookie context + anonymous downgrade.

Distinct from ``core/observability/middleware.py:ContextMiddleware``
which trusts the ``X-User-Id`` header (used by internal server-to-server
traffic on the inference servers). For public web that trust model is
unsafe — the browser must authenticate via cookie.

Behaviour per JWT decode result:

* :class:`~claritymed.web.jwt.Valid` — apply_context with the claim's
  ``sub`` and ``lang``.
* :class:`~claritymed.web.jwt.Tamper` — fail loud: clear the cookie,
  emit ``web.jwt.tamper_suspected``, return 401. A valid HMAC with a
  schema violation implies secret compromise or a buggy issuer;
  silently downgrading would mask the signal as "session expired".
* :class:`~claritymed.web.jwt.Invalid` — silent downgrade to anonymous,
  emit ``web.jwt.invalid`` only when a cookie was actually presented.
* No cookie — silent downgrade to anonymous, no audit.

ContextVars are ALWAYS set (even for anonymous requests with
``user_id="__anonymous__"``) so handlers can call ``audit_event()``
without risking ``MissingContextError``.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from claritymed.context import apply_context, new_request_id, reset_context
from claritymed.core.observability.audit import audit_event
from claritymed.core.observability.request_scope import request_scope
from claritymed.web.jwt import (
    Invalid,
    Tamper,
    Valid,
    decode_token,
)

# Sentinel user id used when no valid JWT cookie is presented. Route
# handlers' ``Depends(get_current_user)`` rejects this before any
# AccountStore call — see ``web/deps.py``.
ANONYMOUS_USER_ID = "__anonymous__"

COOKIE_ACCESS_TOKEN = "access_token"
HEADER_REQUEST_ID = "X-Request-ID"

# Paths excluded from ``request_scope`` so audit.log isn't flooded by
# infrastructure traffic. Kubernetes-style liveness probes hit /health
# every few seconds; browsers issue an OPTIONS preflight before every
# cross-origin POST. Neither carries user-meaningful state.
_AUDIT_SKIP_PATHS = frozenset({"/health"})


class WebContextMiddleware(BaseHTTPMiddleware):
    """Set request_id / user_id / language ContextVars from the JWT cookie."""

    def __init__(self, app: ASGIApp, default_lang: str = "en") -> None:
        super().__init__(app)
        self._default_lang = default_lang

    async def dispatch(self, request: Request, call_next) -> Response:
        rid = new_request_id()
        cookie = request.cookies.get(COOKIE_ACCESS_TOKEN)
        user_id = ANONYMOUS_USER_ID
        lang = self._default_lang
        decode_result = decode_token(cookie) if cookie else None

        if isinstance(decode_result, Valid):
            user_id = decode_result.claims["sub"]
            lang = decode_result.claims["lang"]
        elif isinstance(decode_result, Tamper):
            # Set anonymous context so the audit emit + response build
            # below have ContextVars populated.
            pass

        tokens = apply_context(rid, user_id, lang)
        try:
            if isinstance(decode_result, Tamper):
                audit_event(
                    "web.jwt.tamper_suspected",
                    payload={"reason": decode_result.reason},
                )
                resp = JSONResponse(
                    {"detail": "Invalid token"},
                    status_code=401,
                )
                resp.delete_cookie(COOKIE_ACCESS_TOKEN, path="/")
                resp.headers[HEADER_REQUEST_ID] = rid
                return resp

            if isinstance(decode_result, Invalid):
                # Silent downgrade — but record so post-hoc analysis can
                # distinguish "session expired" from "no cookie at all".
                audit_event(
                    "web.jwt.invalid",
                    payload={"reason": decode_result.reason},
                )

            # Bracket the actual handler in request_scope so audit.log
            # gets the same paired start/end emit as CLI entries. Skip
            # liveness probes + CORS preflight so the log isn't drowned
            # in infrastructure traffic. Sync ``with`` around ``await``
            # is intentional — request_scope is sync by design.
            method = request.method
            path = request.url.path
            if method == "OPTIONS" or path in _AUDIT_SKIP_PATHS:
                response = await call_next(request)
            else:
                with request_scope("web", method=method, path=path):
                    response = await call_next(request)
            response.headers[HEADER_REQUEST_ID] = rid
            return response
        finally:
            reset_context(tokens)
