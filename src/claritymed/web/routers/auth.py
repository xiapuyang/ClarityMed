"""``/auth/login`` + ``/auth/logout`` — the web layer's only password path.

Login flow:

1. Pydantic validates the request body (``user_id`` regex; ``password``
   minimum length 1). A malformed user_id reaches the response as 422
   without touching the store.
2. ``PasswordStore.verify_password`` returns False for unknown user or
   wrong password; the response body is **identical** in either case
   (no account enumeration). The audit row distinguishes via
   ``user_id_attempted`` + ``ip_hmac``.
3. On success: load the Account, mint a JWT, set the ``access_token``
   cookie (httpOnly + SameSite=Lax + Secure outside dev), and return a
   ``LoginResponse`` so the SPA can render the chrome without an extra
   ``GET /api/v1/me`` call.

Logout flow: clear the ``access_token`` cookie, return 204. The
``csrf_token`` cookie is refreshed by ``CsrfMiddleware`` on the
response (every safe-method response rotates, plus the exempt-path
branch in the middleware refreshes too — see ``web/csrf.py``).

Both endpoints are in the CSRF exempt allowlist (``EXEMPT_PATHS`` in
``web/csrf.py``) — login because there is no session to fix at the
anonymous boundary, logout to avoid a chicken-and-egg with expired
sessions.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, Response
from starlette.status import HTTP_204_NO_CONTENT

from claritymed.context import user_id_ctx
from claritymed.core.observability.audit import audit_event
from claritymed.stores.account import AccountStore
from claritymed.stores.auth import PasswordStore
from claritymed.web.jwt import create_token, hmac_ip
from claritymed.web.middleware import ANONYMOUS_USER_ID, COOKIE_ACCESS_TOKEN
from claritymed.web.schemas import LoginRequest, LoginResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# Cookie lifetime in seconds — matches the default JWT exp of 7 days so
# the browser drops the cookie around the same time the token would
# fail to decode anyway.
_COOKIE_MAX_AGE_S = 7 * 24 * 60 * 60


@router.post(
    "/login",
    response_model=LoginResponse,
    responses={401: {"description": "Invalid credentials"}},
)
async def login(
    req: LoginRequest, request: Request, response: Response
) -> LoginResponse:
    """Verify credentials, mint a JWT, set the httpOnly cookie."""
    # Capture the client IP for forensics BEFORE any branching so the
    # failure path is identical to the success path's bookkeeping.
    client_host = request.client.host if request.client else "-"

    if not PasswordStore.verify_password(req.user_id, req.password):
        # Same response shape for unknown user vs wrong password — no
        # account enumeration. The audit distinguishes via ip_hmac so
        # post-hoc clustering of login spray is still possible.
        audit_event(
            "web.auth.login_failed",
            payload={
                "user_id_attempted": req.user_id,
                "ip_hmac": hmac_ip(client_host),
            },
        )
        raise HTTPException(status_code=401, detail="Invalid credentials")

    store = AccountStore(req.user_id)
    if not store.exists():
        # PasswordStore.verify_password returned True but the Account
        # YAML is missing — internal inconsistency. Surface as a
        # generic 401 to avoid leaking the discrepancy.
        logger.error(
            "login: verify_password passed for %r but no settings.yaml",
            req.user_id,
        )
        audit_event(
            "web.auth.login_failed",
            payload={
                "user_id_attempted": req.user_id,
                "ip_hmac": hmac_ip(client_host),
            },
        )
        raise HTTPException(status_code=401, detail="Invalid credentials")

    account = store.load()
    token = create_token(account.user_id, account.language)
    response.set_cookie(
        COOKIE_ACCESS_TOKEN,
        token,
        max_age=_COOKIE_MAX_AGE_S,
        httponly=True,
        samesite="lax",
        secure=_secure_cookie(request),
        path="/",
    )
    audit_event(
        "web.auth.login_success",
        payload={"user_id": account.user_id},
    )
    return LoginResponse(
        user_id=account.user_id,
        display_name=account.display_name,
        role=account.role,
        language=account.language,
    )


@router.post("/logout", status_code=HTTP_204_NO_CONTENT)
async def logout(request: Request) -> Response:
    """Clear the access_token cookie. CSRF-exempt by design.

    The current user_id (if any) is captured from ContextVars BEFORE
    the cookie is cleared — gives the audit row something more useful
    than ``__anonymous__`` when a logged-in user logs out. The response
    is built directly (rather than using a ``response: Response``
    parameter) so the cookie deletion lands on the response FastAPI
    actually returns to the client.
    """
    uid = user_id_ctx.get() or ANONYMOUS_USER_ID
    resp = Response(status_code=HTTP_204_NO_CONTENT)
    resp.delete_cookie(
        COOKIE_ACCESS_TOKEN,
        path="/",
        samesite="lax",
        secure=_secure_cookie(request),
        httponly=True,
    )
    audit_event("web.auth.logout", payload={"user_id": uid})
    return resp


def _secure_cookie(request: Request) -> bool:
    """Match the app's dev / prod cookie secure-flag policy.

    The app sets ``request.app.state.dev`` in lifespan; outside dev
    (i.e. production), all auth cookies must carry ``Secure`` so the
    browser refuses to send them over plaintext HTTP.
    """
    return not getattr(request.app.state, "dev", False)
