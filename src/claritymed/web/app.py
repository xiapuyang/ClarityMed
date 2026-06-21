"""FastAPI app entry point — middleware chain, OpenAPI gating, serve().

Middleware order (outermost → innermost, i.e. last ``add_middleware``
call wins outer position):

* ``CORSMiddleware`` (outermost) — empty allowlist by default; honours
  ``CLARITYMED_CORS_ORIGINS`` and adds ``http://localhost:5173`` when
  ``CLARITYMED_DEV=1``. Wildcard ``*`` is rejected at startup because
  browsers reject it together with credentials.
* :class:`~claritymed.web.middleware.WebContextMiddleware` — JWT cookie
  decode + ContextVar plumbing. Anonymous downgrade or fail-loud
  tamper response, depending on the decoded result.
* :class:`~claritymed.web.csrf.CsrfMiddleware` — rotate-on-safe-method
  + double-submit validation on mutations (exempt allowlist for
  ``/auth/login`` and ``/auth/logout``).

OpenAPI gating: the FastAPI constructor disables the default routes
(``openapi_url=None``, ``docs_url=None``, ``redoc_url=None``). Custom
endpoints at ``/openapi.json`` and ``/docs`` return 404 in production
unless the request authenticates as ``role == "admin"``. With
``CLARITYMED_DEV=1`` both routes are open unconditionally.

``serve()`` boots uvicorn with ``--access-log`` disabled in production
(``CLARITYMED_DEV`` unset) — belt-and-suspenders against an endpoint
that might leak sensitive query params in a later iteration. The chat
endpoint already keeps PHI off the URL by using POST + body.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from starlette.responses import JSONResponse, Response

from claritymed import config as _cfg
from claritymed.core.observability.audit import audit_event
from claritymed.errors import (
    InvalidUserIdError,
    PermissionDeniedError,
    UnknownProviderError,
    UserNotFoundError,
)
from claritymed.web.csrf import CsrfMiddleware
from claritymed.web.deps import require_admin
from claritymed.web.jwt import validate_secret_or_raise
from claritymed.web.middleware import WebContextMiddleware
from claritymed.web.routers.auth import router as auth_router
from claritymed.web.routers.chat import (
    build_default_ask_service,
    router as chat_router,
)
from claritymed.web.routers.me import router as me_router

ENV_DEV = "CLARITYMED_DEV"
ENV_CORS_ORIGINS = "CLARITYMED_CORS_ORIGINS"
DEV_FRONTEND_ORIGIN = "http://localhost:5173"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

logger = logging.getLogger(__name__)


def _is_dev() -> bool:
    """True when ``CLARITYMED_DEV=1`` (or any non-empty value)."""
    return bool(os.environ.get(ENV_DEV, "").strip())


def _resolve_cors_origins(dev: bool) -> list[str]:
    """Build the CORS allowlist from env + dev injection.

    Wildcard ``*`` raises — browsers reject ``*`` with credentials and
    the resulting failure mode is a confusing CORS-blocked request that
    looks like a network problem. Catching it at startup is far cheaper.
    """
    raw = os.environ.get(ENV_CORS_ORIGINS, "").strip()
    origins = [o.strip() for o in raw.split(",") if o.strip()] if raw else []
    if "*" in origins:
        raise RuntimeError(
            f"{ENV_CORS_ORIGINS} must not include '*' — wildcards are "
            "incompatible with allow_credentials=True. Use an explicit "
            "comma-separated list."
        )
    if dev and DEV_FRONTEND_ORIGIN not in origins:
        origins.append(DEV_FRONTEND_ORIGIN)
    return origins


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001 — FastAPI signature
    """Startup gate: JWT secret + default lang + state dict scaffolding.

    AskService construction is deferred to Unit 4 where the chat router
    actually needs it — building one ``AskService`` per provider on
    every test app startup would be prohibitively heavy. The state
    dicts (``ask_services``, ``session_locks``) are initialized here so
    later units can populate them without touching this lifespan.
    """
    validate_secret_or_raise()
    app.state.default_lang = _cfg.default_lang()
    app.state.dev = _is_dev()
    # Per-session busy set — synchronous check-and-add in the chat
    # route handler keeps the 409-on-concurrency decision atomic in
    # the single-threaded asyncio loop. asyncio.Lock would force the
    # acquire INSIDE the streaming generator, which is too late.
    app.state.busy_sessions = set()
    # ``ask_service_factory`` is overridable per-app (tests inject a
    # fake before issuing chat requests). Default is the catalog-driven
    # builder defined in the chat router.
    if not getattr(app.state, "ask_service_factory", None):
        app.state.ask_service_factory = build_default_ask_service
    logger.info(
        "web.app.lifespan ready dev=%s default_lang=%s",
        app.state.dev,
        app.state.default_lang,
    )
    yield
    app.state.busy_sessions.clear()


def create_app() -> FastAPI:
    """Factory used by both ``serve()`` and the test suite."""
    dev = _is_dev()
    cors_origins = _resolve_cors_origins(dev)

    app = FastAPI(
        title="ClarityMed Web",
        version="1.0.0",
        lifespan=lifespan,
        # Default OpenAPI / docs routes disabled; we re-register them
        # behind ``require_admin`` below so the OpenAPI surface is not
        # discoverable in production by unauthenticated visitors.
        openapi_url=None,
        docs_url=None,
        redoc_url=None,
    )

    # Add middleware in INNER → OUTER order (Starlette wraps in reverse
    # of add_middleware call order). We want: CORS (outermost) →
    # WebContext → Csrf (innermost). So add Csrf first, then
    # WebContext, then CORS.
    app.add_middleware(CsrfMiddleware, secure_cookie=not dev)
    app.add_middleware(WebContextMiddleware, default_lang=_cfg.default_lang())
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["X-CSRF-Token", "Content-Type", "Authorization"],
    )

    _register_exception_handlers(app)
    _register_health(app)
    _register_openapi_routes(app)

    # Business routers — auth lives at /auth/* (unversioned); v1 data
    # routers go under /api/v1.
    app.include_router(auth_router)
    app.include_router(me_router)
    app.include_router(chat_router)

    return app


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(PermissionDeniedError)
    async def _on_permission_denied(_request: Request, exc: PermissionDeniedError):
        return JSONResponse({"detail": str(exc)}, status_code=403)

    @app.exception_handler(UnknownProviderError)
    async def _on_unknown_provider(_request: Request, exc: UnknownProviderError):
        # Configuration error, not client error — surface as 500 so the
        # operator notices the broken catalog. Body carries no PHI.
        logger.error("UnknownProviderError: %s", exc)
        return JSONResponse({"detail": "Configuration error"}, status_code=500)

    @app.exception_handler(UserNotFoundError)
    async def _on_user_not_found(_request: Request, _exc: UserNotFoundError):
        # Treat as anonymous; don't leak account existence.
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)

    @app.exception_handler(InvalidUserIdError)
    async def _on_invalid_user_id(_request: Request, _exc: InvalidUserIdError):
        return JSONResponse({"detail": "Invalid request"}, status_code=400)


def _register_health(app: FastAPI) -> None:
    @app.get("/health")
    async def _health() -> dict[str, str]:
        """Liveness probe. No auth, no CSRF check (it's a GET).

        Future Tauri / Capacitor wrappers use this to confirm the
        backend is reachable before showing the login screen.
        """
        return {"status": "ok"}


def _register_openapi_routes(app: FastAPI) -> None:
    """Re-register /openapi.json and /docs behind dev-or-admin gating.

    Production mode returns 404 (not 401) on denied access so the
    routes' existence is not revealed to unauthenticated probes. Dev
    mode (``CLARITYMED_DEV=1``) opens both routes unconditionally for
    local development convenience.
    """

    async def _openapi_response() -> Any:
        return get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )

    @app.get("/openapi.json", include_in_schema=False)
    async def _openapi(request: Request) -> Response:
        if _is_dev():
            return JSONResponse(await _openapi_response())
        try:
            await require_admin()
        except Exception:
            audit_event(
                "web.openapi.access_blocked",
                payload={"path": "/openapi.json", "reason": "not_admin"},
            )
            return JSONResponse({"detail": "Not found"}, status_code=404)
        return JSONResponse(await _openapi_response())

    @app.get("/docs", include_in_schema=False)
    async def _docs(request: Request) -> Response:
        if _is_dev():
            return get_swagger_ui_html(
                openapi_url="/openapi.json",
                title=f"{app.title} — Swagger UI",
            )
        try:
            await require_admin()
        except Exception:
            audit_event(
                "web.openapi.access_blocked",
                payload={"path": "/docs", "reason": "not_admin"},
            )
            return JSONResponse({"detail": "Not found"}, status_code=404)
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title=f"{app.title} — Swagger UI",
        )


def serve() -> None:
    """``claritymed-web`` console-script entry point.

    Production posture: ``--access-log`` disabled so the uvicorn access
    log doesn't capture sensitive query strings on future endpoints.
    Dev keeps it on for visibility.
    """
    import uvicorn

    host = os.environ.get("CLARITYMED_WEB_HOST", DEFAULT_HOST)
    port = int(os.environ.get("CLARITYMED_WEB_PORT", DEFAULT_PORT))
    dev = _is_dev()
    uvicorn.run(
        "claritymed.web.app:create_app",
        host=host,
        port=port,
        factory=True,
        log_level="info",
        access_log=dev,
    )
