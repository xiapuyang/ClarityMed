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

import asyncio
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
from claritymed.bootstrap import bootstrap_once, prefetch_models
from claritymed.core.observability.audit import audit_event
from claritymed.core.observability.tracing import setup_tracing
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
from claritymed.web.admin.jobs import JobRegistry
from claritymed.web.routers.admin import router as admin_router
from claritymed.web.routers.auth import router as auth_router
from claritymed.web.routers.chat import (
    build_default_ask_service,
    router as chat_router,
)
from claritymed.web.routers.attachments import router as attachments_router
from claritymed.web.routers.library import router as library_router
from claritymed.web.routers.me import router as me_router
from claritymed.web.routers.providers import router as providers_router

ENV_DEV = "CLARITYMED_DEV"
ENV_CORS_ORIGINS = "CLARITYMED_CORS_ORIGINS"
DEV_FRONTEND_ORIGIN = "http://localhost:5173"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8120

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
    # Shared host bootstrap — same code path as the CLI's Typer callback
    # (see ``cli/common.py``). Installs app/access/audit/llm file
    # handlers and silences noisy library loggers (httpx, urllib3, …).
    # Idempotent; safe across test app re-creation.
    bootstrap_once(script_name="claritymed-web")
    # Pre-download the PHI scrubber's HF model if ``privacy_filter.enabled``
    # — cloud-bound requests would crash on first turn otherwise. No-op when
    # the privacy filter is off. ``SystemExit(1)`` on failure: surfacing the
    # missing-deps error during startup beats serving 500s.
    prefetch_models()
    # Install OTel tracing alongside CLI/TUI hosts. ``setup_tracing``
    # is idempotent and reads ``tracing.enabled`` from ``configs/app.yaml``
    # — disabled paths short-circuit before any exporter touches the
    # network, so the web worker stays no-op when tracing is off.
    setup_tracing()
    app.state.default_lang = _cfg.default_lang()
    app.state.dev = _is_dev()
    # Per-session busy set — synchronous check-and-add in the chat
    # route handler keeps the 409-on-concurrency decision atomic in
    # the single-threaded asyncio loop. asyncio.Lock would force the
    # acquire INSIDE the streaming generator, which is too late.
    app.state.busy_sessions = set()
    # Process-wide RAG strategy cache. Sentinel ``_RAG_UNSET`` means
    # "not yet built"; the chat router's first request acquires
    # ``rag_strategy_lock`` and stores either a :class:`RagStrategy`
    # or ``None`` (when ``rag.enabled=false`` in ``retrieval.yaml``).
    # ``None`` is a valid cached value — hence the sentinel. The
    # underlying qdrant clients are released by Python GC at process
    # exit, matching the TUI's lifetime semantics.
    from claritymed.web.routers.chat import RAG_UNSET

    app.state.rag_strategy = RAG_UNSET
    app.state.rag_strategy_lock = asyncio.Lock()
    # Per-app in-memory rendezvous for ask_user_question / tool_approval
    # interactions. Key = interaction_id, value = {session_id, user_id,
    # kind, future, ...}. Populated by ``web/channels.py``; resolved by
    # ``POST /api/v1/sessions/{id}/interactions/{interaction_id}``.
    app.state.web_interactions = {}
    # ``ask_service_factory`` is overridable per-app (tests inject a
    # fake before issuing chat requests). Default is the catalog-driven
    # builder defined in the chat router.
    if not getattr(app.state, "ask_service_factory", None):
        app.state.ask_service_factory = build_default_ask_service
    # Admin jobs registry — single instance per app. ``recover_on_startup``
    # marks any spec stuck in ``running`` as ``crashed`` and emits the
    # matching audit event. Runners are wired by U7 (rag_*) and U8
    # (benchmark_run) so a submit before those land returns a clean 400.
    app.state.jobs = JobRegistry()
    _register_job_runners(app.state.jobs)
    app.state.jobs.recover_on_startup()
    logger.info(
        "web.app.lifespan ready dev=%s default_lang=%s",
        app.state.dev,
        app.state.default_lang,
    )
    yield
    app.state.busy_sessions.clear()
    # Cancel any still-pending interaction futures so listeners awaiting
    # them get a clean error instead of hanging on shutdown.
    for rec in list(app.state.web_interactions.values()):
        fut = rec.get("future")
        if fut is not None and not fut.done():
            fut.cancel()
    app.state.web_interactions.clear()
    # Drain the OCR worker so a torn-down uvicorn loop doesn't leave a
    # background task pointing at a dead provider chain.
    worker = getattr(app.state, "ocr_worker", None)
    if worker is not None:
        try:
            await worker.stop()
        except Exception:  # noqa: BLE001
            logger.exception("ocr worker stop failed during lifespan shutdown")
        app.state.ocr_worker = None


def create_app() -> FastAPI:
    """Factory used by both ``serve()`` and the test suite."""
    # Mirror CLI ``bootstrap_once``: load ``~/.claritymed/.env`` so provider
    # API keys (OMLX_API_KEY, OPENAI_API_KEY, ...) are visible to the
    # uvicorn worker. Idempotent — keys already in os.environ win.
    _cfg.load_env_file()
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
    app.include_router(providers_router)
    app.include_router(attachments_router)
    app.include_router(library_router)
    app.include_router(chat_router)
    app.include_router(admin_router)

    _mount_admin_ui(app)

    return app


def _register_job_runners(registry: JobRegistry) -> None:
    """Wire admin job kinds to their runner callables.

    U7 supplies rag_ingest + rag_bootstrap; U8 supplies benchmark_run.
    Importing the runners lazily avoids forcing the runner modules to
    load at every web import — useful when running tests that don't
    touch RAG or benchmarks.
    """
    from claritymed.web.admin.job_runners import (
        benchmark_run,
        rag_bootstrap,
        rag_ingest,
    )

    registry.register_runner("rag_ingest", rag_ingest.run)
    registry.register_runner("rag_bootstrap", rag_bootstrap.run)
    registry.register_runner("benchmark_run", benchmark_run.run)


def _mount_admin_ui(app: FastAPI) -> None:
    """Mount the admin SPA at ``/admin/*`` if its build is on disk.

    The Vite build target is ``src/claritymed/web/admin_ui/dist/``. When
    the directory is missing we log a warning at startup — the API at
    ``/api/v1/admin/*`` still works (so curl/automation can drive admin
    without the SPA), but operators visiting ``/admin/`` get a 404. The
    one-time fix is ``make admin-ui``.

    ``html=True`` tells StaticFiles to serve ``index.html`` for any path
    that doesn't resolve to a file, which is exactly the client-side
    routing behaviour React-Router needs (``/admin/users``, etc., all
    fall back to the SPA bundle).
    """
    from pathlib import Path

    from fastapi.staticfiles import StaticFiles

    dist_dir = Path(__file__).parent / "admin_ui" / "dist"
    if not dist_dir.exists():
        logger.warning(
            "admin_ui dist missing at %s — run `make admin-ui` to enable the "
            "admin SPA; the /api/v1/admin/* API still works.",
            dist_dir,
        )
        return
    app.mount("/admin", StaticFiles(directory=dist_dir, html=True), name="admin_ui")


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
