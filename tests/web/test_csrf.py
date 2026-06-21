"""CsrfMiddleware: rotate-on-safe-method + double-submit validation + exempt."""

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from claritymed import config as _cfg
from claritymed.web.csrf import (
    COOKIE_CSRF_TOKEN,
    EXEMPT_PATHS,
    HEADER_CSRF_TOKEN,
    CsrfMiddleware,
)
from claritymed.web.middleware import WebContextMiddleware


def _csrf_app(*, secure_cookie: bool = False) -> FastAPI:
    """Tiny app with WebContext+CSRF wired and one POST + one GET endpoint."""
    app = FastAPI()
    # WebContext sets ContextVars so audit_event inside csrf doesn't blow up.
    app.add_middleware(CsrfMiddleware, secure_cookie=secure_cookie)
    app.add_middleware(WebContextMiddleware, default_lang=_cfg.default_lang())

    @app.get("/safe")
    async def safe():
        return {"ok": True}

    @app.post("/mutate")
    async def mutate():
        return {"mutated": True}

    @app.post("/auth/login")
    async def login():
        return {"login": True}

    @app.post("/auth/logout")
    async def logout():
        return {"logout": True}

    return app


@pytest.fixture
async def csrf_client(jwt_secret):  # noqa: ARG001
    app = _csrf_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


def _extract_csrf_cookie(resp) -> str | None:
    for raw in resp.headers.get_list("set-cookie"):
        if raw.startswith(f"{COOKIE_CSRF_TOKEN}="):
            return raw.split(";", 1)[0].split("=", 1)[1]
    return None


async def test_safe_method_response_sets_fresh_csrf_cookie(csrf_client):
    resp = await csrf_client.get("/safe")
    assert resp.status_code == 200
    cookie_val = _extract_csrf_cookie(resp)
    assert cookie_val is not None
    # 32-byte token hex = 64 chars.
    assert len(cookie_val) == 64


async def test_csrf_rotates_between_consecutive_safe_calls(csrf_client):
    r1 = await csrf_client.get("/safe")
    r2 = await csrf_client.get("/safe")
    v1 = _extract_csrf_cookie(r1)
    v2 = _extract_csrf_cookie(r2)
    assert v1 != v2


async def test_mutation_without_header_blocked(csrf_client):
    # Even with a cookie, missing X-CSRF-Token header → 403.
    resp = await csrf_client.post(
        "/mutate",
        cookies={COOKIE_CSRF_TOKEN: "abc"},
    )
    assert resp.status_code == 403


async def test_mutation_with_mismatched_header_blocked(csrf_client):
    resp = await csrf_client.post(
        "/mutate",
        cookies={COOKIE_CSRF_TOKEN: "abc"},
        headers={HEADER_CSRF_TOKEN: "def"},
    )
    assert resp.status_code == 403


async def test_mutation_with_matching_pair_passes(csrf_client):
    resp = await csrf_client.post(
        "/mutate",
        cookies={COOKIE_CSRF_TOKEN: "matching-token"},
        headers={HEADER_CSRF_TOKEN: "matching-token"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"mutated": True}


async def test_mutation_blocked_emits_audit(csrf_client, caplog):
    with caplog.at_level(logging.INFO):
        await csrf_client.post(
            "/mutate",
            cookies={COOKIE_CSRF_TOKEN: "abc"},
            headers={HEADER_CSRF_TOKEN: "def"},
        )
    assert any("web.csrf.blocked" in rec.message for rec in caplog.records), [
        rec.message for rec in caplog.records
    ]


async def test_login_is_exempt(csrf_client):
    resp = await csrf_client.post("/auth/login")
    assert resp.status_code == 200


async def test_logout_is_exempt(csrf_client):
    resp = await csrf_client.post("/auth/logout")
    assert resp.status_code == 200


async def test_exempt_paths_constant_is_minimal():
    # Belt-and-suspenders: prevent accidental wildcard expansion of the
    # CSRF exempt allowlist. Any addition needs an explicit code review
    # AND this assertion update.
    assert EXEMPT_PATHS == {"/auth/login", "/auth/logout"}


async def test_csrf_cookie_attributes_in_secure_mode(jwt_secret):  # noqa: ARG001
    # In secure_cookie=True mode the Set-Cookie carries Secure and
    # SameSite=Strict; in dev (secure=False) the Secure flag is dropped.
    app = _csrf_app(secure_cookie=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        resp = await c.get("/safe")
    raw = next(
        line
        for line in resp.headers.get_list("set-cookie")
        if line.startswith(f"{COOKIE_CSRF_TOKEN}=")
    )
    assert "Secure" in raw
    assert "samesite=strict" in raw.lower()
    assert "Path=/" in raw
    # non-httpOnly so the client JS can read it.
    assert "HttpOnly" not in raw
