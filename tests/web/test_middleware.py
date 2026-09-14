"""WebContextMiddleware: JWT cookie → ContextVars + Tamper / Invalid dispatch."""

from __future__ import annotations

import logging
import time

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from jose import jwt as _jose_jwt

from claritymed import config as _cfg
from claritymed.context import language_ctx, user_id_ctx
from claritymed.core.observability.audit import audit_event
from claritymed.web.jwt import JWT_ALGORITHM, create_token
from claritymed.web.middleware import (
    ANONYMOUS_USER_ID,
    COOKIE_ACCESS_TOKEN,
    HEADER_REQUEST_ID,
    WebContextMiddleware,
)


def _app_with_context_inspector() -> FastAPI:
    """Tiny app that echoes the current ContextVars back to the caller."""
    app = FastAPI()
    app.add_middleware(WebContextMiddleware, default_lang=_cfg.default_lang())

    @app.get("/whoami")
    async def whoami():
        return {"user_id": user_id_ctx.get(), "language": language_ctx.get()}

    @app.get("/emit-audit")
    async def emit_audit():
        # Proves ContextVars are populated even for anonymous traffic;
        # would raise MissingContextError otherwise.
        audit_event("request_start", payload={"entry": "test"})
        return {"ok": True}

    return app


@pytest.fixture
async def ctx_client(jwt_secret):  # noqa: ARG001
    app = _app_with_context_inspector()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


async def test_no_cookie_yields_anonymous_context(ctx_client):
    resp = await ctx_client.get("/whoami")
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == ANONYMOUS_USER_ID


async def test_valid_jwt_sets_sub_and_lang(ctx_client, jwt_secret):  # noqa: ARG001
    token = create_token("user-1", "zh")
    resp = await ctx_client.get("/whoami", cookies={COOKIE_ACCESS_TOKEN: token})
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == "user-1"
    assert body["language"] == "zh"


async def test_tampered_jwt_signature_silently_downgrades_to_anonymous(
    ctx_client, jwt_secret
):  # noqa: ARG001
    token = create_token("user-1", "en")
    bad = token[:-4] + "AAAA"
    resp = await ctx_client.get("/whoami", cookies={COOKIE_ACCESS_TOKEN: bad})
    assert resp.status_code == 200
    assert resp.json()["user_id"] == ANONYMOUS_USER_ID


async def test_expired_jwt_silently_downgrades(ctx_client, jwt_secret):
    expired = _jose_jwt.encode(
        {"sub": "user-1", "lang": "en", "iat": 0, "exp": 1},
        jwt_secret,
        algorithm=JWT_ALGORITHM,
    )
    resp = await ctx_client.get("/whoami", cookies={COOKIE_ACCESS_TOKEN: expired})
    assert resp.status_code == 200
    assert resp.json()["user_id"] == ANONYMOUS_USER_ID


async def test_tamper_lang_outside_allowlist_fails_loud(ctx_client, jwt_secret):
    # Valid HMAC + disallowed lang ⇒ Tamper ⇒ 401 + cookie cleared.
    bad = _jose_jwt.encode(
        {"sub": "user-1", "lang": "ja", "iat": 0, "exp": int(time.time()) + 60},
        jwt_secret,
        algorithm=JWT_ALGORITHM,
    )
    resp = await ctx_client.get("/whoami", cookies={COOKIE_ACCESS_TOKEN: bad})
    assert resp.status_code == 401
    # Cookie cleared via Set-Cookie with Max-Age=0 / expires past.
    set_cookies = resp.headers.get_list("set-cookie")
    assert any(
        COOKIE_ACCESS_TOKEN in c and ("Max-Age=0" in c or "expires=" in c.lower())
        for c in set_cookies
    ), set_cookies


async def test_request_id_header_set_on_response(ctx_client):
    resp = await ctx_client.get("/whoami")
    assert HEADER_REQUEST_ID in resp.headers
    rid = resp.headers[HEADER_REQUEST_ID]
    # 22-char request id (14-digit timestamp + 8-hex suffix).
    assert len(rid) == 22


async def test_audit_event_inside_handler_does_not_raise(ctx_client, caplog):
    # MissingContextError would surface as a 500 if ContextVars were unset.
    with caplog.at_level(logging.INFO):
        resp = await ctx_client.get("/emit-audit")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


async def test_tamper_emits_audit_event(ctx_client, jwt_secret, caplog):
    bad = _jose_jwt.encode(
        {"sub": "user-1", "lang": "ja", "iat": 0, "exp": int(time.time()) + 60},
        jwt_secret,
        algorithm=JWT_ALGORITHM,
    )
    with caplog.at_level(logging.INFO):
        await ctx_client.get("/whoami", cookies={COOKIE_ACCESS_TOKEN: bad})
    # Audit logger emits JSON lines containing the kind; grep them.
    assert any("web.jwt.tamper_suspected" in rec.message for rec in caplog.records), [
        rec.message for rec in caplog.records
    ]


async def test_invalid_jwt_emits_lifecycle_audit(ctx_client, jwt_secret, caplog):  # noqa: ARG001
    token = create_token("user-1", "en")
    bad = token[:-4] + "AAAA"
    with caplog.at_level(logging.INFO):
        await ctx_client.get("/whoami", cookies={COOKIE_ACCESS_TOKEN: bad})
    assert any("web.jwt.invalid" in rec.message for rec in caplog.records), [
        rec.message for rec in caplog.records
    ]


async def test_no_cookie_does_not_emit_jwt_audit(ctx_client, caplog):
    # Anonymous-by-design traffic should not pollute the audit log.
    with caplog.at_level(logging.INFO):
        await ctx_client.get("/whoami")
    assert not any(
        "web.jwt.invalid" in rec.message or "web.jwt.tamper_suspected" in rec.message
        for rec in caplog.records
    )
