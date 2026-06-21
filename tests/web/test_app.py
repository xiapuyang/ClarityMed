"""App-level tests: /health, OpenAPI gating, startup gates, CORS."""

from __future__ import annotations

import logging

import pytest
from httpx import ASGITransport, AsyncClient

from claritymed.web.app import create_app
from claritymed.web.jwt import ENV_JWT_SECRET, create_token
from claritymed.web.middleware import COOKIE_ACCESS_TOKEN


# --- /health ------------------------------------------------------------


async def test_health_endpoint(web_client):
    resp = await web_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_health_sets_csrf_cookie(web_client):
    resp = await web_client.get("/health")
    set_cookies = resp.headers.get_list("set-cookie")
    assert any("csrf_token=" in c for c in set_cookies), set_cookies


# --- JWT secret startup gate -------------------------------------------


async def test_lifespan_raises_when_jwt_secret_unset(monkeypatch):
    monkeypatch.delenv(ENV_JWT_SECRET, raising=False)
    monkeypatch.delenv("CLARITYMED_DEV", raising=False)
    app = create_app()
    with pytest.raises(RuntimeError, match=ENV_JWT_SECRET):
        async with app.router.lifespan_context(app):
            pass  # pragma: no cover


@pytest.mark.parametrize("bad", ["", "changeme", "secret"])
async def test_lifespan_raises_on_known_bad_secret(monkeypatch, bad: str):
    monkeypatch.setenv(ENV_JWT_SECRET, bad)
    monkeypatch.delenv("CLARITYMED_DEV", raising=False)
    app = create_app()
    with pytest.raises(RuntimeError):
        async with app.router.lifespan_context(app):
            pass  # pragma: no cover


# --- CORS configuration ------------------------------------------------


async def test_cors_wildcard_rejected_at_startup(monkeypatch, jwt_secret):  # noqa: ARG001
    monkeypatch.setenv("CLARITYMED_CORS_ORIGINS", "*")
    with pytest.raises(RuntimeError, match="must not include"):
        create_app()


async def test_cors_dev_adds_localhost_frontend(monkeypatch, jwt_secret):  # noqa: ARG001
    monkeypatch.setenv("CLARITYMED_DEV", "1")
    monkeypatch.delenv("CLARITYMED_CORS_ORIGINS", raising=False)
    app = create_app()
    transport = ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=transport, base_url="http://testserver") as c:
            # OPTIONS preflight from the dev frontend should be allowed.
            resp = await c.options(
                "/health",
                headers={
                    "Origin": "http://localhost:5173",
                    "Access-Control-Request-Method": "GET",
                },
            )
    assert resp.status_code in (200, 204)
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"


# --- OpenAPI gating ----------------------------------------------------


async def test_openapi_in_production_unauthenticated_returns_404(web_client):
    resp = await web_client.get("/openapi.json")
    assert resp.status_code == 404


async def test_openapi_in_production_emits_audit_on_block(web_client, caplog):
    with caplog.at_level(logging.INFO):
        await web_client.get("/openapi.json")
    assert any("web.openapi.access_blocked" in rec.message for rec in caplog.records)


async def test_openapi_admin_can_access_in_production(web_client, test_user):
    # test_user is the first user → auto-promoted to admin.
    assert test_user.role == "admin"
    token = create_token(test_user.user_id, test_user.language)
    resp = await web_client.get(
        "/openapi.json",
        cookies={COOKIE_ACCESS_TOKEN: token},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["info"]["title"] == "ClarityMed Web"


async def test_openapi_non_admin_returns_404_in_production(
    web_client, test_user, non_admin_user
):
    assert non_admin_user.role == "user"
    token = create_token(non_admin_user.user_id, non_admin_user.language)
    resp = await web_client.get(
        "/openapi.json",
        cookies={COOKIE_ACCESS_TOKEN: token},
    )
    assert resp.status_code == 404


async def test_openapi_dev_mode_opens_unauthenticated(monkeypatch, jwt_secret):  # noqa: ARG001
    monkeypatch.setenv("CLARITYMED_DEV", "1")
    monkeypatch.delenv("CLARITYMED_CORS_ORIGINS", raising=False)
    app = create_app()
    transport = ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=transport, base_url="http://testserver") as c:
            resp = await c.get("/openapi.json")
    assert resp.status_code == 200


async def test_docs_route_admin_gated_in_production(web_client, test_user):
    # Admin path → 200 + HTML.
    token = create_token(test_user.user_id, test_user.language)
    resp = await web_client.get("/docs", cookies={COOKIE_ACCESS_TOKEN: token})
    assert resp.status_code == 200
    assert "swagger" in resp.text.lower()

    # Anonymous → 404 (don't reveal existence).
    anon = await web_client.get("/docs")
    assert anon.status_code == 404


# --- Exception handling -----------------------------------------------


async def test_unknown_provider_error_maps_to_500(web_client):
    # Synthetic: register a route that raises UnknownProviderError; the
    # global handler should map it to 500 with a non-PHI body.
    app = web_client._transport.app  # type: ignore[attr-defined]

    from claritymed.errors import UnknownProviderError

    @app.get("/__test_unknown_provider")
    async def _raise():
        raise UnknownProviderError("test")

    resp = await web_client.get("/__test_unknown_provider")
    assert resp.status_code == 500
    assert resp.json() == {"detail": "Configuration error"}
