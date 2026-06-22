"""``GET /api/v1/providers`` integration tests."""

from __future__ import annotations

import pytest

from claritymed.web.jwt import create_token
from claritymed.web.middleware import COOKIE_ACCESS_TOKEN

PROVIDERS_URL = "/api/v1/providers"


@pytest.fixture
def auth_cookies(test_user):
    token = create_token(test_user.user_id, test_user.language)
    return {COOKIE_ACCESS_TOKEN: token, "csrf_token": "csrf-test"}


async def test_get_providers_returns_catalog(web_client, test_user, auth_cookies):
    resp = await web_client.get(PROVIDERS_URL, cookies=auth_cookies)
    assert resp.status_code == 200
    body = resp.json()
    assert "providers" in body
    assert "default_provider_id" in body
    assert "current_provider_id" in body
    # At least one local provider is shipped in the default catalog (omlx
    # and ollama) so the list is never empty in practice.
    assert len(body["providers"]) > 0
    ids = {p["id"] for p in body["providers"]}
    # ``omlx`` is the catalog default and must appear.
    assert "omlx" in ids


async def test_get_providers_entry_shape(web_client, test_user, auth_cookies):
    resp = await web_client.get(PROVIDERS_URL, cookies=auth_cookies)
    entry = next(p for p in resp.json()["providers"] if p["id"] == "omlx")
    assert entry["kind"] == "local"
    assert entry["model"]  # non-empty
    assert "family" in entry
    assert "display_name" in entry
    assert "context_window" in entry
    assert "available" in entry


async def test_get_providers_current_falls_back_to_default(
    web_client, test_user, auth_cookies
):
    # New user has no provider_id set → current == default.
    assert test_user.provider_id is None
    resp = await web_client.get(PROVIDERS_URL, cookies=auth_cookies)
    body = resp.json()
    assert body["current_provider_id"] == body["default_provider_id"]


async def test_get_providers_unauthenticated_returns_401(web_client):
    resp = await web_client.get(PROVIDERS_URL)
    assert resp.status_code == 401
