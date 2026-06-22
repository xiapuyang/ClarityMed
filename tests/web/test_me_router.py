"""``/api/v1/me`` GET/PATCH integration tests."""

from __future__ import annotations

import logging

import pytest

from claritymed.stores.account import AccountStore
from claritymed.web.csrf import HEADER_CSRF_TOKEN
from claritymed.web.jwt import Valid, create_token, decode_token
from claritymed.web.middleware import COOKIE_ACCESS_TOKEN

ME_URL = "/api/v1/me"


@pytest.fixture
def auth_cookies(test_user):
    """Cookies for a logged-in admin (first user → admin)."""
    token = create_token(test_user.user_id, test_user.language)
    return {COOKIE_ACCESS_TOKEN: token, "csrf_token": "csrf-test"}


def _csrf_headers() -> dict[str, str]:
    return {HEADER_CSRF_TOKEN: "csrf-test"}


# --- GET --------------------------------------------------------------


async def test_get_me_returns_account_shape(web_client, test_user, auth_cookies):
    resp = await web_client.get(ME_URL, cookies=auth_cookies)
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "user_id": test_user.user_id,
        "display_name": test_user.display_name,
        "role": test_user.role,
        "language": test_user.language,
        "provider_id": test_user.provider_id,
    }
    # Belt-and-suspenders: never leak the password hash through this
    # endpoint even if a future schema change adds it.
    assert "password_hash" not in body


async def test_get_me_unauthenticated_returns_401(web_client):
    resp = await web_client.get(ME_URL)
    assert resp.status_code == 401


# --- PATCH display_name -----------------------------------------------


async def test_patch_display_name_persists(web_client, test_user, auth_cookies):
    resp = await web_client.patch(
        ME_URL,
        json={"display_name": "New Name"},
        cookies=auth_cookies,
        headers=_csrf_headers(),
    )
    assert resp.status_code == 200
    assert resp.json()["display_name"] == "New Name"
    # Verify it actually persisted to disk (not just an optimistic
    # in-memory mutation).
    reread = AccountStore(test_user.user_id).load()
    assert reread.display_name == "New Name"


# --- PATCH language ---------------------------------------------------


async def test_patch_language_reissues_cookie(web_client, test_user, auth_cookies):
    assert test_user.language == "en"
    resp = await web_client.patch(
        ME_URL,
        json={"language": "zh"},
        cookies=auth_cookies,
        headers=_csrf_headers(),
    )
    assert resp.status_code == 200
    assert resp.json()["language"] == "zh"

    # Cookie reissued with new lang claim.
    new_cookie = resp.cookies.get(COOKIE_ACCESS_TOKEN)
    assert new_cookie is not None
    decoded = decode_token(new_cookie)
    assert isinstance(decoded, Valid)
    assert decoded.claims["lang"] == "zh"


async def test_patch_language_does_not_reissue_when_unchanged(
    web_client, test_user, auth_cookies
):
    resp = await web_client.patch(
        ME_URL,
        json={"display_name": "Just A Name Change"},
        cookies=auth_cookies,
        headers=_csrf_headers(),
    )
    assert resp.status_code == 200
    # No access_token Set-Cookie when language didn't change.
    set_cookies = resp.headers.get_list("set-cookie")
    assert not any(c.startswith(f"{COOKIE_ACCESS_TOKEN}=") for c in set_cookies)


async def test_patch_language_emits_audit(web_client, test_user, auth_cookies, caplog):
    with caplog.at_level(logging.INFO):
        await web_client.patch(
            ME_URL,
            json={"language": "zh"},
            cookies=auth_cookies,
            headers=_csrf_headers(),
        )
    msgs = [r.message for r in caplog.records if "web.me.language_changed" in r.message]
    assert msgs
    # Payload carries from/to.
    assert '"from":"en"' in msgs[0]
    assert '"to":"zh"' in msgs[0]


# --- input validation -------------------------------------------------


async def test_patch_invalid_language_returns_422(web_client, test_user, auth_cookies):
    resp = await web_client.patch(
        ME_URL,
        json={"language": "ja"},
        cookies=auth_cookies,
        headers=_csrf_headers(),
    )
    assert resp.status_code == 422


async def test_patch_extra_field_returns_422(web_client, test_user, auth_cookies):
    # Trying to promote self to admin must be rejected at the schema layer.
    resp = await web_client.patch(
        ME_URL,
        json={"role": "admin"},
        cookies=auth_cookies,
        headers=_csrf_headers(),
    )
    assert resp.status_code == 422


async def test_patch_unknown_provider_id_returns_422(
    web_client, test_user, auth_cookies
):
    # Unknown provider ids are rejected at the router, BEFORE the change
    # lands on disk — otherwise a typo would survive in settings.yaml and
    # blow up at the next stream call with a confusing UnknownProviderError.
    resp = await web_client.patch(
        ME_URL,
        json={"provider_id": "this-provider-does-not-exist"},
        cookies=auth_cookies,
        headers=_csrf_headers(),
    )
    assert resp.status_code == 422


async def test_patch_known_provider_id_persists(web_client, test_user, auth_cookies):
    # ``omlx`` is the catalog default and is always present in models.yaml.
    resp = await web_client.patch(
        ME_URL,
        json={"provider_id": "omlx"},
        cookies=auth_cookies,
        headers=_csrf_headers(),
    )
    assert resp.status_code == 200
    assert resp.json()["provider_id"] == "omlx"
    reread = AccountStore(test_user.user_id).load()
    assert reread.provider_id == "omlx"


# --- CSRF -------------------------------------------------------------


async def test_patch_without_csrf_header_returns_403(web_client, auth_cookies):
    resp = await web_client.patch(
        ME_URL,
        json={"display_name": "x"},
        cookies=auth_cookies,
    )
    assert resp.status_code == 403


# --- empty body -------------------------------------------------------


async def test_patch_empty_body_is_no_op(web_client, test_user, auth_cookies):
    resp = await web_client.patch(
        ME_URL,
        json={},
        cookies=auth_cookies,
        headers=_csrf_headers(),
    )
    assert resp.status_code == 200
    assert resp.json()["display_name"] == test_user.display_name
    assert resp.json()["language"] == test_user.language
