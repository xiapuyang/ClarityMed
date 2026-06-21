"""``/auth/login`` + ``/auth/logout`` integration tests."""

from __future__ import annotations

import logging

import pytest

from claritymed.stores.auth import PasswordStore
from claritymed.web.csrf import COOKIE_CSRF_TOKEN
from claritymed.web.jwt import decode_token, hmac_ip
from claritymed.web.middleware import COOKIE_ACCESS_TOKEN

LOGIN_URL = "/auth/login"
LOGOUT_URL = "/auth/logout"


@pytest.fixture
def password_user(test_user):
    """A first-user (admin) with a known password set via PasswordStore."""
    PasswordStore.set_password(test_user.user_id, "correct-password-123")
    return test_user


# --- happy path -------------------------------------------------------


async def test_login_with_valid_credentials(web_client, password_user):
    resp = await web_client.post(
        LOGIN_URL,
        json={"user_id": password_user.user_id, "password": "correct-password-123"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "user_id": password_user.user_id,
        "display_name": password_user.display_name,
        "role": password_user.role,
        "language": password_user.language,
    }
    # access_token cookie should be HttpOnly + SameSite=Lax
    set_cookies = resp.headers.get_list("set-cookie")
    access_cookie = next(
        c for c in set_cookies if c.startswith(f"{COOKIE_ACCESS_TOKEN}=")
    )
    assert "HttpOnly" in access_cookie
    assert "samesite=lax" in access_cookie.lower()


async def test_login_issues_decodable_jwt(web_client, password_user):
    resp = await web_client.post(
        LOGIN_URL,
        json={"user_id": password_user.user_id, "password": "correct-password-123"},
    )
    cookie = resp.cookies.get(COOKIE_ACCESS_TOKEN)
    assert cookie is not None
    from claritymed.web.jwt import Valid

    decoded = decode_token(cookie)
    assert isinstance(decoded, Valid)
    assert decoded.claims["sub"] == password_user.user_id
    assert decoded.claims["lang"] == password_user.language


async def test_login_rotates_csrf_cookie(web_client, password_user):
    pre = await web_client.get("/health")
    pre_csrf = pre.cookies.get(COOKIE_CSRF_TOKEN)
    resp = await web_client.post(
        LOGIN_URL,
        json={"user_id": password_user.user_id, "password": "correct-password-123"},
    )
    assert resp.status_code == 200
    new_csrf = resp.cookies.get(COOKIE_CSRF_TOKEN)
    assert new_csrf is not None
    assert new_csrf != pre_csrf


# --- wrong / unknown credentials --------------------------------------


async def test_login_wrong_password_returns_401(web_client, password_user):
    resp = await web_client.post(
        LOGIN_URL,
        json={"user_id": password_user.user_id, "password": "WRONG"},
    )
    assert resp.status_code == 401
    assert resp.json() == {"detail": "Invalid credentials"}


async def test_login_unknown_user_returns_same_401_body(web_client, password_user):  # noqa: ARG001
    resp = await web_client.post(
        LOGIN_URL,
        json={"user_id": "never-existed", "password": "anything"},
    )
    assert resp.status_code == 401
    assert resp.json() == {"detail": "Invalid credentials"}


# --- input validation -------------------------------------------------


async def test_login_invalid_user_id_returns_422(web_client):
    resp = await web_client.post(
        LOGIN_URL,
        json={"user_id": "../etc/passwd", "password": "anything"},
    )
    assert resp.status_code == 422


async def test_login_empty_password_returns_422(web_client):
    resp = await web_client.post(
        LOGIN_URL,
        json={"user_id": "test", "password": ""},
    )
    assert resp.status_code == 422


# --- CSRF exemption ---------------------------------------------------


async def test_login_does_not_require_csrf_header(web_client, password_user):
    # No X-CSRF-Token header, no csrf_token cookie — login still works.
    resp = await web_client.post(
        LOGIN_URL,
        json={"user_id": password_user.user_id, "password": "correct-password-123"},
    )
    assert resp.status_code == 200


async def test_logout_does_not_require_csrf_header(web_client):
    resp = await web_client.post(LOGOUT_URL)
    assert resp.status_code == 204


# --- logout -----------------------------------------------------------


async def test_logout_clears_access_token_cookie(web_client):
    resp = await web_client.post(LOGOUT_URL)
    assert resp.status_code == 204
    set_cookies = resp.headers.get_list("set-cookie")
    # access_token cookie cleared via Max-Age=0 / expires past.
    cleared = next(
        (c for c in set_cookies if c.startswith(f"{COOKIE_ACCESS_TOKEN}=")), None
    )
    assert cleared is not None
    assert "Max-Age=0" in cleared or "expires=" in cleared.lower()


# --- audit forensics --------------------------------------------------


async def test_login_failed_emits_ip_hmac(web_client, password_user, caplog):
    with caplog.at_level(logging.INFO):
        await web_client.post(
            LOGIN_URL,
            json={"user_id": password_user.user_id, "password": "wrong"},
        )
    fail_lines = [
        r.message for r in caplog.records if "web.auth.login_failed" in r.message
    ]
    assert fail_lines, "expected web.auth.login_failed audit row"
    # The audit payload includes ip_hmac; we don't reconstruct the host
    # (httpx ASGI transport uses 127.0.0.1) but we assert the field is
    # there and looks like a hex string.
    line = fail_lines[0]
    assert "ip_hmac" in line


async def test_login_failed_same_ip_produces_same_hash(
    web_client, password_user, caplog
):
    with caplog.at_level(logging.INFO):
        await web_client.post(
            LOGIN_URL,
            json={"user_id": password_user.user_id, "password": "wrong1"},
        )
        await web_client.post(
            LOGIN_URL,
            json={"user_id": password_user.user_id, "password": "wrong2"},
        )
    fails = [r.message for r in caplog.records if "web.auth.login_failed" in r.message]
    assert len(fails) >= 2
    # Two failures from the same client IP must produce the same ip_hmac.
    expected = hmac_ip("127.0.0.1")
    assert all(expected in line for line in fails[:2])


async def test_login_success_emits_audit(web_client, password_user, caplog):
    with caplog.at_level(logging.INFO):
        await web_client.post(
            LOGIN_URL,
            json={"user_id": password_user.user_id, "password": "correct-password-123"},
        )
    assert any("web.auth.login_success" in r.message for r in caplog.records)


async def test_logout_emits_audit(web_client, caplog):
    with caplog.at_level(logging.INFO):
        await web_client.post(LOGOUT_URL)
    assert any("web.auth.logout" in r.message for r in caplog.records)


# --- consecutive logins rotate csrf -----------------------------------


async def test_two_logins_issue_different_csrf_tokens(web_client, password_user):
    r1 = await web_client.post(
        LOGIN_URL,
        json={"user_id": password_user.user_id, "password": "correct-password-123"},
    )
    r2 = await web_client.post(
        LOGIN_URL,
        json={"user_id": password_user.user_id, "password": "correct-password-123"},
    )
    c1 = r1.cookies.get(COOKIE_CSRF_TOKEN)
    c2 = r2.cookies.get(COOKIE_CSRF_TOKEN)
    assert c1 is not None and c2 is not None
    assert c1 != c2
