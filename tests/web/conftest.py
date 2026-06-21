"""Web test fixtures.

The root ``tests/conftest.py`` already re-points ``CLARITYMED_HOME`` per
test (so every test gets its own data tree). This file adds web-specific
fixtures:

* ``jwt_secret`` — a deterministic 64-hex secret installed via env so
  ``validate_secret_or_raise`` passes and ``create_token`` /
  ``decode_token`` share the same key.
* ``web_app`` — a freshly-built FastAPI app with lifespan exercised.
* ``web_client`` — an ``httpx.AsyncClient`` bound to the app via
  ``ASGITransport``. Use ``follow_redirects=False`` so test assertions
  on Set-Cookie / status code see the raw response.

Both fixtures are function-scoped because the lifespan is light (no
provider building yet) and per-test isolation matters more than speed.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from claritymed.stores.account import init_user
from claritymed.web.app import create_app
from claritymed.web.jwt import ENV_JWT_SECRET, create_token

# Use the CLAUDE.md-mandated test user_id rather than alice/bob/carol.
TEST_USER_ID = "test"

# Deterministic 64-hex secret. Anything that isn't a known-bad literal
# satisfies ``validate_secret_or_raise``.
TEST_JWT_SECRET = "a" * 64


@pytest.fixture
def jwt_secret(monkeypatch: pytest.MonkeyPatch) -> str:
    """Install a stable JWT secret for the test."""
    monkeypatch.setenv(ENV_JWT_SECRET, TEST_JWT_SECRET)
    return TEST_JWT_SECRET


@pytest.fixture
def dev_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force production-equivalent dev=0 unless a test opts in."""
    monkeypatch.delenv("CLARITYMED_DEV", raising=False)


@pytest.fixture
async def web_app(jwt_secret: str, dev_mode):  # noqa: ARG001 — fixture wiring
    """A FastAPI app with the lifespan exercised."""
    app = create_app()
    async with app.router.lifespan_context(app):
        yield app


@pytest.fixture
async def web_client(web_app):
    """``httpx.AsyncClient`` bound to the web app via ASGITransport."""
    transport = ASGITransport(app=web_app)
    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
        follow_redirects=False,
    ) as client:
        yield client


@pytest.fixture
def test_user():
    """First user → auto-promoted to admin by ``init_user``."""
    return init_user(TEST_USER_ID, display_name="Test")


@pytest.fixture
def non_admin_user(test_user):  # noqa: ARG001 — depends on test_user existing
    """Second user → default ``user`` role. ``test_user`` precedes for ordering."""
    return init_user("test-user", display_name="Non-Admin")


def make_auth_cookies(user_id: str, lang: str = "en") -> dict[str, str]:
    """Build the cookie pair an authenticated browser request would send."""
    token = create_token(user_id, lang)
    return {
        "access_token": token,
        "csrf_token": "csrf-test-token",
    }


def auth_headers(method: str = "POST") -> dict[str, str]:
    """Build the headers for an authenticated mutation (matches csrf cookie)."""
    if method.upper() in {"GET", "HEAD", "OPTIONS"}:
        return {}
    return {"X-CSRF-Token": "csrf-test-token"}
