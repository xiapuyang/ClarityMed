"""Tests for ``/api/v1/admin/users``."""

from __future__ import annotations

import pytest

from claritymed.stores.account import AccountStore
from claritymed.stores.auth import PasswordStore


@pytest.mark.asyncio
async def test_list_users_requires_admin(web_client, non_admin_cookies):
    response = await web_client.get("/api/v1/admin/users", cookies=non_admin_cookies)
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_list_users_returns_all(web_client, admin_cookies, non_admin_user):
    response = await web_client.get("/api/v1/admin/users", cookies=admin_cookies)
    assert response.status_code == 200
    body = response.json()
    ids = {u["user_id"] for u in body["items"]}
    assert "test" in ids
    assert "test-user" in ids


@pytest.mark.asyncio
async def test_get_user_404_on_missing(web_client, admin_cookies):
    response = await web_client.get(
        "/api/v1/admin/users/no-such-user", cookies=admin_cookies
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_patch_role_user_to_admin(web_client, admin_cookies, non_admin_user):
    response = await web_client.patch(
        "/api/v1/admin/users/test-user",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        json={"role": "admin"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["role"] == "admin"
    # On-disk verify.
    reloaded = AccountStore("test-user").load()
    assert reloaded.role == "admin"


@pytest.mark.asyncio
async def test_patch_self_demotion_blocked(web_client, admin_cookies):
    response = await web_client.patch(
        "/api/v1/admin/users/test",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        json={"role": "user"},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "demote" in detail.lower()


@pytest.mark.asyncio
async def test_patch_unknown_provider_400(web_client, admin_cookies, non_admin_user):
    response = await web_client.patch(
        "/api/v1/admin/users/test-user",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        json={"provider_id": "not-a-provider"},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_patch_invalid_role_returns_422(
    web_client, admin_cookies, non_admin_user
):
    response = await web_client.patch(
        "/api/v1/admin/users/test-user",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        json={"role": "superuser"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_reset_password_writes_hash(web_client, admin_cookies, non_admin_user):
    PasswordStore.set_password("test-user", "initial-password")
    assert PasswordStore.verify_password("test-user", "initial-password")
    response = await web_client.post(
        "/api/v1/admin/users/test-user/reset-password",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        json={"new_password": "next-password"},
    )
    assert response.status_code == 204
    assert not PasswordStore.verify_password("test-user", "initial-password")
    assert PasswordStore.verify_password("test-user", "next-password")


@pytest.mark.asyncio
async def test_get_user_returns_existing(web_client, admin_cookies, non_admin_user):
    """GET /admin/users/{uid} for an existing user returns their summary."""
    response = await web_client.get(
        "/api/v1/admin/users/test-user", cookies=admin_cookies
    )
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "test-user"
    assert body["role"] == "user"


@pytest.mark.asyncio
async def test_patch_user_404_on_missing(web_client, admin_cookies):
    """PATCH /admin/users/{uid} for a non-existent user → 404."""
    response = await web_client.patch(
        "/api/v1/admin/users/no-such-user",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        json={"language": "zh"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_patch_user_no_fields_returns_account_unchanged(
    web_client, admin_cookies, non_admin_user
):
    """PATCH with all-None fields returns the account without mutation."""
    response = await web_client.patch(
        "/api/v1/admin/users/test-user",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        json={},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "test-user"


@pytest.mark.asyncio
async def test_reset_password_404_on_missing(web_client, admin_cookies):
    """POST /admin/users/{uid}/reset-password for a non-existent user → 404."""
    response = await web_client.post(
        "/api/v1/admin/users/no-such-user/reset-password",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        json={"new_password": "newpass1234"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_list_users_skips_user_without_account_file(
    web_client, admin_cookies, monkeypatch
):
    """list_users skips a uid that list_user_ids returns but has no settings.yaml."""
    from claritymed.web.routers.admin import users as users_router

    monkeypatch.setattr(
        users_router, "list_user_ids", lambda: ["test", "ghost-no-acct"]
    )
    response = await web_client.get("/api/v1/admin/users", cookies=admin_cookies)
    assert response.status_code == 200
    body = response.json()
    ids = {u["user_id"] for u in body["items"]}
    assert "ghost-no-acct" not in ids
    assert "test" in ids
