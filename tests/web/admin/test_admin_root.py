"""Smoke tests for the parent admin router gating."""

from __future__ import annotations

import pytest

ADMIN_HEALTH = "/api/v1/admin/_health"


@pytest.mark.asyncio
async def test_admin_health_requires_auth(web_client):
    """No cookies → 401."""
    response = await web_client.get(ADMIN_HEALTH)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_admin_health_accessible_to_any_user(web_client, non_admin_cookies):
    """role=user → 200 (admin routes open to all authenticated users)."""
    response = await web_client.get(ADMIN_HEALTH, cookies=non_admin_cookies)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_admin_health_ok_for_admin(web_client, admin_cookies):
    """role=admin → 200 with health payload."""
    response = await web_client.get(ADMIN_HEALTH, cookies=admin_cookies)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
