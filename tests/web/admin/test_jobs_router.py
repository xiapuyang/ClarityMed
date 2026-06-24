"""Tests for the ``/api/v1/admin/jobs`` router."""

from __future__ import annotations

import asyncio

import pytest

from claritymed.web.admin.jobs import JobRegistry


def _slow_runner(delay: float = 1.0):
    async def runner(_spec, _registry):
        await asyncio.sleep(delay)

    return runner


@pytest.mark.asyncio
async def test_list_jobs_requires_admin(web_client, non_admin_cookies):
    response = await web_client.get("/api/v1/admin/jobs", cookies=non_admin_cookies)
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_list_jobs_empty(web_client, admin_cookies):
    response = await web_client.get("/api/v1/admin/jobs", cookies=admin_cookies)
    assert response.status_code == 200
    body = response.json()
    assert body["items"] == []


@pytest.mark.asyncio
async def test_list_jobs_returns_submitted(web_app, web_client, admin_cookies):
    registry: JobRegistry = web_app.state.jobs
    registry.register_runner("rag_ingest", _slow_runner(2.0))
    spec = await registry.submit("rag_ingest", {"file": "x.md"})

    response = await web_client.get("/api/v1/admin/jobs", cookies=admin_cookies)
    body = response.json()
    assert body["total"] >= 1
    ids = [j["id"] for j in body["items"]]
    assert spec.id in ids

    # Cleanup so the test doesn't leave a pending task.
    await registry.cancel(spec.id)


@pytest.mark.asyncio
async def test_get_job_404(web_client, admin_cookies):
    response = await web_client.get(
        "/api/v1/admin/jobs/does-not-exist", cookies=admin_cookies
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_cancel_job_via_delete(web_app, web_client, admin_cookies):
    registry: JobRegistry = web_app.state.jobs
    registry.register_runner("rag_ingest", _slow_runner(2.0))
    spec = await registry.submit("rag_ingest", {})

    response = await web_client.delete(
        f"/api/v1/admin/jobs/{spec.id}",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
    )
    assert response.status_code == 200
    # Either cancelled now or in flight to cancelled; the runner is
    # cooperative so wait briefly.
    for _ in range(30):
        await asyncio.sleep(0.02)
        spec_now = registry.get(spec.id)
        if spec_now.state == "cancelled":
            break
    assert registry.get(spec.id).state == "cancelled"
