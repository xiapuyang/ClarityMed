"""Tests for ``/api/v1/admin/benchmark``."""

from __future__ import annotations

import json

import pytest

from claritymed import config as _cfg


@pytest.fixture
def tmp_evals_dir(tmp_path, monkeypatch):
    shared = tmp_path / "benchmark_shared"
    shared.mkdir(parents=True, exist_ok=True)
    evals = shared / "evals"
    evals.mkdir(parents=True, exist_ok=True)
    # Two runs: one complete, one missing manifest.
    complete = evals / "run-001"
    complete.mkdir()
    (complete / "result.json").write_text(
        json.dumps({"accuracy": 0.71}), encoding="utf-8"
    )
    incomplete = evals / "run-002"
    incomplete.mkdir()
    monkeypatch.setattr(_cfg, "SHARED_DIR", shared)
    yield evals


@pytest.mark.asyncio
async def test_runs_requires_admin(web_client, non_admin_cookies):
    response = await web_client.get(
        "/api/v1/admin/benchmark/runs", cookies=non_admin_cookies
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_list_runs_empty(web_client, admin_cookies, tmp_path, monkeypatch):
    monkeypatch.setattr(_cfg, "SHARED_DIR", tmp_path)
    response = await web_client.get(
        "/api/v1/admin/benchmark/runs", cookies=admin_cookies
    )
    body = response.json()
    assert body["items"] == []


@pytest.mark.asyncio
async def test_list_runs_with_complete_and_incomplete(
    web_client, admin_cookies, tmp_evals_dir
):
    response = await web_client.get(
        "/api/v1/admin/benchmark/runs", cookies=admin_cookies
    )
    body = response.json()
    ids = {r["run_id"]: r for r in body["items"]}
    assert ids["run-001"]["complete"] is True
    assert ids["run-002"]["complete"] is False


@pytest.mark.asyncio
async def test_get_run_404(web_client, admin_cookies, tmp_evals_dir):
    response = await web_client.get(
        "/api/v1/admin/benchmark/runs/not-here", cookies=admin_cookies
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_run_returns_manifest(web_client, admin_cookies, tmp_evals_dir):
    response = await web_client.get(
        "/api/v1/admin/benchmark/runs/run-001", cookies=admin_cookies
    )
    body = response.json()
    assert body["manifest"]["accuracy"] == 0.71
    assert body["complete"] is True


@pytest.mark.asyncio
async def test_trigger_run_creates_job(web_client, admin_cookies, tmp_evals_dir):
    response = await web_client.post(
        "/api/v1/admin/benchmark/runs",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        json={"dataset": "medqa", "provider_id": "openai-gpt-4o"},
    )
    assert response.status_code == 200
    spec = response.json()
    assert spec["kind"] == "benchmark_run"
