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
async def test_runs_accessible_to_any_user(web_client, non_admin_cookies):
    response = await web_client.get(
        "/api/v1/admin/benchmark/runs", cookies=non_admin_cookies
    )
    assert response.status_code == 200


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


@pytest.mark.asyncio
async def test_list_runs_skips_non_directory_entries(
    web_client, admin_cookies, tmp_evals_dir
):
    """A plain file in the evals root is skipped; only dirs appear in the list."""
    (tmp_evals_dir / "stray-file.txt").write_text("noise", encoding="utf-8")
    response = await web_client.get(
        "/api/v1/admin/benchmark/runs", cookies=admin_cookies
    )
    body = response.json()
    run_ids = {r["run_id"] for r in body["items"]}
    assert "stray-file.txt" not in run_ids


@pytest.mark.asyncio
async def test_read_manifest_unreadable_returns_incomplete(
    web_client, admin_cookies, tmp_evals_dir
):
    """A manifest file that contains invalid JSON is treated as missing."""
    (tmp_evals_dir / "run-bad").mkdir()
    (tmp_evals_dir / "run-bad" / "result.json").write_text(
        "{not-json!", encoding="utf-8"
    )
    response = await web_client.get(
        "/api/v1/admin/benchmark/runs/run-bad", cookies=admin_cookies
    )
    body = response.json()
    assert body["complete"] is False
    assert body["manifest"] is None


@pytest.mark.asyncio
async def test_get_run_dot_prefix_returns_400(web_client, admin_cookies, tmp_evals_dir):
    """run_id with a leading dot is rejected with 400."""
    response = await web_client.get(
        "/api/v1/admin/benchmark/runs/.hidden-run",
        cookies=admin_cookies,
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_trigger_run_no_runner_registered_returns_503(
    web_app, web_client, admin_cookies
):
    """Submitting a benchmark when the runner isn't wired → 503."""
    web_app.state.jobs._runners.pop("benchmark_run", None)
    try:
        response = await web_client.post(
            "/api/v1/admin/benchmark/runs",
            cookies=admin_cookies,
            headers={"X-CSRF-Token": "csrf-test-token"},
            json={"dataset": "medqa"},
        )
        assert response.status_code == 503
    finally:
        from claritymed.web.admin.job_runners import benchmark_run

        web_app.state.jobs.register_runner("benchmark_run", benchmark_run.run)


# --- benchmark_run runner unit tests -------------------------------------------


def test_build_args_with_all_params():
    """_build_args includes sample_size and extra_args when provided."""
    from claritymed.web.admin.job_runners.benchmark_run import _build_args

    args = _build_args(
        {
            "dataset": "medqa",
            "provider_id": "openai-gpt-4o",
            "sample_size": 50,
            "extra_args": ["--seed", "42"],
        }
    )
    assert "--dataset" in args
    assert "medqa" in args
    assert "--provider" in args
    assert "openai-gpt-4o" in args
    assert "--sample-size" in args
    assert "50" in args
    assert "--seed" in args
    assert "42" in args


def test_build_args_minimal():
    """_build_args with no optional params returns empty list."""
    from claritymed.web.admin.job_runners.benchmark_run import _build_args

    args = _build_args({})
    assert args == []


@pytest.mark.asyncio
async def test_stream_stdout_processes_lines_and_stops_on_empty():
    """_stream_stdout reads lines until empty chunk; appends non-empty ones."""
    from unittest.mock import AsyncMock, MagicMock

    from claritymed.web.admin.job_runners.benchmark_run import _stream_stdout

    proc = MagicMock()
    proc.stdout = MagicMock()
    proc.stdout.readline = AsyncMock(side_effect=[b"line one\n", b"line two\n", b""])
    spec = MagicMock()
    spec.id = "bench-job-001"
    registry = MagicMock()

    await _stream_stdout(proc, spec, registry)

    assert registry.append_stdout.call_count == 2
    registry.append_stdout.assert_any_call("bench-job-001", "line one")
    registry.append_stdout.assert_any_call("bench-job-001", "line two")
