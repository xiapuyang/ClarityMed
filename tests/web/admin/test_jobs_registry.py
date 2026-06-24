"""Tests for ``claritymed.web.admin.jobs.JobRegistry``.

The registry is exercised in isolation (no FastAPI app needed) — runners
are tiny coroutines that flip a flag or sleep + raise so the state
machine paths can be covered cheaply.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from claritymed.stores.paths import job_path
from claritymed.web.admin.jobs import (
    CONCURRENCY_CAP,
    JobRegistry,
    JobSpec,
    TERMINAL_STATES,
)


def _dummy_runner(flag: dict[str, bool]):
    async def runner(_spec, _registry):
        flag["ran"] = True

    return runner


def _slow_runner(delay: float = 0.05):
    async def runner(_spec, _registry):
        await asyncio.sleep(delay)

    return runner


def _failing_runner():
    async def runner(_spec, _registry):
        raise RuntimeError("boom")

    return runner


async def test_submit_to_done_happy_path(tmp_jobs_dir):
    """A submitted job transitions queued → running → done."""
    flag = {"ran": False}
    registry = JobRegistry()
    registry.register_runner("rag_ingest", _dummy_runner(flag))
    spec = await registry.submit("rag_ingest", {"x": 1})
    for _ in range(50):
        await asyncio.sleep(0.01)
        if registry.get(spec.id).state == "done":
            break
    assert flag["ran"]
    assert registry.get(spec.id).state == "done"
    assert registry.get(spec.id).finished_at is not None


async def test_concurrency_cap_queues_overflow(tmp_jobs_dir):
    """A third submit while two slow runners are active stays queued."""
    registry = JobRegistry()
    registry.register_runner("rag_ingest", _slow_runner(0.1))
    s1 = await registry.submit("rag_ingest", {})
    s2 = await registry.submit("rag_ingest", {})
    s3 = await registry.submit("rag_ingest", {})
    # Let the dispatcher run.
    await asyncio.sleep(0.01)
    states = [registry.get(j.id).state for j in (s1, s2, s3)]
    assert states.count("running") == CONCURRENCY_CAP
    assert states.count("queued") == 1
    # Drain.
    for _ in range(100):
        await asyncio.sleep(0.02)
        if all(registry.get(j.id).state == "done" for j in (s1, s2, s3)):
            break
    assert all(registry.get(j.id).state == "done" for j in (s1, s2, s3))


async def test_cancel_running_marks_cancelled(tmp_jobs_dir):
    registry = JobRegistry()
    registry.register_runner("rag_ingest", _slow_runner(1.0))
    spec = await registry.submit("rag_ingest", {})
    # Wait for the runner to actually start.
    for _ in range(50):
        await asyncio.sleep(0.01)
        if registry.get(spec.id).state == "running":
            break
    await registry.cancel(spec.id)
    for _ in range(50):
        await asyncio.sleep(0.01)
        if registry.get(spec.id).state in TERMINAL_STATES:
            break
    assert registry.get(spec.id).state == "cancelled"


async def test_cancel_queued_short_circuits(tmp_jobs_dir):
    registry = JobRegistry()
    registry.register_runner("rag_ingest", _slow_runner(1.0))
    s1 = await registry.submit("rag_ingest", {})
    s2 = await registry.submit("rag_ingest", {})
    s3 = await registry.submit("rag_ingest", {})
    await asyncio.sleep(0.01)
    assert registry.get(s3.id).state == "queued"
    await registry.cancel(s3.id)
    assert registry.get(s3.id).state == "cancelled"
    # Drain the first two.
    await registry.cancel(s1.id)
    await registry.cancel(s2.id)


async def test_failing_runner_records_error(tmp_jobs_dir):
    registry = JobRegistry()
    registry.register_runner("rag_ingest", _failing_runner())
    spec = await registry.submit("rag_ingest", {})
    for _ in range(50):
        await asyncio.sleep(0.01)
        if registry.get(spec.id).state == "failed":
            break
    final = registry.get(spec.id)
    assert final.state == "failed"
    assert "boom" in (final.error or "")


async def test_recover_on_startup_marks_crashed(tmp_jobs_dir):
    """A pre-existing ``running`` spec on disk becomes ``crashed``."""
    spec = JobSpec(
        id="cafebabe-1234-5678-90ab-cdef00000000",
        kind="rag_ingest",
        state="running",
        params={"x": 1},
        started_at=time.time() - 5,
    )
    job_path(spec.id).parent.mkdir(parents=True, exist_ok=True)
    job_path(spec.id).write_text(json.dumps(spec.to_json()), encoding="utf-8")
    registry = JobRegistry()
    registry.recover_on_startup()
    assert registry.get(spec.id).state == "crashed"
    assert registry.get(spec.id).error is not None


async def test_recover_discards_queued(tmp_jobs_dir):
    spec = JobSpec(
        id="deadbeef-1234-5678-90ab-cdef00000000",
        kind="rag_ingest",
        state="queued",
        params={},
    )
    job_path(spec.id).parent.mkdir(parents=True, exist_ok=True)
    job_path(spec.id).write_text(json.dumps(spec.to_json()), encoding="utf-8")
    registry = JobRegistry()
    registry.recover_on_startup()
    assert registry.get(spec.id) is None
    assert not job_path(spec.id).exists()


async def test_submit_without_runner_raises(tmp_jobs_dir):
    registry = JobRegistry()
    with pytest.raises(ValueError, match="no runner"):
        await registry.submit("rag_ingest", {})


def test_cleanup_keeps_recent_window(tmp_jobs_dir):
    registry = JobRegistry()
    # Seed 150 finished specs all inside the 30-day window.
    now = time.time()
    for i in range(150):
        spec = JobSpec(
            id=f"job-{i:03d}",
            kind="rag_ingest",
            state="done",
            params={},
            finished_at=now - i,
            created_at=now - i,
        )
        registry._jobs[spec.id] = spec
        registry._persist(spec)
    registry._cleanup()
    # All inside window — keep everything.
    assert len(registry.list()) == 150


def test_cleanup_drops_old_finished(tmp_jobs_dir):
    registry = JobRegistry()
    now = time.time()
    # 120 finished specs, all >30 days old. Keep 100, drop 20.
    for i in range(120):
        spec = JobSpec(
            id=f"old-{i:03d}",
            kind="rag_ingest",
            state="done",
            params={},
            finished_at=now - (60 * 24 * 60 * 60) - i,
            created_at=now - (60 * 24 * 60 * 60) - i,
        )
        registry._jobs[spec.id] = spec
        registry._persist(spec)
    registry._cleanup()
    assert len(registry.list()) == 100


@pytest.fixture
def tmp_jobs_dir(tmp_path, monkeypatch):
    """Redirect DATA_DIR so jobs land under tmp."""
    home = tmp_path / "home"
    home.mkdir()
    data = home / "data"
    data.mkdir()
    monkeypatch.setenv("CLARITYMED_HOME", str(home))
    monkeypatch.setenv("CLARITYMED_DATA_DIR", str(data))
    import importlib

    from claritymed import config as _cfg
    from claritymed.stores import paths as _paths

    importlib.reload(_cfg)
    importlib.reload(_paths)
    yield data / "jobs"
    importlib.reload(_cfg)
    importlib.reload(_paths)
