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


def test_list_filter_by_state(tmp_jobs_dir):
    """list(state=...) returns only jobs matching that state."""
    registry = JobRegistry()
    now = time.time()
    for sid, st in [("a1", "done"), ("a2", "failed"), ("a3", "done")]:
        spec = JobSpec(id=sid, kind="rag_ingest", state=st, params={}, finished_at=now)
        registry._jobs[sid] = spec
    done = registry.list(state="done")
    assert len(done) == 2
    assert all(j.state == "done" for j in done)


def test_list_filter_by_kind(tmp_jobs_dir):
    """list(kind=...) returns only jobs of that kind."""
    registry = JobRegistry()
    now = time.time()
    for sid, kind in [
        ("b1", "rag_ingest"),
        ("b2", "benchmark_run"),
        ("b3", "rag_ingest"),
    ]:
        spec = JobSpec(id=sid, kind=kind, state="done", params={}, finished_at=now)
        registry._jobs[sid] = spec
    ingest = registry.list(kind="rag_ingest")
    assert len(ingest) == 2
    assert all(j.kind == "rag_ingest" for j in ingest)


async def test_cancel_terminal_job_returns_spec(tmp_jobs_dir):
    """cancel() on a terminal job returns the spec unchanged."""
    registry = JobRegistry()
    registry.register_runner("rag_ingest", _dummy_runner({}))
    spec = await registry.submit("rag_ingest", {})
    # Wait for it to finish.
    for _ in range(50):
        await asyncio.sleep(0.01)
        if registry.get(spec.id).state == "done":
            break
    result = await registry.cancel(spec.id)
    assert result is not None
    assert result.state == "done"


async def test_cancel_non_cancellable_job_returns_spec(tmp_jobs_dir):
    """cancel() on a non-cancellable job returns the spec without cancelling."""
    from claritymed.web.admin.jobs import JobSpec as JS

    registry = JobRegistry()
    spec = JS(
        id="nc-001", kind="rag_ingest", state="running", params={}, cancellable=False
    )
    registry._jobs[spec.id] = spec
    result = await registry.cancel(spec.id)
    assert result is not None
    assert result.state == "running"


def test_append_stdout_trims_on_overflow(tmp_jobs_dir):
    """append_stdout keeps only STDOUT_TAIL_MAX lines when the tail overflows."""
    from claritymed.web.admin.jobs import STDOUT_TAIL_MAX

    registry = JobRegistry()
    spec = JobSpec(id="stdout-001", kind="rag_ingest", state="running", params={})
    registry._jobs[spec.id] = spec
    registry._persist(spec)
    for i in range(STDOUT_TAIL_MAX + 5):
        registry.append_stdout(spec.id, f"line {i}")
    tail = registry.get(spec.id).stdout_tail
    assert len(tail) <= STDOUT_TAIL_MAX


def test_drop_from_disk_handles_missing_file(tmp_jobs_dir):
    """_drop_from_disk does not raise when the file has already been deleted."""
    registry = JobRegistry()
    # Call with a non-existent job id — the file doesn't exist but should not raise.
    registry._drop_from_disk("nonexistent-job-id-xyz")


def test_recover_on_startup_skips_malformed_json(tmp_jobs_dir):
    """A corrupt JSON file in the jobs dir is skipped, not crashing recovery."""
    from claritymed.stores.paths import job_path

    bad_path = job_path("bad-json-job-id")
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text("{not-json", encoding="utf-8")
    registry = JobRegistry()
    registry.recover_on_startup()
    assert registry.get("bad-json-job-id") is None


def test_recover_on_startup_skips_malformed_spec(tmp_jobs_dir):
    """A JSON file with missing required fields is skipped (KeyError path)."""
    from claritymed.stores.paths import job_path

    bad_path = job_path("bad-spec-job-id")
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text('{"some": "garbage"}', encoding="utf-8")
    registry = JobRegistry()
    registry.recover_on_startup()
    assert registry.get("bad-spec-job-id") is None


def test_recover_on_startup_restores_finished_job(tmp_jobs_dir):
    """A 'done' job on disk is loaded back into the registry."""
    from claritymed.stores.paths import job_path

    spec = JobSpec(
        id="done-restore-001",
        kind="rag_ingest",
        state="done",
        params={},
        finished_at=time.time(),
    )
    job_path(spec.id).parent.mkdir(parents=True, exist_ok=True)
    job_path(spec.id).write_text(json.dumps(spec.to_json()), encoding="utf-8")
    registry = JobRegistry()
    registry.recover_on_startup()
    restored = registry.get(spec.id)
    assert restored is not None
    assert restored.state == "done"


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
