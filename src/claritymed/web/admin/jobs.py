"""Cheap-path background job registry for the admin module.

Two job kinds ship with the admin surface:

* ``rag_ingest`` — ingest one or more documents into a system RAG
  collection (also creates the collection on first ingest if needed)
* ``benchmark_run`` — kick off an eval runner subprocess

Each job is dispatched as an ``asyncio.Task`` running a runner callable.
The registry caps concurrency at :data:`CONCURRENCY_CAP` (2) so two
concurrent benchmark subprocesses can't starve the event loop.

State lives in two places:

* An in-memory ``dict[job_id, JobSpec]`` for fast reads.
* A per-job JSON file at ``DATA_DIR/jobs/<job_id>.json``. Every state
  transition rewrites this file atomically so the registry can recover
  what was running when the process died.

Recovery: on startup, the registry scans ``DATA_DIR/jobs/``. Any spec
marked ``running`` becomes ``crashed`` plus an
``admin.job.crashed_on_restart`` audit event. ``queued`` specs are
discarded — operators re-trigger.

Cleanup: after each ``done`` / ``failed`` / ``cancelled`` / ``crashed``
transition, the registry keeps the most recent 100 finished + everything
in the last 30 days, whichever is larger. Older specs are removed both
in-memory and on disk.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

from claritymed.context import MissingContextError
from claritymed.core.observability.audit import audit_event
from claritymed.stores.paths import job_path, jobs_dir

logger = logging.getLogger(__name__)

CONCURRENCY_CAP = 2
KEEP_FINISHED_COUNT = 100
KEEP_FINISHED_WINDOW_S = 30 * 24 * 60 * 60
STDOUT_TAIL_MAX = 200

JobKind = Literal["rag_ingest", "benchmark_run"]
JobState = Literal["queued", "running", "done", "failed", "cancelled", "crashed"]
TERMINAL_STATES: frozenset[JobState] = frozenset(
    ["done", "failed", "cancelled", "crashed"]
)

JobRunner = Callable[["JobSpec", "JobRegistry"], Awaitable[None]]


def _safe_audit(kind: str, payload: dict[str, Any]) -> None:
    """``audit_event`` that no-ops when the ContextVars are unset.

    Background runners and the startup recovery path don't have a
    request context — we want their state transitions to be visible in
    the audit log when invoked from a real request, but silently
    skipped during tests and lifespan. Same posture as
    :class:`OcrWorker` (which uses ``contextvars.copy_context()`` to
    propagate request context into its tasks).
    """
    try:
        audit_event(kind, payload=payload)  # type: ignore[arg-type]
    except MissingContextError:
        logger.debug("admin.jobs audit %s skipped — no request context", kind)


@dataclass
class JobSpec:
    """One row in the JobRegistry. Mirrored to ``data/jobs/<id>.json``.

    The runner mutates the spec via :meth:`JobRegistry.update` rather than
    writing fields directly so every state transition flushes to disk.
    """

    id: str
    kind: JobKind
    state: JobState
    params: dict[str, Any]
    progress: str = ""
    stdout_tail: list[str] = field(default_factory=list)
    started_at: float | None = None
    finished_at: float | None = None
    exit_code: int | None = None
    error: str | None = None
    cancellable: bool = True
    created_at: float = field(default_factory=time.time)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> JobSpec:
        # Field defaults pick up any new fields a future version adds so
        # an old job file doesn't break recovery.
        return cls(
            id=raw["id"],
            kind=raw["kind"],
            state=raw["state"],
            params=raw.get("params", {}),
            progress=raw.get("progress", ""),
            stdout_tail=list(raw.get("stdout_tail", [])),
            started_at=raw.get("started_at"),
            finished_at=raw.get("finished_at"),
            exit_code=raw.get("exit_code"),
            error=raw.get("error"),
            cancellable=raw.get("cancellable", True),
            created_at=raw.get("created_at", time.time()),
        )


class JobRegistry:
    """In-memory registry mirrored to disk.

    Always instantiate one per FastAPI app (the lifespan attaches it to
    ``app.state.jobs``). The constructor does not run the recovery scan
    — callers do that explicitly via :meth:`recover_on_startup` so test
    fixtures can opt out.
    """

    def __init__(
        self,
        runners: dict[JobKind, JobRunner] | None = None,
        concurrency_cap: int = CONCURRENCY_CAP,
    ) -> None:
        self._jobs: dict[str, JobSpec] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._runners: dict[JobKind, JobRunner] = dict(runners or {})
        self._cap = concurrency_cap
        self._lock = asyncio.Lock()

    # --- runner wiring (per kind) --------------------------------------

    def register_runner(self, kind: JobKind, runner: JobRunner) -> None:
        """Wire a runner callable for ``kind``.

        Later units (U7, U8) call this from lifespan after the registry
        is built so a kind without a registered runner is rejected at
        submit time with a clear error rather than crashing the task.
        """
        self._runners[kind] = runner

    def has_runner(self, kind: JobKind) -> bool:
        return kind in self._runners

    # --- registry queries ----------------------------------------------

    def get(self, job_id: str) -> JobSpec | None:
        return self._jobs.get(job_id)

    def list(
        self,
        state: JobState | None = None,
        kind: JobKind | None = None,
        limit: int | None = None,
    ) -> list[JobSpec]:
        items = list(self._jobs.values())
        if state is not None:
            items = [j for j in items if j.state == state]
        if kind is not None:
            items = [j for j in items if j.kind == kind]
        items.sort(key=lambda j: j.created_at, reverse=True)
        if limit is not None:
            items = items[:limit]
        return items

    def has_running_or_queued(self) -> bool:
        return any(j.state in ("queued", "running") for j in self._jobs.values())

    # --- lifecycle: submit / cancel ------------------------------------

    async def submit(self, kind: JobKind, params: dict[str, Any]) -> JobSpec:
        """Create a job and either dispatch or queue it.

        The decision between ``queued`` and ``running`` is taken under
        ``_lock`` so two concurrent submits don't both think they're
        under the cap.

        Returns the *current* spec from the registry rather than the
        captured pre-dispatch snapshot — :meth:`_maybe_dispatch_locked`
        transitions the spec to ``running`` and sets ``started_at``
        before this call returns, so callers (and the HTTP response)
        see the post-dispatch state.
        """
        if kind not in self._runners:
            raise ValueError(f"no runner registered for kind {kind!r}")
        spec = JobSpec(
            id=str(uuid.uuid4()),
            kind=kind,
            state="queued",
            params=dict(params),
        )
        async with self._lock:
            self._jobs[spec.id] = spec
            self._persist(spec)
            _safe_audit(
                "admin.job.triggered",
                payload={"job_id": spec.id, "kind": kind},
            )
            self._maybe_dispatch_locked()
        return self._jobs[spec.id]

    async def cancel(self, job_id: str) -> JobSpec | None:
        """Cancel a queued or running job. Returns the spec on success."""
        spec = self._jobs.get(job_id)
        if spec is None or spec.state in TERMINAL_STATES:
            return spec
        if not spec.cancellable:
            return spec
        task = self._tasks.get(job_id)
        if task is not None and not task.done():
            task.cancel()
        else:
            # Was queued — never started. Mark cancelled directly.
            self.transition(job_id, "cancelled", finished_at=time.time())
            _safe_audit(
                "admin.job.cancelled",
                payload={"job_id": job_id, "kind": spec.kind, "from": "queued"},
            )
        return self._jobs.get(job_id)

    # --- runner-side state machine helpers -----------------------------

    def transition(self, job_id: str, state: JobState, **updates: Any) -> JobSpec:
        """Atomically update state + arbitrary fields and re-persist.

        Returns the new spec. Callers should not mutate the returned
        object directly — call :meth:`update` for further changes.
        """
        old = self._jobs[job_id]
        new = replace(old, state=state, **updates)
        self._jobs[job_id] = new
        self._persist(new)
        return new

    def update(self, job_id: str, **updates: Any) -> JobSpec:
        return self.transition(job_id, self._jobs[job_id].state, **updates)

    def append_stdout(self, job_id: str, line: str) -> None:
        spec = self._jobs[job_id]
        tail = spec.stdout_tail
        tail.append(line)
        # Trim by mutation; the rewritten spec persists below.
        if len(tail) > STDOUT_TAIL_MAX:
            del tail[: len(tail) - STDOUT_TAIL_MAX]
        self._persist(spec)

    # --- dispatch / cleanup --------------------------------------------

    def _maybe_dispatch_locked(self) -> None:
        running = sum(1 for j in self._jobs.values() if j.state == "running")
        if running >= self._cap:
            return
        next_job = next(
            (
                j
                for j in sorted(self._jobs.values(), key=lambda j: j.created_at)
                if j.state == "queued"
            ),
            None,
        )
        if next_job is None:
            return
        runner = self._runners[next_job.kind]
        next_job = self.transition(next_job.id, "running", started_at=time.time())
        loop = asyncio.get_running_loop()
        task = loop.create_task(self._run(next_job.id, runner))
        self._tasks[next_job.id] = task
        task.add_done_callback(lambda _t: self._tasks.pop(next_job.id, None))

    async def _run(self, job_id: str, runner: JobRunner) -> None:
        spec = self._jobs[job_id]
        try:
            await runner(spec, self)
            current = self._jobs.get(job_id)
            if current and current.state == "running":
                self.transition(job_id, "done", finished_at=time.time())
            _safe_audit(
                "admin.job.completed",
                payload={
                    "job_id": job_id,
                    "kind": spec.kind,
                    "duration_s": time.time() - (spec.started_at or time.time()),
                },
            )
        except asyncio.CancelledError:
            self.transition(job_id, "cancelled", finished_at=time.time())
            _safe_audit(
                "admin.job.cancelled",
                payload={"job_id": job_id, "kind": spec.kind, "from": "running"},
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("admin job %s failed", job_id)
            self.transition(
                job_id,
                "failed",
                finished_at=time.time(),
                error=str(exc),
            )
            _safe_audit(
                "admin.job.failed",
                payload={"job_id": job_id, "kind": spec.kind, "error": str(exc)},
            )
        finally:
            self._cleanup()
            async with self._lock:
                self._maybe_dispatch_locked()

    # --- persistence ----------------------------------------------------

    def _persist(self, spec: JobSpec) -> None:
        path = job_path(spec.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(spec.to_json(), ensure_ascii=False)
        fd, tmp = tempfile.mkstemp(
            prefix=f".{spec.id}.", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(body)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise

    def _drop_from_disk(self, job_id: str) -> None:
        try:
            job_path(job_id).unlink()
        except FileNotFoundError:
            pass

    # --- recovery + cleanup --------------------------------------------

    def recover_on_startup(self) -> None:
        """Scan ``DATA_DIR/jobs/`` and rebuild the in-memory state.

        Anything ``running`` at process death becomes ``crashed`` with a
        matching audit event. ``queued`` specs are discarded.
        """
        d = jobs_dir()
        if not d.exists():
            return
        now = time.time()
        for path in d.glob("*.json"):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning("admin.jobs: failed to parse %s", path)
                continue
            try:
                spec = JobSpec.from_json(raw)
            except KeyError:
                logger.warning("admin.jobs: skipping malformed %s", path)
                continue
            if spec.state == "running":
                spec = replace(
                    spec,
                    state="crashed",
                    finished_at=now,
                    error="server restart during execution",
                )
                self._jobs[spec.id] = spec
                self._persist(spec)
                _safe_audit(
                    "admin.job.crashed_on_restart",
                    payload={"job_id": spec.id, "kind": spec.kind},
                )
            elif spec.state == "queued":
                # Re-trigger is the operator's call; drop the file so it
                # doesn't accumulate.
                self._drop_from_disk(spec.id)
            else:
                self._jobs[spec.id] = spec
        self._cleanup()

    def _cleanup(self) -> None:
        """Keep last 100 finished + everything inside 30-day window."""
        finished = [j for j in self._jobs.values() if j.state in TERMINAL_STATES]
        finished.sort(key=lambda j: j.finished_at or j.created_at, reverse=True)
        keep_ids = {j.id for j in finished[:KEEP_FINISHED_COUNT]}
        cutoff = time.time() - KEEP_FINISHED_WINDOW_S
        for j in finished:
            ref = j.finished_at or j.created_at
            if ref >= cutoff:
                keep_ids.add(j.id)
        for j in list(finished):
            if j.id not in keep_ids:
                self._jobs.pop(j.id, None)
                self._drop_from_disk(j.id)
