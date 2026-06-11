"""Cross-platform file locking primitive used by every multi-writer store.

The brainstorm sketched ``fcntl.flock`` but the project ships an Inno Setup
installer (Windows is in scope) and ``fcntl`` is POSIX-only. ``filelock``
abstracts ``fcntl`` / ``msvcrt`` with a single context-manager API at the
cost of one ~30KB MIT-licensed dependency.

Two primitives are exported:

* :func:`file_lock` — raw ``filelock.FileLock`` wrapper. Works across
  processes. **Does not protect against same-process threads** because
  ``filelock`` advisory locks are process-scoped on POSIX; two threads in
  the same process both succeed in acquiring.
* :func:`process_and_thread_lock` — composes an in-process
  ``asyncio.Lock`` keyed by the lock path with a process-level
  ``file_lock``. Use this anywhere a single Python process might run
  concurrent coroutines that touch the same on-disk file (TUI + headless
  CLI sharing a TUI session via subprocess, AskService rewrites of
  ``attachments.json`` from background OCR + foreground turn, etc.).

NFS / SMB are explicitly out of scope for v1; ``filelock``'s POSIX advisory
locks misbehave on networked filesystems and we don't try to remediate.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import AsyncIterator, Iterator

from filelock import FileLock, Timeout as FileLockTimeout


class LockTimeout(RuntimeError):
    """Raised when a file lock could not be acquired within the timeout.

    Wraps both ``filelock.Timeout`` (process-level) and the asyncio
    ``asyncio.TimeoutError`` (in-process) so callers branch on one type
    rather than two. The message includes the path for grep-ability.
    """


_thread_locks: dict[str, asyncio.Lock] = {}


def _thread_lock_for(path: Path) -> asyncio.Lock:
    """Return the process-wide ``asyncio.Lock`` keyed by ``path``.

    Same path → same Lock; different path → different Lock. Lazily
    instantiated. The dict grows monotonically — we never delete entries
    because the cost is a few hundred bytes per distinct path and the
    test-isolation fixture clears it on session boundary anyway.
    """
    key = str(path)
    lock = _thread_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _thread_locks[key] = lock
    return lock


def _reset_thread_locks_for_tests() -> None:
    """Clear the in-process Lock cache. Test-isolation fixture calls this."""
    _thread_locks.clear()


@contextlib.contextmanager
def file_lock(path: Path | str, timeout: float = 10.0) -> Iterator[None]:
    """Acquire a cross-process advisory lock at ``path``.

    The lock file lives alongside the file it protects (e.g.
    ``manifest.yaml.lock`` next to ``manifest.yaml``); ``filelock`` creates
    the lock file lazily and leaves it behind on release (intentional —
    re-acquiring is faster, and the file is empty).

    Same-process threads are NOT serialized by this primitive. Use
    :func:`process_and_thread_lock` for that.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(target), timeout=timeout)
    try:
        lock.acquire()
    except FileLockTimeout as exc:
        raise LockTimeout(f"timed out acquiring {target}") from exc
    try:
        yield
    finally:
        lock.release()


@contextlib.asynccontextmanager
async def process_and_thread_lock(
    path: Path | str, timeout: float = 10.0
) -> AsyncIterator[None]:
    """Acquire (in-process asyncio Lock keyed by ``path``) AND (process-level file lock).

    Order matters: the asyncio Lock is acquired first so a coroutine queue
    forms inside one process before any of them attempts to grab the OS
    advisory lock. That keeps the file-lock contention to one outstanding
    waiter per process, which avoids unfair starvation across many
    coroutines.

    Both halves are released in ``finally`` even when the body raises.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tlock = _thread_lock_for(target)
    try:
        await asyncio.wait_for(tlock.acquire(), timeout=timeout)
    except asyncio.TimeoutError as exc:  # pragma: no cover - rare under tests
        raise LockTimeout(f"timed out on in-process lock for {target}") from exc
    try:
        flock = FileLock(str(target), timeout=timeout)
        try:
            # filelock has no native async API; the file_lock acquire is
            # blocking but is bounded by ``timeout``. asyncio is fine with
            # short bounded sync work — long blocking work would belong in
            # ``asyncio.to_thread``.
            flock.acquire()
        except FileLockTimeout as exc:
            raise LockTimeout(f"timed out acquiring {target}") from exc
        try:
            yield
        finally:
            flock.release()
    finally:
        tlock.release()
