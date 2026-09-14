"""Tests for ``claritymed.core.locks``."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest

from claritymed.core.locks import (
    LockTimeout,
    _reset_thread_locks_for_tests,
    file_lock,
    process_and_thread_lock,
)


@pytest.fixture(autouse=True)
def _reset_locks():
    _reset_thread_locks_for_tests()
    yield
    _reset_thread_locks_for_tests()


def test_file_lock_basic(tmp_path: Path):
    """The lock primitive protects a sibling ``.lock`` file; the data
    file is what the caller writes inside the lock."""
    data = tmp_path / "guarded.txt"
    lock_path = tmp_path / "guarded.txt.lock"
    with file_lock(lock_path):
        data.write_text("hello", encoding="utf-8")
    assert data.read_text(encoding="utf-8") == "hello"


def test_file_lock_creates_parent_directory(tmp_path: Path):
    target = tmp_path / "nested" / "subdir" / "a.lock"
    with file_lock(target):
        pass
    assert target.parent.is_dir()


def test_file_lock_blocks_when_held(tmp_path: Path):
    """filelock is process-scoped: a separate thread that uses a *new*
    FileLock instance on the same path should be blocked until release.

    NOTE: filelock on POSIX is advisory at process level, but multiple
    FileLock instances inside the same process queue properly via its
    internal counter, so this also exercises the same-process contention
    path (which is the only thing we can portably test in CI)."""
    target = tmp_path / "guarded.target.lock"

    holder_acquired = threading.Event()
    holder_release = threading.Event()
    contender_acquired = threading.Event()

    def holder():
        with file_lock(target, timeout=5.0):
            holder_acquired.set()
            holder_release.wait(timeout=5.0)

    def contender():
        with file_lock(target, timeout=5.0):
            contender_acquired.set()

    t1 = threading.Thread(target=holder)
    t1.start()
    holder_acquired.wait(timeout=5.0)

    t2 = threading.Thread(target=contender)
    t2.start()
    # Contender should NOT have acquired yet (lock is held).
    time.sleep(0.1)
    assert not contender_acquired.is_set()

    # Release; contender should now make progress.
    holder_release.set()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)
    assert contender_acquired.is_set()


def test_file_lock_timeout_raises_typed_error(tmp_path: Path):
    """A second acquirer with a short timeout must raise ``LockTimeout``."""
    target = tmp_path / "guarded.lock"

    holder_acquired = threading.Event()
    holder_release = threading.Event()
    failure: list[BaseException] = []

    def holder():
        with file_lock(target, timeout=5.0):
            holder_acquired.set()
            holder_release.wait(timeout=5.0)

    def contender():
        try:
            with file_lock(target, timeout=0.2):
                pass
        except BaseException as exc:  # noqa: BLE001
            failure.append(exc)

    t1 = threading.Thread(target=holder)
    t1.start()
    holder_acquired.wait(timeout=5.0)

    t2 = threading.Thread(target=contender)
    t2.start()
    t2.join(timeout=5.0)
    holder_release.set()
    t1.join(timeout=5.0)

    assert failure, "contender should have raised LockTimeout"
    assert isinstance(failure[0], LockTimeout)


async def test_process_and_thread_lock_serializes_coroutines(tmp_path: Path):
    """Two coroutines on the same path should be serialized in-process."""
    target = tmp_path / "guarded.lock"
    order: list[str] = []

    async def worker(name: str, delay: float):
        async with process_and_thread_lock(target, timeout=5.0):
            order.append(f"{name}:enter")
            await asyncio.sleep(delay)
            order.append(f"{name}:exit")

    await asyncio.gather(worker("A", 0.05), worker("B", 0.05))
    # Order must be A-enter/A-exit/B-enter/B-exit or the reverse; never
    # interleaved (which would happen without serialization).
    assert order in (
        ["A:enter", "A:exit", "B:enter", "B:exit"],
        ["B:enter", "B:exit", "A:enter", "A:exit"],
    )


async def test_process_and_thread_lock_releases_on_exception(tmp_path: Path):
    """An exception inside the body must not strand the lock acquired."""
    target = tmp_path / "guarded.lock"

    with pytest.raises(RuntimeError):
        async with process_and_thread_lock(target, timeout=5.0):
            raise RuntimeError("bang")

    # Second acquire must succeed quickly — the first one released.
    async with process_and_thread_lock(target, timeout=1.0):
        pass
