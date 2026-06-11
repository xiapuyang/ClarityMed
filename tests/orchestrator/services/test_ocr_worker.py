"""Tests for ``OcrWorker``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claritymed.context import apply_context, reset_context
from claritymed.core.ocr.base import OcrError, OcrProvider
from claritymed.orchestrator.services.ocr_worker import (
    OcrCompleted,
    OcrJob,
    OcrWorker,
)
from claritymed.orchestrator.services.session_attachments import SessionAttachments
from claritymed.stores.blob_store import BlobStore


class _StubProvider(OcrProvider):
    is_local = True

    def __init__(self, *, text: str = "extracted text", raise_with: str | None = None):
        self._text = text
        self._raise = raise_with

    async def extract_text(self, path: Path) -> str:
        if self._raise:
            raise OcrError(self._raise)
        return self._text


@pytest.fixture
def _ctx():
    tokens = apply_context("20260611000000ABCDEF12", "alice", "en")
    yield
    reset_context(tokens)


async def test_extract_done_writes_sentinel(_ctx):
    bs = BlobStore("alice")
    sha = bs.store(b"some bytes", "pdf")
    sa = SessionAttachments("alice", "sess-1")
    sa.add(sha256=sha, filename="r.pdf", mime="application/pdf", size=10)

    completions: list[OcrCompleted] = []

    async def _listen(c: OcrCompleted) -> None:
        completions.append(c)

    worker = OcrWorker(_StubProvider(text="hello"), listener=_listen)
    worker.start()
    worker.enqueue(
        OcrJob(
            user_id="alice",
            session_id="sess-1",
            sha256=sha,
            blob_path=bs.path(sha, "pdf"),
        )
    )
    await worker._queue.join()
    await worker.stop()

    assert bs.ocr_done(sha)
    sentinel = json.loads(bs.ocr_meta_path(sha).read_text(encoding="utf-8"))
    assert sentinel["status"] == "done"
    assert completions and completions[0].status == "done"
    # Row was updated in the session tray.
    row = sa.get(sha)
    assert row is not None
    assert row.ocr_status == "done"


async def test_empty_text_marks_empty(_ctx):
    bs = BlobStore("alice")
    sha = bs.store(b"some bytes", "pdf")
    sa = SessionAttachments("alice", "sess-1")
    sa.add(sha256=sha, filename="r.pdf", mime="application/pdf", size=10)

    completions: list[OcrCompleted] = []
    worker = OcrWorker(_StubProvider(text=""), listener=lambda c: completions.append(c))
    worker.start()
    worker.enqueue(
        OcrJob(
            user_id="alice",
            session_id="sess-1",
            sha256=sha,
            blob_path=bs.path(sha, "pdf"),
        )
    )
    await worker._queue.join()
    await worker.stop()

    sentinel = json.loads(bs.ocr_meta_path(sha).read_text(encoding="utf-8"))
    assert sentinel["status"] == "empty"
    assert completions[0].status == "empty"


async def test_provider_failure_marks_failed(_ctx):
    bs = BlobStore("alice")
    sha = bs.store(b"some bytes", "pdf")
    sa = SessionAttachments("alice", "sess-1")
    sa.add(sha256=sha, filename="r.pdf", mime="application/pdf", size=10)

    completions: list[OcrCompleted] = []
    worker = OcrWorker(
        _StubProvider(raise_with="provider go boom"),
        listener=lambda c: completions.append(c),
    )
    worker.start()
    worker.enqueue(
        OcrJob(
            user_id="alice",
            session_id="sess-1",
            sha256=sha,
            blob_path=bs.path(sha, "pdf"),
        )
    )
    await worker._queue.join()
    await worker.stop()

    sentinel = json.loads(bs.ocr_meta_path(sha).read_text(encoding="utf-8"))
    assert sentinel["status"] == "failed"
    assert completions[0].status == "failed"
    row = sa.get(sha)
    assert row.ocr_status == "failed"


async def test_cache_short_circuit_when_sentinel_present(_ctx):
    """An ``ocr.json`` already on disk → worker skips extraction and emits
    a synthetic ``done`` event with provider='cache'."""
    bs = BlobStore("alice")
    sha = bs.store(b"x", "txt")
    bs.ocr_path(sha).write_text("cached text", encoding="utf-8")
    bs.ocr_meta_path(sha).write_text(
        json.dumps({"status": "done", "provider": "pre-existing"}),
        encoding="utf-8",
    )

    completions: list[OcrCompleted] = []
    # Stub will raise if it's actually called — proving the cache hit.
    worker = OcrWorker(
        _StubProvider(raise_with="should not call"),
        listener=lambda c: completions.append(c),
    )
    worker.start()
    worker.enqueue(
        OcrJob(
            user_id="alice",
            session_id="sess-1",
            sha256=sha,
            blob_path=bs.path(sha, "txt"),
        )
    )
    await worker._queue.join()
    await worker.stop()

    assert completions[0].status == "done"
    assert completions[0].provider == "cache"
