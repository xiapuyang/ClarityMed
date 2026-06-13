"""Tests for ``OcrWorker``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claritymed.context import apply_context, reset_context
from claritymed.core.ocr.base import ExtractResult, OcrError, OcrProvider
from claritymed.orchestrator.services.ocr_worker import (
    OcrCompleted,
    OcrJob,
    OcrWorker,
)
from claritymed.orchestrator.services.session_attachments import SessionAttachments
from claritymed.stores.blob_store import BlobStore


class _StubProvider(OcrProvider):
    is_local = True
    label = "stub"

    def __init__(self, *, text: str = "extracted text", raise_with: str | None = None):
        self._text = text
        self._raise = raise_with

    async def extract_text(self, path: Path) -> ExtractResult:
        if self._raise:
            raise OcrError(self._raise)
        return ExtractResult(
            text=self._text, provider_used=self.label, chain_tried=[self.label]
        )


@pytest.fixture
def _ctx():
    tokens = apply_context("20260611000000ABCDEF12", "test", "en")
    yield
    reset_context(tokens)


async def test_extract_done_writes_sentinel(_ctx):
    bs = BlobStore("test")
    sha = bs.store(b"some bytes", "pdf")
    sa = SessionAttachments("test", "sess-1")
    sa.add(sha256=sha, filename="r.pdf", mime="application/pdf", size=10)

    completions: list[OcrCompleted] = []

    async def _listen(c: OcrCompleted) -> None:
        completions.append(c)

    worker = OcrWorker(_StubProvider(text="hello"), listener=_listen)
    worker.start()
    worker.enqueue(
        OcrJob(
            user_id="test",
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
    # Sentinel records the real leaf provider + the chain that was tried,
    # not the wrapping RoutingOcrProvider class name. Regression guard:
    # previously this surfaced as "RoutingOcrProvider" in ocr.json.
    assert sentinel["provider"] == "stub"
    assert sentinel["chain_tried"] == ["stub"]
    # Worker records kind="ocr" + the source extension so the unified
    # reader knows to look at ocr.md (not content.pdf).
    assert sentinel["kind"] == "ocr"
    assert sentinel["ext"] == "pdf"
    assert completions and completions[0].status == "done"
    assert completions[0].provider == "stub"
    # Row was updated in the session tray.
    row = sa.get(sha)
    assert row is not None
    assert row.ocr_status == "done"


async def test_empty_text_marks_empty(_ctx):
    bs = BlobStore("test")
    sha = bs.store(b"some bytes", "pdf")
    sa = SessionAttachments("test", "sess-1")
    sa.add(sha256=sha, filename="r.pdf", mime="application/pdf", size=10)

    completions: list[OcrCompleted] = []
    worker = OcrWorker(_StubProvider(text=""), listener=lambda c: completions.append(c))
    worker.start()
    worker.enqueue(
        OcrJob(
            user_id="test",
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
    bs = BlobStore("test")
    sha = bs.store(b"some bytes", "pdf")
    sa = SessionAttachments("test", "sess-1")
    sa.add(sha256=sha, filename="r.pdf", mime="application/pdf", size=10)

    completions: list[OcrCompleted] = []
    worker = OcrWorker(
        _StubProvider(raise_with="provider go boom"),
        listener=lambda c: completions.append(c),
    )
    worker.start()
    worker.enqueue(
        OcrJob(
            user_id="test",
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
    bs = BlobStore("test")
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
            user_id="test",
            session_id="sess-1",
            sha256=sha,
            blob_path=bs.path(sha, "txt"),
        )
    )
    await worker._queue.join()
    await worker.stop()

    assert completions[0].status == "done"
    assert completions[0].provider == "cache"


async def test_cache_propagates_empty_status(_ctx):
    """A cached ``empty`` sentinel must not be reported as ``done``.

    The old worker synthesized status=done on any cache hit, which lied
    about both empty and failed results. The empty case must keep its
    real status so the UI can show "no text extracted" instead of "ok".
    """
    bs = BlobStore("test")
    sha = bs.store(b"x", "txt")
    bs.ocr_path(sha).write_text("", encoding="utf-8")
    bs.ocr_meta_path(sha).write_text(
        json.dumps({"status": "empty", "provider": "pre-existing"}),
        encoding="utf-8",
    )

    completions: list[OcrCompleted] = []
    worker = OcrWorker(
        _StubProvider(raise_with="should not call"),
        listener=lambda c: completions.append(c),
    )
    worker.start()
    worker.enqueue(
        OcrJob(
            user_id="test",
            session_id="sess-1",
            sha256=sha,
            blob_path=bs.path(sha, "txt"),
        )
    )
    await worker._queue.join()
    await worker.stop()

    assert completions[0].status == "empty"
    assert completions[0].provider == "cache"


async def test_cached_failure_triggers_retry(_ctx):
    """A cached ``failed`` sentinel must NOT short-circuit.

    Otherwise a single bad run (e.g. provider env var not set) keeps the
    sha permanently marked failed even after the cause is fixed — the
    user has no way to retry short of manually deleting the sentinel.
    """
    bs = BlobStore("test")
    sha = bs.store(b"x", "pdf")
    bs.ocr_path(sha).write_text("", encoding="utf-8")
    bs.ocr_meta_path(sha).write_text(
        json.dumps(
            {
                "status": "failed",
                "provider": None,
                "reason": "no OCR providers available for .pdf",
                "chars": 0,
            }
        ),
        encoding="utf-8",
    )

    completions: list[OcrCompleted] = []
    worker = OcrWorker(
        _StubProvider(text="recovered text"),
        listener=lambda c: completions.append(c),
    )
    worker.start()
    worker.enqueue(
        OcrJob(
            user_id="test",
            session_id="sess-1",
            sha256=sha,
            blob_path=bs.path(sha, "pdf"),
        )
    )
    await worker._queue.join()
    await worker.stop()

    assert completions[0].status == "done"
    assert completions[0].provider != "cache"
    # Sentinel overwritten with the fresh result.
    sentinel = json.loads(bs.ocr_meta_path(sha).read_text(encoding="utf-8"))
    assert sentinel["status"] == "done"
    assert sentinel["chars"] == len("recovered text")
