"""Tests for ``SessionAttachments`` IO."""

from __future__ import annotations

import pytest

from claritymed.context import apply_context, reset_context
from claritymed.orchestrator.services.session_attachments import SessionAttachments


SHA_A = "a" * 64
SHA_B = "b" * 64


@pytest.fixture
def _ctx():
    tokens = apply_context("20260611000000ABCDEF12", "alice", "en")
    yield
    reset_context(tokens)


def test_add_and_list(_ctx):
    sa = SessionAttachments("alice", "sess-1")
    sa.add(sha256=SHA_A, filename="r.pdf", mime="application/pdf", size=1234)
    rows = sa.list()
    assert len(rows) == 1
    assert rows[0].sha256 == SHA_A
    assert rows[0].ocr_status == "pending"


def test_add_same_sha_is_idempotent(_ctx):
    sa = SessionAttachments("alice", "sess-1")
    sa.add(sha256=SHA_A, filename="r.pdf", mime="application/pdf", size=1234)
    sa.add(sha256=SHA_A, filename="renamed.pdf", mime="application/pdf", size=1234)
    rows = sa.list()
    assert len(rows) == 1
    assert rows[0].filename == "renamed.pdf"


def test_mark_ocr_status_updates_row(_ctx):
    sa = SessionAttachments("alice", "sess-1")
    sa.add(sha256=SHA_A, filename="r.pdf", mime="application/pdf", size=10)
    sa.mark_ocr_status(SHA_A, "done", provider="pymupdf")
    row = sa.get(SHA_A)
    assert row.ocr_status == "done"
    assert row.ocr_provider == "pymupdf"


def test_mark_ocr_status_missing_returns_none(_ctx):
    sa = SessionAttachments("alice", "sess-1")
    assert sa.mark_ocr_status(SHA_B, "done") is None


def test_to_manifest_attachments_promotes_rows(_ctx):
    sa = SessionAttachments("alice", "sess-1")
    sa.add(sha256=SHA_A, filename="r.pdf", mime="application/pdf", size=1234)
    sa.add(sha256=SHA_B, filename="x.png", mime="image/png", size=500)
    out = sa.to_manifest_attachments()
    assert len(out) == 2
    assert {a.sha256 for a in out} == {SHA_A, SHA_B}


def test_two_session_ids_are_isolated(_ctx):
    a = SessionAttachments("alice", "sess-1")
    b = SessionAttachments("alice", "sess-2")
    a.add(sha256=SHA_A, filename="r.pdf", mime="application/pdf", size=1)
    assert b.list() == []


# ---------------------------------------------------------------------------
# AskService._await_pending_ocr — turn waits for OCR before proceeding
# ---------------------------------------------------------------------------


import asyncio  # noqa: E402

from claritymed.stores.blob_store import BlobStore  # noqa: E402


def _make_ask_service():
    """Construct AskService skipping __init__.

    ``_await_pending_ocr`` only touches ``self`` to call no methods,
    so a bare instance is enough — full init pulls in PHI guard /
    model build / RAG strategy which we don't need to exercise.
    """
    from claritymed.orchestrator.services.ask_service import AskService

    return AskService.__new__(AskService)


async def test_await_pending_ocr_returns_immediately_when_no_pending(_ctx):
    """No attachments → no wait. Also no attachments rows → still no wait."""
    svc = _make_ask_service()
    import time as _t

    t0 = _t.monotonic()
    await svc._await_pending_ocr("alice", "sess-empty")
    assert _t.monotonic() - t0 < 0.5  # well under one poll interval


async def test_await_pending_ocr_unblocks_when_sentinel_lands(_ctx, monkeypatch):
    """Add a pending attachment; in the background, write the OCR sentinel
    after a short delay. The wait must return shortly after — proving the
    polling loop actually observes BlobStore.ocr_done flipping to True.

    Regression: prior to this wait, pressing Enter during OCR meant the
    LLM saw ``OCR in progress`` and answered without the extracted text.
    """
    # Shrink the poll interval so the test isn't dominated by it.
    monkeypatch.setattr(
        "claritymed.orchestrator.services.ask_service.OCR_AWAIT_POLL_S", 0.02
    )

    sa = SessionAttachments("alice", "sess-await")
    sha = "c" * 64
    sa.add(sha256=sha, filename="img.png", mime="image/png", size=10)

    # Plant a real blob so ocr_path / ocr_meta_path resolve to real files
    # the sentinel write below can land on.
    blob_store = BlobStore("alice")
    blob_store.store(b"fake png bytes", "png")

    async def _land_sentinel():
        await asyncio.sleep(0.1)
        ocr_md = blob_store.ocr_path(sha)
        ocr_md.parent.mkdir(parents=True, exist_ok=True)
        ocr_md.write_text("extracted text", encoding="utf-8")
        # ocr.json sentinel is what ocr_done checks.
        import json as _json

        ocr_json = blob_store.ocr_meta_path(sha)
        ocr_json.write_text(
            _json.dumps({"status": "done", "provider": "stub", "chars": 14}),
            encoding="utf-8",
        )

    svc = _make_ask_service()
    import time as _t

    t0 = _t.monotonic()
    await asyncio.gather(
        svc._await_pending_ocr("alice", "sess-await"),
        _land_sentinel(),
    )
    elapsed = _t.monotonic() - t0
    assert 0.05 < elapsed < 2.0, (
        elapsed
    )  # unblocked after sentinel, well before timeout


async def test_await_pending_ocr_returns_on_timeout(_ctx, monkeypatch):
    """A pending attachment that never settles must not hang the turn —
    after OCR_AWAIT_TIMEOUT_S we proceed so AttachmentsFeature can render
    ``OCR in progress`` to the LLM rather than dead-ending the user.
    """
    monkeypatch.setattr(
        "claritymed.orchestrator.services.ask_service.OCR_AWAIT_TIMEOUT_S", 0.2
    )
    monkeypatch.setattr(
        "claritymed.orchestrator.services.ask_service.OCR_AWAIT_POLL_S", 0.05
    )

    sa = SessionAttachments("alice", "sess-stuck")
    sa.add(sha256="d" * 64, filename="x.png", mime="image/png", size=4)
    # Never write the sentinel — simulates a stuck provider.

    svc = _make_ask_service()
    import time as _t

    t0 = _t.monotonic()
    await svc._await_pending_ocr("alice", "sess-stuck")
    elapsed = _t.monotonic() - t0
    assert 0.2 <= elapsed < 1.0, elapsed
