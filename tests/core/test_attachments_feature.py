"""Unit tests for ``AttachmentsFeature``.

The E2E flow (paste → OCR → feature) is covered by
``tests/e2e/test_clipboard_to_tools.py``; this file focuses on the
per-status render branches so future regressions in the "failed" or
"pending" messages fail fast without dragging in the OcrWorker."""

from __future__ import annotations

import json

import pytest

from claritymed.context import apply_context, reset_context
from claritymed.core.attachments_feature import AttachmentsFeature
from claritymed.orchestrator.services.session_attachments import SessionAttachments
from claritymed.stores.blob_store import BlobStore


_USER_ID = "alice"
_SESSION_ID = "sess-feature-unit"


@pytest.fixture
def _ctx():
    tokens = apply_context("20260611000000ABCDEF12", _USER_ID, "en")
    yield
    reset_context(tokens)


def _make_ctx_obj(language: str = "en"):
    """Build a minimal TurnContext-compatible duck-type for the feature."""
    lang = language

    class _Deps:
        user_id = _USER_ID

    deps = _Deps()
    deps.language = lang

    class _Ctx:
        pass

    ctx = _Ctx()
    ctx.deps = deps
    ctx.scrubbed = ""
    return ctx


def _seed_ocr_done(sha: str, text: str) -> None:
    """Write the ``ocr.md`` + ``ocr.json`` sentinel that ``BlobStore.ocr_done`` checks."""
    bs = BlobStore(_USER_ID)
    bs.ocr_path(sha).parent.mkdir(parents=True, exist_ok=True)
    bs.ocr_path(sha).write_text(text, encoding="utf-8")
    bs.ocr_meta_path(sha).write_text(json.dumps({"status": "done"}), encoding="utf-8")


async def test_no_session_returns_empty(_ctx):
    feature = AttachmentsFeature(get_session_id=lambda: None)
    assert await feature.pre_invoke(_make_ctx_obj()) == ""


async def test_no_attachments_returns_empty(_ctx):
    # Make sure the session exists but holds no rows.
    SessionAttachments(_USER_ID, _SESSION_ID)
    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    assert await feature.pre_invoke(_make_ctx_obj()) == ""


async def test_done_status_renders_ocr_text(_ctx):
    bs = BlobStore(_USER_ID)
    sha = bs.store(b"abc", "png")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha, filename="x.png", mime="image/png", size=3, source="paste"
    )
    SessionAttachments(_USER_ID, _SESSION_ID).mark_ocr_status(
        sha, "done", provider="StubProvider"
    )
    _seed_ocr_done(sha, "extracted body")

    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    block = await feature.pre_invoke(_make_ctx_obj())
    assert "Attachments (extracted by OCR):" in block
    assert "extracted body" in block
    assert "x.png" in block


async def test_zh_header_when_language_is_zh(_ctx):
    bs = BlobStore(_USER_ID)
    sha = bs.store(b"abc", "png")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha, filename="x.png", mime="image/png", size=3, source="paste"
    )
    SessionAttachments(_USER_ID, _SESSION_ID).mark_ocr_status(sha, "done")
    _seed_ocr_done(sha, "提取的内容")

    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    block = await feature.pre_invoke(_make_ctx_obj(language="zh"))
    assert "附件（OCR 提取的内容）" in block


async def test_pending_status_renders_in_progress(_ctx):
    bs = BlobStore(_USER_ID)
    sha = bs.store(b"abc", "png")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha, filename="report.png", mime="image/png", size=3, source="paste"
    )
    # Default status is ``pending``.

    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    block = await feature.pre_invoke(_make_ctx_obj())
    assert "OCR in progress" in block


async def test_failed_status_renders_reason(_ctx):
    bs = BlobStore(_USER_ID)
    sha = bs.store(b"abc", "png")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha, filename="bad.png", mime="image/png", size=3, source="paste"
    )
    SessionAttachments(_USER_ID, _SESSION_ID).mark_ocr_status(
        sha, "failed", reason="tesseract not found"
    )

    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    block = await feature.pre_invoke(_make_ctx_obj())
    assert "OCR failed: tesseract not found" in block


async def test_empty_status_renders_no_text(_ctx):
    bs = BlobStore(_USER_ID)
    sha = bs.store(b"abc", "png")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha, filename="blank.png", mime="image/png", size=3, source="paste"
    )
    SessionAttachments(_USER_ID, _SESSION_ID).mark_ocr_status(sha, "empty")

    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    block = await feature.pre_invoke(_make_ctx_obj())
    assert "OCR returned no text" in block


async def test_done_but_missing_ocr_file_renders_missing_message(_ctx):
    """If a status row says ``done`` but ``ocr.md`` was lost on disk,
    surface that explicitly instead of silently skipping — silent
    skipping would mask a disk-corruption bug."""
    bs = BlobStore(_USER_ID)
    sha = bs.store(b"abc", "png")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha, filename="ghost.png", mime="image/png", size=3, source="paste"
    )
    SessionAttachments(_USER_ID, _SESSION_ID).mark_ocr_status(sha, "done")
    # Status says done but the file was never written — simulate the
    # corruption / partial-write scenario the branch defends against.

    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    block = await feature.pre_invoke(_make_ctx_obj())
    assert "OCR text missing on disk" in block


async def test_done_with_empty_text_renders_empty_message(_ctx):
    bs = BlobStore(_USER_ID)
    sha = bs.store(b"abc", "png")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha, filename="blanky.png", mime="image/png", size=3, source="paste"
    )
    SessionAttachments(_USER_ID, _SESSION_ID).mark_ocr_status(sha, "done")
    _seed_ocr_done(sha, "   ")  # whitespace-only

    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    block = await feature.pre_invoke(_make_ctx_obj())
    assert "OCR returned empty text" in block


async def test_session_attachments_failure_returns_empty(_ctx, monkeypatch):
    """A broken SessionAttachments path must not crash the turn — the
    plugin returns an empty block and lets the LLM answer without
    attachments. (Robustness: an OS-level read error mid-question
    shouldn't take the whole answer down.)"""

    def _boom(self):
        raise OSError("disk gone")

    monkeypatch.setattr(SessionAttachments, "list", _boom)
    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    assert await feature.pre_invoke(_make_ctx_obj()) == ""
