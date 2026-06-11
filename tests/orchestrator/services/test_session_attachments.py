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
