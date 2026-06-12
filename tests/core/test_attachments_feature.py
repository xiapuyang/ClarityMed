"""Unit tests for ``AttachmentsFeature``.

The E2E flow (paste → OCR → feature) is covered by
``tests/e2e/test_clipboard_to_tools.py``; this file focuses on the
placeholder-expansion logic per OCR status so future regressions in the
inline tag formatting (and the "delete the placeholder = skip OCR"
semantic) fail fast without dragging in the OcrWorker.
"""

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
    """Build a minimal ``TurnContext``-compatible duck-type for the feature."""
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


def _placeholder(sha: str, kind: str = "Image") -> str:
    return f"[{kind} sha:{sha[:8]}]"


# ----- pre_invoke is now a no-op ---------------------------------------


async def test_pre_invoke_returns_empty_no_session(_ctx):
    """``pre_invoke`` always returns empty — content lives inline now."""
    feature = AttachmentsFeature(get_session_id=lambda: None)
    assert await feature.pre_invoke(_make_ctx_obj()) == ""


async def test_pre_invoke_returns_empty_with_done_attachment(_ctx):
    """Even with a fully-OCR'd attachment, ``pre_invoke`` is empty —
    the OCR text only appears via inline placeholder expansion."""
    bs = BlobStore(_USER_ID)
    sha = bs.store(b"abc", "png")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha, filename="x.png", mime="image/png", size=3, source="paste"
    )
    SessionAttachments(_USER_ID, _SESSION_ID).mark_ocr_status(sha, "done")
    _seed_ocr_done(sha, "extracted body")

    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    assert await feature.pre_invoke(_make_ctx_obj()) == ""


# ----- expand_placeholders: happy path ---------------------------------


async def test_done_status_inlines_ocr_inside_image_tag(_ctx):
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
    text = f"check this {_placeholder(sha)} please"
    out = await feature.expand_placeholders(text, _make_ctx_obj())

    assert f'<image sha="{sha}">' in out
    assert "extracted body" in out
    assert "</image>" in out
    # Surrounding user text is preserved verbatim.
    assert out.startswith("check this ")
    assert out.endswith(" please")
    # The placeholder itself is gone — replaced by the inline tag.
    assert _placeholder(sha) not in out


async def test_file_kind_renders_file_tag(_ctx):
    bs = BlobStore(_USER_ID)
    sha = bs.store(b"pdf-bytes", "pdf")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha,
        filename="report.pdf",
        mime="application/pdf",
        size=9,
        source="paste",
    )
    SessionAttachments(_USER_ID, _SESSION_ID).mark_ocr_status(sha, "done")
    _seed_ocr_done(sha, "Lab Report — patient X")

    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    text = _placeholder(sha, kind="File")
    out = await feature.expand_placeholders(text, _make_ctx_obj())

    assert f'<file sha="{sha}">' in out
    assert "Lab Report" in out
    assert "</file>" in out


async def test_two_placeholders_both_expand(_ctx):
    bs = BlobStore(_USER_ID)
    sha_a = bs.store(b"aaa-content", "png")
    sha_b = bs.store(b"bbb-content", "png")
    session = SessionAttachments(_USER_ID, _SESSION_ID)
    for sha, name, body in [
        (sha_a, "a.png", "first OCR"),
        (sha_b, "b.png", "second OCR"),
    ]:
        session.add(
            sha256=sha, filename=name, mime="image/png", size=11, source="paste"
        )
        session.mark_ocr_status(sha, "done")
        _seed_ocr_done(sha, body)

    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    text = f"compare {_placeholder(sha_a)} vs {_placeholder(sha_b)}"
    out = await feature.expand_placeholders(text, _make_ctx_obj())

    assert "first OCR" in out
    assert "second OCR" in out
    assert f'<image sha="{sha_a}">' in out
    assert f'<image sha="{sha_b}">' in out


# ----- expand_placeholders: deletion semantic --------------------------


async def test_session_attachment_not_in_text_is_skipped(_ctx):
    """User pastes an image, then deletes the placeholder before sending.
    The attachment stays in the session index but **no** OCR content
    appears in the expanded prompt. This is the load-bearing UX gesture:
    deleting the placeholder = "don't include this image this turn."""
    bs = BlobStore(_USER_ID)
    sha = bs.store(b"abc", "png")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha, filename="x.png", mime="image/png", size=3, source="paste"
    )
    SessionAttachments(_USER_ID, _SESSION_ID).mark_ocr_status(sha, "done")
    _seed_ocr_done(sha, "secret content the user didn't want sent")

    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    text = "Hi, just a quick question with no placeholder."
    out = await feature.expand_placeholders(text, _make_ctx_obj())

    assert out == text  # untouched
    assert "secret content" not in out


async def test_no_text_returns_input_unchanged(_ctx):
    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    assert await feature.expand_placeholders("", _make_ctx_obj()) == ""


async def test_no_session_returns_input_unchanged(_ctx):
    feature = AttachmentsFeature(get_session_id=lambda: None)
    text = "Hello [Image sha:deadbeef]"
    assert await feature.expand_placeholders(text, _make_ctx_obj()) == text


# ----- expand_placeholders: status branches ----------------------------


async def test_pending_status_renders_self_closing_tag(_ctx):
    bs = BlobStore(_USER_ID)
    sha = bs.store(b"abc", "png")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha, filename="report.png", mime="image/png", size=3, source="paste"
    )
    # Default status is ``pending``.

    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    out = await feature.expand_placeholders(_placeholder(sha), _make_ctx_obj())
    assert out == f'<image sha="{sha}" ocr_status="pending"/>'


async def test_failed_status_renders_reason_attribute(_ctx):
    bs = BlobStore(_USER_ID)
    sha = bs.store(b"abc", "png")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha, filename="bad.png", mime="image/png", size=3, source="paste"
    )
    SessionAttachments(_USER_ID, _SESSION_ID).mark_ocr_status(
        sha, "failed", reason="tesseract not found"
    )

    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    out = await feature.expand_placeholders(_placeholder(sha), _make_ctx_obj())
    assert (
        out == f'<image sha="{sha}" ocr_status="failed" reason="tesseract not found"/>'
    )


async def test_failed_reason_with_quotes_is_escaped(_ctx):
    bs = BlobStore(_USER_ID)
    sha = bs.store(b"abc", "png")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha, filename="bad.png", mime="image/png", size=3, source="paste"
    )
    SessionAttachments(_USER_ID, _SESSION_ID).mark_ocr_status(
        sha, "failed", reason='server said "down"'
    )

    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    out = await feature.expand_placeholders(_placeholder(sha), _make_ctx_obj())
    assert 'reason="server said &quot;down&quot;"' in out


async def test_empty_status_renders_self_closing_tag(_ctx):
    bs = BlobStore(_USER_ID)
    sha = bs.store(b"abc", "png")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha, filename="blank.png", mime="image/png", size=3, source="paste"
    )
    SessionAttachments(_USER_ID, _SESSION_ID).mark_ocr_status(sha, "empty")

    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    out = await feature.expand_placeholders(_placeholder(sha), _make_ctx_obj())
    assert out == f'<image sha="{sha}" ocr_status="empty"/>'


async def test_done_but_missing_ocr_file_renders_missing_status(_ctx):
    """Sentinel says ``done`` but ``ocr.md`` was lost on disk → surface
    explicitly as ``ocr_status="missing"`` so a silent drop can't mask
    a disk-corruption bug."""
    bs = BlobStore(_USER_ID)
    sha = bs.store(b"abc", "png")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha, filename="ghost.png", mime="image/png", size=3, source="paste"
    )
    SessionAttachments(_USER_ID, _SESSION_ID).mark_ocr_status(sha, "done")
    # Status says done but the file was never written.

    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    out = await feature.expand_placeholders(_placeholder(sha), _make_ctx_obj())
    assert out == f'<image sha="{sha}" ocr_status="missing"/>'


async def test_done_with_whitespace_only_text_renders_empty_status(_ctx):
    bs = BlobStore(_USER_ID)
    sha = bs.store(b"abc", "png")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha, filename="blanky.png", mime="image/png", size=3, source="paste"
    )
    SessionAttachments(_USER_ID, _SESSION_ID).mark_ocr_status(sha, "done")
    _seed_ocr_done(sha, "   ")  # whitespace-only

    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    out = await feature.expand_placeholders(_placeholder(sha), _make_ctx_obj())
    assert out == f'<image sha="{sha}" ocr_status="empty"/>'


# ----- expand_placeholders: edge cases ---------------------------------


async def test_unknown_sha_prefix_is_left_unchanged(_ctx):
    """Placeholder whose 8-char prefix doesn't match any session
    attachment is left verbatim — the model sees the raw form and can
    ask the user to disambiguate."""
    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    text = "Hello [Image sha:deadbeef]"
    out = await feature.expand_placeholders(text, _make_ctx_obj())
    assert out == text


async def test_session_attachments_failure_returns_input_unchanged(_ctx, monkeypatch):
    """A broken SessionAttachments path must not crash the turn — the
    feature returns the input unchanged and lets the LLM answer with
    whatever the user typed. (Robustness: an OS-level read error
    mid-question shouldn't take the whole answer down.)"""

    def _boom(self):
        raise OSError("disk gone")

    monkeypatch.setattr(SessionAttachments, "list", _boom)
    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    text = "Hello [Image sha:deadbeef]"
    out = await feature.expand_placeholders(text, _make_ctx_obj())
    assert out == text


async def test_zh_user_input_is_preserved(_ctx):
    """Language doesn't change inline expansion — only the placeholder
    target matters. The bilingual header / block convention is gone."""
    bs = BlobStore(_USER_ID)
    sha = bs.store(b"abc", "png")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha, filename="x.png", mime="image/png", size=3, source="paste"
    )
    SessionAttachments(_USER_ID, _SESSION_ID).mark_ocr_status(sha, "done")
    _seed_ocr_done(sha, "提取的内容")

    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    text = f"帮我看看 {_placeholder(sha)}"
    out = await feature.expand_placeholders(text, _make_ctx_obj(language="zh"))
    assert "帮我看看 " in out
    assert "提取的内容" in out
    assert f'<image sha="{sha}">' in out


async def test_ocr_with_brackets_doesnt_break_format(_ctx):
    """OCR'd content can contain ``[``/``]``/``"``; XML-style tags are
    robust because content lives between unambiguous open/close
    markers, not bracket pairs."""
    bs = BlobStore(_USER_ID)
    sha = bs.store(b"abc", "png")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha, filename="code.png", mime="image/png", size=3, source="paste"
    )
    SessionAttachments(_USER_ID, _SESSION_ID).mark_ocr_status(sha, "done")
    weird = 'def foo(x: list[int]) -> str:\n    return f"x[{len(x)}]"'
    _seed_ocr_done(sha, weird)

    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    out = await feature.expand_placeholders(_placeholder(sha), _make_ctx_obj())
    assert weird in out
    assert out.startswith(f'<image sha="{sha}">\n')
    assert out.endswith("\n</image>")
