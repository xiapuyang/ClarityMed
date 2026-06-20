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


# ----- expand_placeholders: vision tags (Unit 3) -----------------------


def _seed_ocr_done_with_tags(sha: str, text: str, **vision_fields) -> None:
    """Write the sentinel with the worker's vision-tag fields populated."""
    bs = BlobStore(_USER_ID)
    bs.ocr_path(sha).parent.mkdir(parents=True, exist_ok=True)
    bs.ocr_path(sha).write_text(text, encoding="utf-8")
    payload = {"status": "done", **vision_fields}
    bs.ocr_meta_path(sha).write_text(json.dumps(payload), encoding="utf-8")


async def test_image_tag_includes_modality_is_medical_and_report_attrs(_ctx):
    bs = BlobStore(_USER_ID)
    sha = bs.store(b"abc", "png")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha, filename="us.png", mime="image/png", size=3, source="paste"
    )
    SessionAttachments(_USER_ID, _SESSION_ID).mark_ocr_status(sha, "done")
    _seed_ocr_done_with_tags(
        sha,
        "extracted body",
        modality="ultrasound",
        is_medical=True,
        ocr_has_report=False,
    )

    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    out = await feature.expand_placeholders(_placeholder(sha), _make_ctx_obj())
    # All three vision attributes appear on the opening tag, alongside
    # the existing sha attribute. The tool description (Unit 7) branches
    # off these values, so the exact attribute names + lowercased
    # "true"/"false" XML booleans are load-bearing.
    assert (
        f'<image sha="{sha}" modality="ultrasound" is_medical="true" ocr_has_report="false">'
        in out
    )
    assert "extracted body" in out


async def test_image_tag_omits_attrs_when_sentinel_lacks_them(_ctx):
    """Legacy blob (sentinel pre-dating Unit 3) → no vision attrs.

    Field absence is the LLM-side signal "no opinion on modality"; we
    must not invent defaults like ``modality="unknown"``, which would
    pretend the classifier ran when it hadn't.
    """
    bs = BlobStore(_USER_ID)
    sha = bs.store(b"abc", "png")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha, filename="legacy.png", mime="image/png", size=3, source="paste"
    )
    SessionAttachments(_USER_ID, _SESSION_ID).mark_ocr_status(sha, "done")
    _seed_ocr_done(sha, "extracted body")  # legacy sentinel — no vision fields

    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    out = await feature.expand_placeholders(_placeholder(sha), _make_ctx_obj())
    assert f'<image sha="{sha}">' in out
    assert "modality=" not in out
    assert "is_medical=" not in out
    assert "ocr_has_report=" not in out


async def test_image_tag_carries_modality_unknown_when_classifier_failed(_ctx):
    """Worker tags ``modality=unknown`` when medical-clip was unreachable.

    The LLM-side routing rule treats ``unknown`` the same as a missing
    attribute (asks the user via askuserquestion). What matters for
    this test is that the attribute round-trips verbatim — drift would
    break Unit 7's tool-description contract.
    """
    bs = BlobStore(_USER_ID)
    sha = bs.store(b"abc", "png")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha, filename="us.png", mime="image/png", size=3, source="paste"
    )
    SessionAttachments(_USER_ID, _SESSION_ID).mark_ocr_status(sha, "done")
    _seed_ocr_done_with_tags(
        sha,
        "extracted body",
        modality="unknown",
        is_medical=None,  # None → omitted in renderer (KTD-V8 graceful)
        ocr_has_report=False,
    )

    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    out = await feature.expand_placeholders(_placeholder(sha), _make_ctx_obj())
    assert f'<image sha="{sha}" modality="unknown" ocr_has_report="false">' in out
    # ``is_medical`` is null when the classifier failed; renderer omits
    # the attribute so the LLM doesn't read a default it didn't earn.
    assert "is_medical=" not in out


async def test_file_tag_does_not_carry_vision_attrs(_ctx):
    """Vision tags are image-only — files (PDFs) never carry them.

    A PDF sentinel won't normally have these fields (the worker skips
    classification on non-images), but we belt-and-suspender it here so
    a future worker change that *does* tag them can't leak into the
    `<file>` rendering.
    """
    bs = BlobStore(_USER_ID)
    sha = bs.store(b"pdf-bytes", "pdf")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha, filename="r.pdf", mime="application/pdf", size=9, source="paste"
    )
    SessionAttachments(_USER_ID, _SESSION_ID).mark_ocr_status(sha, "done")
    _seed_ocr_done_with_tags(
        sha,
        "report text",
        modality="document",  # would be wrong, but renderer must ignore for files
        is_medical=False,
        ocr_has_report=True,
    )

    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    out = await feature.expand_placeholders(
        _placeholder(sha, kind="File"), _make_ctx_obj()
    )
    assert out.startswith(f'<file sha="{sha}">')
    assert "modality=" not in out
    assert "is_medical=" not in out
    assert "ocr_has_report=" not in out


async def test_pending_image_with_vision_attrs_renders_self_closing_with_attrs(_ctx):
    """Pre-OCR status branches still get the vision attrs — sentinel may
    arrive before OCR text is ready when the classifier is fast and the
    OCR provider is slow. The rendered tag still carries the modality
    info the LLM needs to decide whether to ask askuserquestion."""
    bs = BlobStore(_USER_ID)
    sha = bs.store(b"abc", "png")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha, filename="x.png", mime="image/png", size=3, source="paste"
    )
    # Sentinel exists but session row stays "pending" — exercises the
    # status-branch path while still reading ocr.json.
    _seed_ocr_done_with_tags(
        sha,
        "",  # OCR text empty / not ready
        modality="ct",
        is_medical=True,
        ocr_has_report=False,
    )

    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    out = await feature.expand_placeholders(_placeholder(sha), _make_ctx_obj())
    assert (
        f'<image sha="{sha}" modality="ct" is_medical="true" ocr_has_report="false" ocr_status="pending"/>'
        in out
    )


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


async def test_file_placeholder_with_vision_png_sidecar_upgrades_to_image(_ctx):
    """A ``[File sha:…]`` placeholder for a PDF that the OCR worker
    rasterized into ``vision.png`` must render as ``<image …>`` with
    the modality/is_medical attrs the LLM-side routing keys off.

    Regression: TUI's paste-time MIME check freezes ``kind=File`` for
    PDFs before the worker peeks inside; without the renderer-level
    sidecar override, a 1-page CT-wrapped-in-PDF ends up as a bare
    ``<file ocr_status="empty"/>`` and the vision tool never fires.
    """
    bs = BlobStore(_USER_ID)
    pdf_bytes = b"%PDF-fake-bytes-for-test"
    sha = bs.store(pdf_bytes, "pdf")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha,
        filename="000108 (3).pdf",
        mime="application/pdf",
        size=len(pdf_bytes),
        source="paste",
    )
    SessionAttachments(_USER_ID, _SESSION_ID).mark_ocr_status(sha, "empty")
    # vision.png sidecar = worker's 1-page image-PDF fast-path fired.
    (bs.dir(sha) / "vision.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    _seed_ocr_done_with_tags(
        sha,
        "",  # PDF page had no text by definition
        status="empty",
        provider="pdf_image_peek",
        modality="ct",
        is_medical=True,
        modality_confidence=0.9995,
    )
    # Sentinel override — ``_seed_ocr_done_with_tags`` defaults status
    # to "done"; this PDF case must use "empty" to match production.
    payload = json.loads(bs.ocr_meta_path(sha).read_text(encoding="utf-8"))
    payload["status"] = "empty"
    bs.ocr_meta_path(sha).write_text(json.dumps(payload), encoding="utf-8")

    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    out = await feature.expand_placeholders(
        _placeholder(sha, kind="File"), _make_ctx_obj()
    )
    # The placeholder said "File" but the sidecar upgraded the kind.
    assert out.startswith("<image "), f"expected <image>, got: {out!r}"
    assert 'modality="ct"' in out
    assert 'is_medical="true"' in out
    assert 'ocr_status="empty"' in out


async def test_file_placeholder_without_sidecar_stays_file(_ctx):
    """A genuine ``<file>`` (multi-page report PDF) keeps rendering as one.

    Guards against the override branch firing for any PDF with a
    sentinel — it must be sidecar-gated, not status-gated.
    """
    bs = BlobStore(_USER_ID)
    sha = bs.store(b"%PDF-multipage-report", "pdf")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha,
        filename="report.pdf",
        mime="application/pdf",
        size=10,
        source="paste",
    )
    SessionAttachments(_USER_ID, _SESSION_ID).mark_ocr_status(sha, "done")
    _seed_ocr_done_with_tags(
        sha, "FINDINGS: ...", modality="document", is_medical=False
    )

    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)
    out = await feature.expand_placeholders(
        _placeholder(sha, kind="File"), _make_ctx_obj()
    )
    assert out.startswith(f'<file sha="{sha}">'), (
        f"no sidecar → must stay <file>, got: {out!r}"
    )
    assert "modality=" not in out
    assert "is_medical=" not in out


async def test_image_tag_includes_vision_disabled_when_feature_offline(_ctx):
    """When the vision feature is disabled (bootstrap drift), every
    medical ``<image>`` tag carries ``vision_disabled="…"`` so the LLM
    stops trying to call the missing tool and explains text-only."""
    bs = BlobStore(_USER_ID)
    sha = bs.store(b"abc", "png")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha, filename="ct.png", mime="image/png", size=3, source="paste"
    )
    SessionAttachments(_USER_ID, _SESSION_ID).mark_ocr_status(sha, "empty")
    _seed_ocr_done_with_tags(
        sha,
        "",
        status="empty",
        modality="ct",
        is_medical=True,
    )
    # Force the sentinel to status=empty (the helper hard-codes done).
    payload = json.loads(bs.ocr_meta_path(sha).read_text(encoding="utf-8"))
    payload["status"] = "empty"
    bs.ocr_meta_path(sha).write_text(json.dumps(payload), encoding="utf-8")

    reason = "VisionCatalogMismatchError: manifest sha drift for breast_busi_unet_v1"
    feature = AttachmentsFeature(
        get_session_id=lambda: _SESSION_ID,
        get_vision_disabled_reason=lambda: reason,
    )
    out = await feature.expand_placeholders(
        _placeholder(sha, kind="Image"), _make_ctx_obj()
    )
    assert f'vision_disabled="{reason}"' in out, (
        f"expected vision_disabled attribute carrying the reason; got: {out!r}"
    )
    assert 'is_medical="true"' in out
    assert 'modality="ct"' in out


async def test_image_tag_omits_vision_disabled_when_feature_healthy(_ctx):
    """Healthy vision (``get_vision_disabled_reason`` returns None) →
    no ``vision_disabled`` attribute is added. Pure guard against a
    refactor that accidentally always emits the attr."""
    bs = BlobStore(_USER_ID)
    sha = bs.store(b"abc", "png")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha, filename="ct.png", mime="image/png", size=3, source="paste"
    )
    SessionAttachments(_USER_ID, _SESSION_ID).mark_ocr_status(sha, "empty")
    _seed_ocr_done_with_tags(sha, "", status="empty", modality="ct", is_medical=True)
    payload = json.loads(bs.ocr_meta_path(sha).read_text(encoding="utf-8"))
    payload["status"] = "empty"
    bs.ocr_meta_path(sha).write_text(json.dumps(payload), encoding="utf-8")

    feature = AttachmentsFeature(
        get_session_id=lambda: _SESSION_ID,
        get_vision_disabled_reason=lambda: None,
    )
    out = await feature.expand_placeholders(
        _placeholder(sha, kind="Image"), _make_ctx_obj()
    )
    assert "vision_disabled" not in out, (
        f"healthy vision must not emit the attribute; got: {out!r}"
    )


async def test_image_tag_omits_vision_disabled_for_non_medical(_ctx):
    """Disabled vision should NOT annotate ``is_medical="false"`` images.

    Rule 1 of the tool-description (non-medical → don't call the tool)
    already short-circuits, so adding ``vision_disabled`` here would be
    noise that confuses the LLM about why it should explain anything.
    Only medical images carry the attr."""
    bs = BlobStore(_USER_ID)
    sha = bs.store(b"abc", "png")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha, filename="cat.png", mime="image/png", size=3, source="paste"
    )
    SessionAttachments(_USER_ID, _SESSION_ID).mark_ocr_status(sha, "done")
    _seed_ocr_done_with_tags(sha, "ocr text", modality="photo", is_medical=False)

    feature = AttachmentsFeature(
        get_session_id=lambda: _SESSION_ID,
        get_vision_disabled_reason=lambda: "vision broken",
    )
    out = await feature.expand_placeholders(
        _placeholder(sha, kind="Image"), _make_ctx_obj()
    )
    assert "vision_disabled" not in out
