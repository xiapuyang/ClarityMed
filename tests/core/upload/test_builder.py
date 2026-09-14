"""Builder tests — exercises placeholder resolution + dedupe against
real ``SessionAttachments`` + ``BlobStore`` writes (small, fast).

Per CLAUDE.md user_id convention: tests use ``user_id="test"`` so the
conftest-managed runtime tree is the only filesystem they touch.
"""

from __future__ import annotations

import json

from claritymed.core.upload.builder import build_upload_bundle
from claritymed.stores.blob_store import BlobStore
from claritymed.stores.session_attachments import SessionAttachments

_USER = "test"
_SESS = "sess-upload-builder"


def _seed_done(sha: str, text: str) -> None:
    """Write the ocr.md + ocr.json sentinel BlobStore.ocr_done checks."""
    bs = BlobStore(_USER)
    bs.ocr_path(sha).parent.mkdir(parents=True, exist_ok=True)
    bs.ocr_path(sha).write_text(text, encoding="utf-8")
    bs.ocr_meta_path(sha).write_text(json.dumps({"kind": "ocr"}), encoding="utf-8")


def _attach(sha: str, *, filename: str, mime: str = "image/png") -> None:
    SessionAttachments(_USER, _SESS).add(
        sha256=sha,
        filename=filename,
        mime=mime,
        size=10,
        source="paste",
    )


def test_pure_text_input_becomes_one_text_part(monkeypatch):
    monkeypatch.setattr(
        "claritymed.core.upload.bundle._cfg.upload_min_part_chars", lambda: 5
    )
    bundle = build_upload_bundle(
        "These are my notes on the paper.", user_id=_USER, session_id=None
    )
    assert len(bundle.parts) == 1
    assert bundle.parts[0].kind == "text"
    assert bundle.parts[0].status == "ok"


def test_short_inline_text_silently_skipped(monkeypatch):
    """Sub-threshold inline text (typically a connector phrase between
    placeholders) is dropped rather than surfaced as a ``low_content``
    gate reason — the user didn't deliberately ask to upload it.
    """
    monkeypatch.setattr(
        "claritymed.core.upload.bundle._cfg.upload_min_part_chars", lambda: 100
    )
    bundle = build_upload_bundle("tiny note", user_id=_USER, session_id=None)
    # No parts at all — the inline-text gate fired silently.
    assert bundle.parts == ()


def test_empty_input_produces_empty_bundle():
    assert build_upload_bundle("", user_id=_USER, session_id=None).parts == ()
    assert build_upload_bundle("   \n  ", user_id=_USER, session_id=None).parts == ()


def test_resolves_image_placeholder_when_ocr_done(monkeypatch):
    monkeypatch.setattr(
        "claritymed.core.upload.bundle._cfg.upload_min_part_chars", lambda: 5
    )
    bs = BlobStore(_USER)
    sha = bs.store(b"png-bytes", "png")
    _attach(sha, filename="scan.png", mime="image/png")
    SessionAttachments(_USER, _SESS).mark_ocr_status(sha, "done")
    _seed_done(sha, "Findings: no acute abnormality.")

    top = "Top text long enough to clear the per-part floor easily."
    bottom = "Bottom text also comfortably above the per-part threshold."
    text = f"{top}\n\n[Image sha:{sha[:8]}]\n\n{bottom}"
    bundle = build_upload_bundle(text, user_id=_USER, session_id=_SESS)

    # Three parts: top text, image, bottom text (all distinct hashes).
    assert len(bundle.parts) == 3
    assert [p.kind for p in bundle.parts] == ["text", "image", "text"]
    image_part = bundle.parts[1]
    assert image_part.source == "scan.png"
    assert image_part.source_hash == sha
    assert image_part.status == "ok"
    assert "Findings" in image_part.content


def test_resolves_file_placeholder_with_ocr_pending(monkeypatch):
    monkeypatch.setattr(
        "claritymed.core.upload.bundle._cfg.upload_min_part_chars", lambda: 5
    )
    bs = BlobStore(_USER)
    sha = bs.store(b"%PDF-1.4\n%FAKE", "pdf")
    _attach(sha, filename="paper.pdf", mime="application/pdf")
    # ocr_status stays "pending" (default).

    bundle = build_upload_bundle(
        f"[File sha:{sha[:8]}]", user_id=_USER, session_id=_SESS
    )
    assert len(bundle.parts) == 1
    part = bundle.parts[0]
    assert part.kind == "file"
    assert part.status == "ocr_pending"
    assert part.content == ""


def test_resolves_file_placeholder_with_ocr_failed(monkeypatch):
    monkeypatch.setattr(
        "claritymed.core.upload.bundle._cfg.upload_min_part_chars", lambda: 5
    )
    bs = BlobStore(_USER)
    sha = bs.store(b"corrupt", "pdf")
    _attach(sha, filename="broken.pdf", mime="application/pdf")
    SessionAttachments(_USER, _SESS).mark_ocr_status(sha, "failed", reason="timeout")

    bundle = build_upload_bundle(
        f"[File sha:{sha[:8]}]", user_id=_USER, session_id=_SESS
    )
    assert bundle.parts[0].status == "ocr_failed"


def test_dedupes_same_placeholder_twice(monkeypatch):
    monkeypatch.setattr(
        "claritymed.core.upload.bundle._cfg.upload_min_part_chars", lambda: 5
    )
    bs = BlobStore(_USER)
    sha = bs.store(b"once", "png")
    _attach(sha, filename="dup.png")
    SessionAttachments(_USER, _SESS).mark_ocr_status(sha, "done")
    _seed_done(sha, "Some extracted text that is long enough.")

    text = f"[Image sha:{sha[:8]}] and again [Image sha:{sha[:8]}]"
    bundle = build_upload_bundle(text, user_id=_USER, session_id=_SESS)
    # Two text fragments ("" stripped) collapse into the spans between
    # placeholders; the image itself contributes exactly one part.
    image_parts = [p for p in bundle.parts if p.kind == "image"]
    assert len(image_parts) == 1


def test_dedupes_identical_inline_text_segments(monkeypatch):
    monkeypatch.setattr(
        "claritymed.core.upload.bundle._cfg.upload_min_part_chars", lambda: 1
    )
    # Same paragraph repeated with whitespace differences — normalisation
    # in hash_inline_text should fold them. Need a placeholder shape
    # between them or builder treats it as one contiguous segment;
    # use a bogus placeholder that won't resolve.
    text_with_sep = "Hello world.\n[Image sha:deadbeef]\nHello   world."
    bundle = build_upload_bundle(text_with_sep, user_id=_USER, session_id=None)
    text_parts = [p for p in bundle.parts if p.kind == "text"]
    # First "Hello world." and the literal-placeholder fragment (since
    # session_id=None means no resolution) are distinct;
    # second "Hello world." dedupes against the first.
    contents = [p.content.strip() for p in text_parts]
    assert contents.count("Hello world.") == 1


def test_unresolved_placeholder_falls_back_to_literal_text(monkeypatch):
    # session_id provided but no matching attachment row → placeholder
    # is left as inline text. The 19-char literal placeholder is below
    # the default per-part floor (30), so we lower the floor here to
    # confirm the fallback shape; in production the unresolved
    # placeholder is silently dropped and the modal surfaces a generic
    # "empty" gate reason instead.
    monkeypatch.setattr(
        "claritymed.core.upload.bundle._cfg.upload_min_part_chars", lambda: 5
    )
    bundle = build_upload_bundle("[File sha:deadbeef]", user_id=_USER, session_id=_SESS)
    assert len(bundle.parts) == 1
    assert bundle.parts[0].kind == "text"
    assert "deadbeef" in bundle.parts[0].content


def test_unresolved_placeholder_with_default_floor_is_empty():
    """Short literal-placeholder fallback (19 chars) is below the default
    per-part floor (30), so the inline-text gate drops it silently and
    the bundle is empty — the modal will render the ``empty`` reason.
    """
    bundle = build_upload_bundle("[File sha:deadbeef]", user_id=_USER, session_id=_SESS)
    assert bundle.parts == ()


def test_low_content_image_marked(monkeypatch):
    monkeypatch.setattr(
        "claritymed.core.upload.bundle._cfg.upload_min_part_chars", lambda: 1000
    )
    bs = BlobStore(_USER)
    sha = bs.store(b"x", "png")
    _attach(sha, filename="tiny.png")
    SessionAttachments(_USER, _SESS).mark_ocr_status(sha, "done")
    _seed_done(sha, "tiny")

    bundle = build_upload_bundle(
        f"[Image sha:{sha[:8]}]", user_id=_USER, session_id=_SESS
    )
    assert bundle.parts[0].status == "low_content"
