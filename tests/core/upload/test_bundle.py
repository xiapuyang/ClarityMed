"""Pure-data tests for UploadPart / UploadBundle / validate.

No filesystem, no monkeypatching of stores — these tests exercise the
shape of the validator only. Threshold accessors are stubbed via
monkeypatch where the test wants a specific floor.
"""

from __future__ import annotations

import pytest

from claritymed.core.upload.bundle import (
    UploadBundle,
    UploadPart,
    hash_inline_text,
    normalize_text,
)


def _part(
    *,
    kind="text",
    source="inline",
    content="hello world",
    status="ok",
    chars: int | None = None,
    source_hash: str | None = None,
) -> UploadPart:
    """Build a part with sensible defaults for tests that only care about one field."""
    return UploadPart(
        kind=kind,
        source=source,
        content=content,
        source_hash=source_hash or hash_inline_text(content),
        status=status,
        chars=chars if chars is not None else len(content.replace(" ", "")),
    )


class TestNormalizeText:
    def test_collapses_whitespace_runs(self):
        assert normalize_text("hello   world\n\nfoo") == "hello world foo"

    def test_strips_leading_trailing(self):
        assert normalize_text("  hello  ").strip() == "hello"

    def test_casefolds(self):
        assert normalize_text("Hello") == normalize_text("HELLO")

    def test_nfkc_full_width_punctuation(self):
        # Chinese full-width comma should fold the same as ASCII comma
        # under NFKC, so "a, b" pasted twice with different keyboards
        # dedupes.
        assert normalize_text("a，b") == normalize_text("a,b")


class TestHashInlineText:
    def test_same_normalized_text_same_hash(self):
        assert hash_inline_text("Hello\nworld") == hash_inline_text("hello   world")

    def test_different_content_different_hash(self):
        assert hash_inline_text("alpha") != hash_inline_text("beta")


class TestPartFromText:
    def test_ok_when_above_floor(self, monkeypatch):
        monkeypatch.setattr(
            "claritymed.core.upload.bundle._cfg.upload_min_part_chars",
            lambda: 5,
        )
        part = UploadPart.from_text("hello world", source="msg-1")
        assert part.status == "ok"
        assert part.kind == "text"
        assert part.source == "msg-1"

    def test_low_content_below_floor(self, monkeypatch):
        monkeypatch.setattr(
            "claritymed.core.upload.bundle._cfg.upload_min_part_chars",
            lambda: 100,
        )
        part = UploadPart.from_text("hi", source="msg-1")
        assert part.status == "low_content"


class TestBundleValidate:
    def test_empty_bundle_fails(self):
        result = UploadBundle(parts=()).validate()
        assert result.ok is False
        assert result.reasons == ["empty"]

    def test_single_ok_part_passes(self, monkeypatch):
        monkeypatch.setattr(
            "claritymed.core.upload.bundle._cfg.upload_min_total_chars",
            lambda: 5,
        )
        bundle = UploadBundle(parts=(_part(content="hello world", chars=10),))
        assert bundle.validate().ok is True

    def test_total_floor_blocks(self, monkeypatch):
        monkeypatch.setattr(
            "claritymed.core.upload.bundle._cfg.upload_min_total_chars",
            lambda: 100,
        )
        bundle = UploadBundle(parts=(_part(content="hi", chars=2),))
        result = bundle.validate()
        assert result.ok is False
        assert any("total_too_short" in r for r in result.reasons)

    def test_ocr_failed_blocks(self, monkeypatch):
        monkeypatch.setattr(
            "claritymed.core.upload.bundle._cfg.upload_min_total_chars",
            lambda: 0,
        )
        bundle = UploadBundle(
            parts=(_part(status="ocr_failed", source="ct.png", chars=50),),
        )
        result = bundle.validate()
        assert result.ok is False
        assert any(r.startswith("ocr_failed:ct.png") for r in result.reasons)

    def test_ocr_pending_blocks(self, monkeypatch):
        monkeypatch.setattr(
            "claritymed.core.upload.bundle._cfg.upload_min_total_chars",
            lambda: 0,
        )
        bundle = UploadBundle(
            parts=(_part(status="ocr_pending", source="scan.pdf", chars=0),),
        )
        assert bundle.validate().ok is False

    def test_low_content_blocks(self, monkeypatch):
        monkeypatch.setattr(
            "claritymed.core.upload.bundle._cfg.upload_min_total_chars",
            lambda: 0,
        )
        bundle = UploadBundle(parts=(_part(status="low_content", chars=3),))
        assert bundle.validate().ok is False

    def test_multiple_reasons_surface_all(self, monkeypatch):
        monkeypatch.setattr(
            "claritymed.core.upload.bundle._cfg.upload_min_total_chars",
            lambda: 1000,
        )
        bundle = UploadBundle(
            parts=(
                _part(status="ocr_failed", source="bad.png", chars=0),
                _part(status="low_content", source="tiny.txt", chars=2),
            )
        )
        result = bundle.validate()
        assert result.ok is False
        # Both per-part reasons plus the total-floor reason should fire.
        joined = " ".join(result.reasons)
        assert "ocr_failed:bad.png" in joined
        assert "low_content:tiny.txt" in joined
        assert "total_too_short" in joined


class TestBundleProperties:
    def test_assembled_text_joins_with_blank_line(self):
        bundle = UploadBundle(
            parts=(
                _part(content="alpha", chars=5),
                _part(content="beta", chars=4),
            )
        )
        assert bundle.assembled_text == "alpha\n\nbeta"

    def test_assembled_text_drops_whitespace_only(self):
        bundle = UploadBundle(
            parts=(
                _part(content="alpha", chars=5),
                _part(content="   ", chars=0),
                _part(content="gamma", chars=5),
            )
        )
        # The empty-content part contributes nothing to the joined output.
        assert bundle.assembled_text == "alpha\n\ngamma"

    def test_total_chars_sums_parts(self):
        bundle = UploadBundle(parts=(_part(chars=10), _part(chars=15), _part(chars=2)))
        assert bundle.total_chars == 27

    @pytest.mark.parametrize(
        "status,attr",
        [
            ("ocr_failed", "has_failed_parts"),
            ("ocr_pending", "has_pending_parts"),
            ("low_content", "has_low_content_parts"),
        ],
    )
    def test_status_flags(self, status, attr):
        bundle = UploadBundle(parts=(_part(status=status, chars=1),))
        assert getattr(bundle, attr) is True
