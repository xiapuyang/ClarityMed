"""Tests for the Magika-backed filetype detector."""

from __future__ import annotations

from pathlib import Path

import pytest

from claritymed.core.filetype.detector import (
    DetectResult,
    _get_engine,
    detect,
    detect_path,
)


def test_detect_returns_result_with_canonical_extension():
    """A real PNG produced by Pillow is detected as a PNG.

    Hand-rolled 16-byte PNG headers don't carry enough signal for
    Magika to commit (its threshold rejects ambiguous inputs); a real
    encoded PNG gives the high-confidence path the gate cares about.
    """
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (16, 16), color="white").save(buf, format="PNG")

    result = detect(buf.getvalue())
    assert result is not None
    assert result.ext == ".png"
    assert result.label == "png"
    assert 0.0 <= result.score <= 1.0
    assert result.mime_type.startswith("image/")


def test_detect_returns_none_when_engine_unavailable(monkeypatch):
    """If Magika fails to import / initialize the detector returns None
    so callers can fall back to the original extension cleanly."""
    monkeypatch.setattr("claritymed.core.filetype.detector._get_engine", lambda: None)
    assert detect(b"anything") is None


def test_detect_swallows_inference_errors(monkeypatch):
    """A runtime failure inside Magika doesn't crash the caller —
    the detector logs and returns None instead."""

    class _Boom:
        def identify_bytes(self, _data):
            raise RuntimeError("onnxruntime crashed")

    monkeypatch.setattr(
        "claritymed.core.filetype.detector._get_engine", lambda: _Boom()
    )
    assert detect(b"anything") is None


def test_detect_path_reads_and_dispatches(tmp_path: Path):
    """``detect_path`` reads bytes and forwards to ``detect``.

    Uses a real reportlab-generated PDF so Magika has enough structure
    (xref table + objects + EOF) to commit confidently.
    """
    pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas

    sample = tmp_path / "doc.pdf"
    c = canvas.Canvas(str(sample))
    c.drawString(100, 750, "hello")
    c.save()

    result = detect_path(sample)
    assert result is not None
    assert result.ext == ".pdf"


def test_detect_path_returns_none_on_read_failure(tmp_path: Path):
    """A missing file is a soft failure — None, not an exception."""
    result = detect_path(tmp_path / "does-not-exist.bin")
    assert result is None


def test_detect_result_is_frozen():
    """``DetectResult`` is immutable so callers can rely on it as a cache key."""
    r = DetectResult(ext=".png", label="png", score=0.9, mime_type="image/png")
    with pytest.raises(Exception):
        r.ext = ".jpg"  # type: ignore[misc]


def test_get_engine_caches_instance(monkeypatch):
    """Repeated ``_get_engine`` calls return the same engine instance —
    Magika is expensive to construct and the cache is process-wide."""
    # Reset the cached engine so the test exercises the construction path.
    import claritymed.core.filetype.detector as det

    monkeypatch.setattr(det, "_engine", None)
    first = _get_engine()
    second = _get_engine()
    # Either both succeeded (and are the same) or both failed (None).
    assert first is second
