"""Decompression-bomb guards for ``maybe_rasterize_single_image_pdf``.

Locks in the cap introduced for ce:review P1 #10 — a crafted PDF that
declares a giant MediaBox or embeds an oversized JBIG2 / JPEG2000
image must not reach ``page.get_pixmap`` (which would allocate GBs
before we ever touch a pixel). Both caps degrade to the existing OCR
path (return ``None``) rather than crash.

Embedded-image bombs are harder to forge in a unit test because
PyMuPDF's ``insert_image`` records the real pixel count of the image
it actually embeds. We exercise the MediaBox guard directly (primary
attack surface) and document the embedded-image guard via the
existing logic rather than mocking PyMuPDF internals.
"""

from __future__ import annotations

import pytest


def _new_blank_pdf(width: float, height: float):
    """Build an in-memory single-page PDF with the given MediaBox.

    Inserts a 1x1 RGB pixel so ``page.get_images()`` is non-empty,
    keeping the rasterizer past the "blank page" early return.
    """
    fitz = pytest.importorskip("fitz")
    from PIL import Image  # noqa: PLC0415

    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    # Embed a 1x1 pixel image so the embedded-raster check passes.
    img = Image.new("RGB", (1, 1), color=(0, 0, 0))
    import io  # noqa: PLC0415

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    page.insert_image(fitz.Rect(0, 0, 100, 100), stream=buf.getvalue())
    return doc


def test_oversized_mediabox_skips_rasterize(tmp_path):
    """A page MediaBox that would produce > _MAX_RASTER_PIXELS at the
    configured DPI returns None instead of allocating the buffer.

    100000 pts × 100000 pts at 200 DPI ≈ 7.7e10 pixels; sits well above
    the 25 MP cap. Without the guard, ``get_pixmap`` would attempt to
    allocate ~230 GB and crash the OCR worker.
    """
    from claritymed.core.ocr.pdf_image_peek import (
        _MAX_RASTER_PIXELS,
        maybe_rasterize_single_image_pdf,
    )

    doc = _new_blank_pdf(width=100_000, height=100_000)
    p = tmp_path / "bomb.pdf"
    doc.save(str(p))
    doc.close()

    assert maybe_rasterize_single_image_pdf(p) is None
    # Sanity: the cap is in fact the boundary we crossed.
    assert _MAX_RASTER_PIXELS < 100_000 * 100_000 * (200 / 72) ** 2


def test_normal_mediabox_still_rasterizes(tmp_path):
    """Counterpart positive case: an ordinary US-letter-sized PDF with
    one embedded image still triggers the fast path. Guards the guard —
    a regression that flipped the comparison would break the entire
    feature.
    """
    from claritymed.core.ocr.pdf_image_peek import maybe_rasterize_single_image_pdf

    # 612 x 792 pts = US letter, ~1700x2200 px at 200 DPI ≈ 3.7 MP — well
    # below the 25 MP cap.
    doc = _new_blank_pdf(width=612, height=792)
    p = tmp_path / "ok.pdf"
    doc.save(str(p))
    doc.close()

    out = maybe_rasterize_single_image_pdf(p)
    assert out is not None
    assert out.startswith(b"\x89PNG\r\n\x1a\n"), "expected PNG header"


def test_embedded_image_guard_lowered_cap_blocks_normal_image(monkeypatch, tmp_path):
    """Drop ``_MAX_EMBEDDED_IMAGE_PIXELS`` to 0 and verify even a normal
    1-pixel embedded image trips the guard. This exercises the embedded-
    image branch directly without needing to forge a malformed PDF
    stream (which PyMuPDF would refuse to open).

    The production cap stays at 100 MP — exercised by the constant, not
    by this test. The test asserts that the branch *runs* when the
    inequality holds.
    """
    from claritymed.core.ocr import pdf_image_peek as mod

    doc = _new_blank_pdf(width=612, height=792)
    p = tmp_path / "ok.pdf"
    doc.save(str(p))
    doc.close()

    monkeypatch.setattr(mod, "_MAX_EMBEDDED_IMAGE_PIXELS", 0)
    assert mod.maybe_rasterize_single_image_pdf(p) is None
