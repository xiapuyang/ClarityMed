"""Unit tests for the vision image-guard.

Synthetic PIL images at boundary sizes are cheaper than fixturing real
medical scans, and they let each axis (bytes / dimension / pixels) be
exercised in isolation — the production datasets all sit comfortably
inside the limits, so they'd never hit the rejection paths.

PNG aggressively compresses solid colors (a 128×128 flat PNG lands
around 300 bytes), so test images are filled with ``os.urandom`` to
make the byte-count match the pixel-count and keep the bytes-axis
independent of the dimension axis.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from claritymed.core.vision.image_guard import ImageLimits, validate_image
from claritymed.errors import ImageTooLargeError, ImageTooSmallError


# bytes ceiling stays loose (2 MB) so dimension / pixel rejects can be
# isolated — a 600×128 noise PNG is ~230 kB, a 500×500 noise PNG is
# ~750 kB; both must clear bytes to reach the axis under test.
_LIMITS = ImageLimits(
    max_bytes=2_000_000,
    max_dimension=512,
    max_pixels=200_000,
    min_bytes=500,
    min_dimension=64,
    min_pixels=8_000,
)


def _write_image(path: Path, width: int, height: int, *, fmt: str = "PNG") -> Path:
    """Write a random-noise raster at ``width × height`` to ``path``."""
    pixels = os.urandom(width * height * 3)
    Image.frombytes("RGB", (width, height), pixels).save(path, fmt)
    return path


# --- happy path -----------------------------------------------------------


def test_validate_accepts_in_range_image(tmp_path):
    """A 128×128 noise PNG sits in the middle of every bound."""
    path = _write_image(tmp_path / "ok.png", 128, 128)
    validate_image(path, _LIMITS)  # must not raise


# --- max bounds -----------------------------------------------------------


def test_validate_rejects_oversize_bytes(tmp_path):
    """A 200kB+1 blob trips bytes before dimensions are even read."""
    path = tmp_path / "big.bin"
    path.write_bytes(b"\x00" * (_LIMITS.max_bytes + 1))
    with pytest.raises(ImageTooLargeError, match="max_bytes"):
        validate_image(path, _LIMITS)


def test_validate_rejects_oversize_dimension(tmp_path):
    """A 600×128 image trips max_dimension on width."""
    path = _write_image(tmp_path / "wide.png", 600, 128)
    with pytest.raises(ImageTooLargeError, match="max_dimension"):
        validate_image(path, _LIMITS)


def test_validate_rejects_oversize_pixels(tmp_path):
    """A 500×500 image (250k px) clears dimension but trips pixels."""
    path = _write_image(tmp_path / "square.png", 500, 500)
    with pytest.raises(ImageTooLargeError, match="max_pixels"):
        validate_image(path, _LIMITS)


# --- min bounds -----------------------------------------------------------


def test_validate_rejects_undersize_bytes(tmp_path):
    """A 100-byte file fails the bytes floor before PIL even opens it."""
    path = tmp_path / "tiny.bin"
    path.write_bytes(b"\x00" * 100)
    with pytest.raises(ImageTooSmallError, match="min_bytes"):
        validate_image(path, _LIMITS)


def test_validate_rejects_undersize_dimension(tmp_path):
    """A 32×128 noise PNG: bytes clear (~12kB) so dimension is the trip."""
    path = _write_image(tmp_path / "thin.png", 32, 128)
    with pytest.raises(ImageTooSmallError, match="min_dimension"):
        validate_image(path, _LIMITS)


def test_validate_rejects_undersize_pixels(tmp_path):
    """A 70×70 noise PNG clears dimension floor (≥64) but fails pixels (4900 < 8000)."""
    path = _write_image(tmp_path / "small.png", 70, 70)
    with pytest.raises(ImageTooSmallError, match="min_pixels"):
        validate_image(path, _LIMITS)


# --- PDF passthrough -----------------------------------------------------


def test_validate_pdf_byte_check_only(tmp_path):
    """PDFs check bytes but skip dimension/pixel — no PIL parse attempt."""
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4\n" + b"x" * _LIMITS.min_bytes)
    validate_image(path, _LIMITS)  # must not raise


def test_validate_pdf_oversize_bytes_rejected(tmp_path):
    """Bytes ceiling applies to PDFs too."""
    path = tmp_path / "bigdoc.pdf"
    path.write_bytes(b"%PDF-1.4\n" + b"x" * _LIMITS.max_bytes)
    with pytest.raises(ImageTooLargeError, match="max_bytes"):
        validate_image(path, _LIMITS)


# --- error message shape -------------------------------------------------


def test_error_names_filename_and_observed_value(tmp_path):
    """Error message includes filename + observed value + violated limit."""
    path = _write_image(tmp_path / "wide.png", 600, 128)
    with pytest.raises(ImageTooLargeError) as exc:
        validate_image(path, _LIMITS)
    msg = str(exc.value)
    assert "wide.png" in msg
    assert "600x128" in msg
    assert "max_dimension=512" in msg
