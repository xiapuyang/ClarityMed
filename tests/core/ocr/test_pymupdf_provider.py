"""Tests for ``PyMuPDFOcrProvider``."""

from __future__ import annotations

from pathlib import Path

import pytest

from claritymed.core.ocr.base import OcrError
from claritymed.core.ocr.pymupdf_provider import PyMuPDFOcrProvider


def _digital_pdf(path: Path, body: str = "Hello digital PDF") -> Path:
    """Write a minimal text-PDF for tests via ``fitz``."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), body)
    doc.save(path)
    doc.close()
    return path


def _scanned_lookalike(path: Path) -> Path:
    """Write a one-page PDF with no text — pymupdf will return empty."""
    import fitz

    doc = fitz.open()
    doc.new_page()  # blank page, no text inserted
    doc.save(path)
    doc.close()
    return path


def test_is_local_true():
    assert PyMuPDFOcrProvider.is_local is True


async def test_extract_digital_pdf_returns_text(tmp_path: Path):
    pdf = _digital_pdf(tmp_path / "digital.pdf")
    provider = PyMuPDFOcrProvider()
    result = await provider.extract_text(pdf)
    assert "Hello digital PDF" in result.text
    assert result.provider_used == "pymupdf"
    assert result.chain_tried == ["pymupdf"]


async def test_extract_scanned_lookalike_raises(tmp_path: Path):
    """Empty extract → OcrError so the chain falls through to next provider."""
    pdf = _scanned_lookalike(tmp_path / "scan.pdf")
    provider = PyMuPDFOcrProvider()
    with pytest.raises(OcrError, match="no text"):
        await provider.extract_text(pdf)


async def test_extract_missing_file_raises(tmp_path: Path):
    provider = PyMuPDFOcrProvider()
    with pytest.raises(OcrError):
        await provider.extract_text(tmp_path / "nope.pdf")
