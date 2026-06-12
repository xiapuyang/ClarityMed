"""End-to-end OCR coverage: every declared ``supported_extensions`` is
verified by running the actual provider against a real fixture.

Goal: detect drift between what providers *claim* and what they
*actually* extract. Each test:

1. Generates a minimal but real-format fixture containing a planted
   marker string (``MARKER_TEXT``).
2. Calls the live provider's ``extract_text``.
3. Asserts the marker appears in the output, OR — for OCR backends
   whose recognition fidelity is approximate — asserts the output is
   non-empty.

Gating:

* Local providers (``pymupdf``, ``rapidocr``, ``pandoc``) — run when
  their optional extra is installed; skip otherwise.
* ``marker`` is excluded from the matrix — it lazy-loads ~2 GB of
  models and isn't suitable for the unit-test cadence.
* ``llm`` is excluded — requires a running vision model server and
  is exercised by the AskService e2e suite under separate flags.
* ``mineru`` runs only when both ``CLARITYMED_ALLOW_MINERU=1`` and
  ``MINERU_API_TOKEN`` are present; otherwise the constructor refuses.

These tests sit under ``tests/e2e/`` so the default ``pytest`` command
(``--ignore=tests/e2e``) skips them. Run explicitly:

    uv run pytest tests/e2e/ocr -v --no-cov
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.e2e.ocr.fixtures import (
    MARKER_TEXT,
    make_bmp,
    make_docx,
    make_epub,
    make_gif,
    make_html,
    make_jpeg,
    make_odt,
    make_pdf,
    make_png,
    make_pptx,
    make_rtf,
    make_tiff,
    make_webp,
    make_xlsx,  # noqa: F401  — used in MineRU matrix
)


# --- pymupdf ---------------------------------------------------------


@pytest.mark.asyncio
async def test_pymupdf_extracts_pdf(tmp_path: Path):
    """PyMuPDF reads digital PDFs verbatim — exact string match."""
    pytest.importorskip("fitz")
    from claritymed.core.ocr.pymupdf_provider import PyMuPDFOcrProvider

    pdf = make_pdf(tmp_path)
    result = await PyMuPDFOcrProvider().extract_text(pdf)
    assert MARKER_TEXT in result.text


# --- pandoc ----------------------------------------------------------


@pytest.fixture
def pandoc_available() -> None:
    pytest.importorskip("pypandoc")
    import pypandoc

    try:
        pypandoc.get_pandoc_version()
    except OSError:
        pytest.skip("pandoc binary not on PATH (install ocr-office extra)")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "make_fn,ext",
    [
        (make_docx, ".docx"),
        (make_pptx, ".pptx"),
        (make_html, ".html"),
        (make_rtf, ".rtf"),
        (make_odt, ".odt"),
        (make_epub, ".epub"),
    ],
)
async def test_pandoc_extracts_format(
    tmp_path: Path, pandoc_available, make_fn, ext: str
):
    """Pandoc round-trips every modern office / markup format it claims.

    Asserts substring presence: pandoc may wrap the marker in markdown
    decorators (``# marker`` vs ``marker``) depending on the input
    structure, but the literal string is preserved.
    """
    from claritymed.core.ocr.pandoc_provider import PandocOcrProvider

    fixture = make_fn(tmp_path)
    assert fixture.suffix == ext
    result = await PandocOcrProvider().extract_text(fixture)
    assert MARKER_TEXT in result.text, (
        f"pandoc extracted from {ext} but marker missing — got: {result.text[:200]!r}"
    )


@pytest.mark.parametrize("removed_ext", [".doc", ".ppt", ".xls", ".xlsx"])
def test_pandoc_does_not_claim_unusable_formats(removed_ext: str):
    """Regression guard: pandoc's ``supported_extensions`` must NOT
    re-acquire formats the e2e matrix proved don't work.

    * Legacy OLE2 (``.doc`` / ``.ppt`` / ``.xls``): need
      antiword/catdoc/libreoffice helpers that ``pypandoc-binary``
      doesn't ship.
    * ``.xlsx``: pandoc 3.9.x has a workbook-relationship path bug
      ("Entry not found: xl//xl/worksheets/sheet1.xml").

    If a future pandoc fixes these and we want to re-claim them, this
    test will tell us by failing — at which point add a real
    ``test_pandoc_extracts_format`` case and remove the entry from
    this list.
    """
    from claritymed.core.ocr.pandoc_provider import PandocOcrProvider

    assert removed_ext not in PandocOcrProvider.supported_extensions


# --- rapidocr --------------------------------------------------------


@pytest.fixture
def rapidocr_available() -> None:
    pytest.importorskip("rapidocr_onnxruntime")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "make_fn,ext",
    [
        (make_png, ".png"),
        (make_jpeg, ".jpg"),
        (make_webp, ".webp"),
        (make_bmp, ".bmp"),
        (make_tiff, ".tiff"),
    ],
)
async def test_rapidocr_extracts_raster_image(
    tmp_path: Path, rapidocr_available, make_fn, ext: str
):
    """RapidOCR extracts SOME text from every image format it claims.

    The default Pillow font is small and may produce imperfect OCR —
    we assert non-empty extraction rather than substring match. The
    Chinese-medical quality assessment lives in
    ``test_rapidocr_chinese_quality.py`` separately.
    """
    from claritymed.core.ocr.rapidocr_provider import RapidOcrProvider

    fixture = make_fn(tmp_path)
    assert fixture.suffix == ext
    result = await RapidOcrProvider().extract_text(fixture)
    assert result.text.strip(), (
        f"rapidocr returned empty extract for {ext} — claim is incorrect?"
    )


def test_rapidocr_does_not_claim_gif():
    """Regression guard: rapidocr's ``supported_extensions`` should NOT
    include ``.gif`` — RapidOCR doesn't unpack multi-frame GIFs and
    single-frame GIFs rarely carry recognisable text in this pipeline.

    If a future RapidOCR version handles GIFs cleanly, add it to the
    set and remove this test.
    """
    from claritymed.core.ocr.rapidocr_provider import RapidOcrProvider

    assert ".gif" not in RapidOcrProvider.supported_extensions


# --- mineru ----------------------------------------------------------


@pytest.fixture
def mineru_available(monkeypatch) -> None:
    if os.environ.get("CLARITYMED_ALLOW_MINERU") != "1":
        pytest.skip("CLARITYMED_ALLOW_MINERU=1 not set")
    if not os.environ.get("MINERU_API_TOKEN"):
        pytest.skip("MINERU_API_TOKEN not set")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "make_fn,ext",
    [
        (make_pdf, ".pdf"),
        (make_docx, ".docx"),
        (make_xlsx, ".xlsx"),
        (make_pptx, ".pptx"),
        (make_png, ".png"),
        (make_jpeg, ".jpg"),
        (make_webp, ".webp"),
        (make_bmp, ".bmp"),
        (make_gif, ".gif"),
    ],
)
async def test_mineru_accepts_claimed_format(
    tmp_path: Path, mineru_available, make_fn, ext: str
):
    """MineRU's API accepts every format the provider claims.

    This test hits the live MineRU API — only runs when explicitly
    opted in. We assert the extraction completes without raising
    ``OcrError``; recognition quality is the API's job, not ours.
    """
    from claritymed.core.ocr.mineru_provider import MineRUOcrProvider

    fixture = make_fn(tmp_path)
    assert fixture.suffix == ext
    provider = MineRUOcrProvider(os.environ["MINERU_API_TOKEN"])
    result = await provider.extract_text(fixture)
    # Non-empty text is the contract; substring match would be
    # over-specifying since MineRU normalises whitespace + adds layout.
    assert result.text.strip()
