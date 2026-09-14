"""Tests for rapidocr, marker-pdf, and pandoc OCR providers.

All three providers shell out to optional heavy dependencies (onnxruntime
models, marker-pdf PyTorch, pandoc binary). Tests use ``unittest.mock``
to exercise the full ``extract_text`` logic without actually running OCR.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from claritymed.core.ocr.base import OcrEmpty, OcrError


# ---------------------------------------------------------------------------
# RapidOcrProvider
# ---------------------------------------------------------------------------


@pytest.fixture
def _rapidocr_provider():
    from claritymed.core.ocr.rapidocr_provider import RapidOcrProvider

    return RapidOcrProvider()


async def test_rapidocr_returns_text_on_success(tmp_path, _rapidocr_provider):
    img = tmp_path / "scan.png"
    img.write_bytes(b"\x89PNG")

    with patch.object(
        _rapidocr_provider, "_extract_sync", return_value="blood pressure 120/80"
    ):
        result = await _rapidocr_provider.extract_text(img)

    assert result.text == "blood pressure 120/80"
    assert result.provider_used == "rapidocr"
    assert result.chain_tried == ["rapidocr"]


async def test_rapidocr_raises_ocr_empty_when_no_text(tmp_path, _rapidocr_provider):
    img = tmp_path / "blank.png"
    img.write_bytes(b"\x89PNG")

    with patch.object(_rapidocr_provider, "_extract_sync", return_value=""):
        with pytest.raises(OcrEmpty, match="no text extracted"):
            await _rapidocr_provider.extract_text(img)


async def test_rapidocr_raises_ocr_error_on_import_error(tmp_path, _rapidocr_provider):
    img = tmp_path / "scan.png"
    img.write_bytes(b"\x89PNG")

    with patch.object(
        _rapidocr_provider,
        "_extract_sync",
        side_effect=ImportError("rapidocr_onnxruntime not found"),
    ):
        with pytest.raises(OcrError, match="rapidocr-onnxruntime is not installed"):
            await _rapidocr_provider.extract_text(img)


async def test_rapidocr_raises_ocr_error_on_file_not_found(
    tmp_path, _rapidocr_provider
):
    img = tmp_path / "ghost.png"

    with patch.object(
        _rapidocr_provider,
        "_extract_sync",
        side_effect=FileNotFoundError("no such file"),
    ):
        with pytest.raises(OcrError, match="file not found"):
            await _rapidocr_provider.extract_text(img)


async def test_rapidocr_raises_ocr_error_on_generic_exception(
    tmp_path, _rapidocr_provider
):
    img = tmp_path / "corrupt.png"
    img.write_bytes(b"\x89PNG")

    with patch.object(
        _rapidocr_provider,
        "_extract_sync",
        side_effect=RuntimeError("onnxruntime crash"),
    ):
        with pytest.raises(OcrError, match="rapidocr extraction failed"):
            await _rapidocr_provider.extract_text(img)


def _patched_rapidocr(engine_result):
    """Return a fake ``rapidocr_onnxruntime`` module whose ``RapidOCR()`` call
    returns an engine mock that yields *engine_result* when called."""
    engine = MagicMock(return_value=engine_result)
    module = MagicMock()
    module.RapidOCR = MagicMock(return_value=engine)
    return module


def test_rapidocr_extract_sync_calls_engine(tmp_path, _rapidocr_provider):
    """_extract_sync builds the engine on first call and joins text lines."""
    img = tmp_path / "scan.png"
    img.write_bytes(b"\x89PNG")

    # result shape: [[box, text, score], ...]
    fake_module = _patched_rapidocr(
        ([[None, "Glucose", 0.99], [None, "120 mg/dL", 0.97]], 0.05)
    )

    with patch.dict("sys.modules", {"rapidocr_onnxruntime": fake_module}):
        with patch("claritymed.core.ocr.rapidocr_provider.silence_fd_stderr") as _s:
            _s.return_value.__enter__ = MagicMock(return_value=None)
            _s.return_value.__exit__ = MagicMock(return_value=False)
            _rapidocr_provider._engine = None
            text = _rapidocr_provider._extract_sync(img)

    assert "Glucose" in text
    assert "120 mg/dL" in text


def test_rapidocr_extract_sync_empty_result_returns_empty(tmp_path, _rapidocr_provider):
    img = tmp_path / "blank.png"
    img.write_bytes(b"\x89PNG")

    fake_module = _patched_rapidocr((None, 0.0))  # None result = no text

    with patch.dict("sys.modules", {"rapidocr_onnxruntime": fake_module}):
        with patch("claritymed.core.ocr.rapidocr_provider.silence_fd_stderr") as _s:
            _s.return_value.__enter__ = MagicMock(return_value=None)
            _s.return_value.__exit__ = MagicMock(return_value=False)
            _rapidocr_provider._engine = None
            text = _rapidocr_provider._extract_sync(img)

    assert text == ""


# ---------------------------------------------------------------------------
# MarkerOcrProvider
# ---------------------------------------------------------------------------


@pytest.fixture
def _marker_provider():
    from claritymed.core.ocr.marker_provider import MarkerOcrProvider

    return MarkerOcrProvider()


async def test_marker_returns_text_on_success(tmp_path, _marker_provider):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF")

    with patch.object(
        _marker_provider, "_extract_sync", return_value="# Report\n\nSome findings."
    ):
        result = await _marker_provider.extract_text(pdf)

    assert result.text == "# Report\n\nSome findings."
    assert result.provider_used == "marker"
    assert result.chain_tried == ["marker"]


async def test_marker_raises_ocr_empty_on_blank_pdf(tmp_path, _marker_provider):
    pdf = tmp_path / "blank.pdf"
    pdf.write_bytes(b"%PDF")

    with patch.object(_marker_provider, "_extract_sync", return_value="   "):
        with pytest.raises(OcrEmpty, match="no text extracted"):
            await _marker_provider.extract_text(pdf)


async def test_marker_raises_ocr_error_on_import_error(tmp_path, _marker_provider):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF")

    with patch.object(
        _marker_provider,
        "_extract_sync",
        side_effect=ImportError("marker not installed"),
    ):
        with pytest.raises(OcrError, match="marker-pdf is not installed"):
            await _marker_provider.extract_text(pdf)


async def test_marker_raises_ocr_error_on_file_not_found(tmp_path, _marker_provider):
    pdf = tmp_path / "ghost.pdf"

    with patch.object(
        _marker_provider,
        "_extract_sync",
        side_effect=FileNotFoundError("no such file"),
    ):
        with pytest.raises(OcrError, match="file not found"):
            await _marker_provider.extract_text(pdf)


async def test_marker_raises_ocr_error_on_generic_exception(tmp_path, _marker_provider):
    pdf = tmp_path / "corrupt.pdf"
    pdf.write_bytes(b"%PDF")

    with patch.object(
        _marker_provider,
        "_extract_sync",
        side_effect=RuntimeError("torch crash"),
    ):
        with pytest.raises(OcrError, match="marker-pdf extraction failed"):
            await _marker_provider.extract_text(pdf)


def test_marker_extract_sync_uses_marker_api(tmp_path, _marker_provider):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF")

    mock_converter = MagicMock()
    mock_converter.return_value = MagicMock()
    mock_text_from_rendered = MagicMock(return_value=("# Findings", {}, {}))
    mock_create_model_dict = MagicMock(return_value={})

    with patch.dict(
        "sys.modules",
        {
            "marker": MagicMock(),
            "marker.converters": MagicMock(),
            "marker.converters.pdf": MagicMock(PdfConverter=mock_converter),
            "marker.models": MagicMock(create_model_dict=mock_create_model_dict),
            "marker.output": MagicMock(text_from_rendered=mock_text_from_rendered),
        },
    ):
        text = _marker_provider._extract_sync(pdf)

    assert text == "# Findings"


# ---------------------------------------------------------------------------
# PandocOcrProvider
# ---------------------------------------------------------------------------


@pytest.fixture
def _pandoc_provider():
    from claritymed.core.ocr.pandoc_provider import PandocOcrProvider

    return PandocOcrProvider()


async def test_pandoc_returns_text_on_success(tmp_path, _pandoc_provider):
    docx = tmp_path / "report.docx"
    docx.write_bytes(b"PK")

    with patch.object(
        _pandoc_provider, "_extract_sync", return_value="# Report\n\nContent here."
    ):
        result = await _pandoc_provider.extract_text(docx)

    assert result.text == "# Report\n\nContent here."
    assert result.provider_used == "pandoc"
    assert result.chain_tried == ["pandoc"]


async def test_pandoc_raises_ocr_empty_on_blank(tmp_path, _pandoc_provider):
    docx = tmp_path / "blank.docx"
    docx.write_bytes(b"PK")

    with patch.object(_pandoc_provider, "_extract_sync", return_value="\n\n"):
        with pytest.raises(OcrEmpty, match="no text extracted"):
            await _pandoc_provider.extract_text(docx)


async def test_pandoc_raises_ocr_error_on_import_error(tmp_path, _pandoc_provider):
    docx = tmp_path / "report.docx"
    docx.write_bytes(b"PK")

    with patch.object(
        _pandoc_provider,
        "_extract_sync",
        side_effect=ImportError("pypandoc not found"),
    ):
        with pytest.raises(OcrError, match="pypandoc is not installed"):
            await _pandoc_provider.extract_text(docx)


async def test_pandoc_raises_ocr_error_on_file_not_found(tmp_path, _pandoc_provider):
    docx = tmp_path / "ghost.docx"

    with patch.object(
        _pandoc_provider,
        "_extract_sync",
        side_effect=FileNotFoundError("no such file"),
    ):
        with pytest.raises(OcrError, match="file not found"):
            await _pandoc_provider.extract_text(docx)


async def test_pandoc_raises_ocr_error_on_generic_exception(tmp_path, _pandoc_provider):
    docx = tmp_path / "corrupt.docx"
    docx.write_bytes(b"PK")

    with patch.object(
        _pandoc_provider,
        "_extract_sync",
        side_effect=RuntimeError("pandoc binary error"),
    ):
        with pytest.raises(OcrError, match="pypandoc extraction failed"):
            await _pandoc_provider.extract_text(docx)


def test_pandoc_extract_sync_calls_pypandoc(tmp_path, _pandoc_provider):
    docx = tmp_path / "report.docx"
    docx.write_bytes(b"PK")

    mock_pypandoc = MagicMock()
    mock_pypandoc.convert_file.return_value = "## Title\n\nBody text."

    with patch.dict("sys.modules", {"pypandoc": mock_pypandoc}):
        text = _pandoc_provider._extract_sync(docx)

    assert text == "## Title\n\nBody text."
    mock_pypandoc.convert_file.assert_called_once_with(str(docx), "markdown")
