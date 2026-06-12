"""RapidOCR — local raster-image OCR fallback for ``image_chain``.

When the vision LLM (omlx) is unreachable — server down, GPU OOM,
model not loaded — ``image_chain`` would otherwise have nothing left
under ``phi_policy="local-only"``. RapidOCR fills that gap:

* PaddleOCR detection + recognition models ported to ONNX, runs on the
  same ``onnxruntime`` we already pull in for the PHI scrubber.
* CPU-only — no GPU dependency, different failure surface from omlx.
* Chinese OCR quality is the primary reason for choosing it over
  tesseract (the alternative): clinical documents in this app's target
  market are predominantly Chinese.
* Apache 2.0.

Optional extra: install with ``uv sync --extra ocr-image``.

Output is a markdown-flavored layout: each detected text region on its
own line, in detection order. Empty extracts raise ``OcrError`` so the
chain falls through.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from claritymed.core.ocr.base import ExtractResult, OcrError, OcrProvider
from claritymed.core.observability.silence import silence_fd_stderr

logger = logging.getLogger(__name__)


class RapidOcrProvider(OcrProvider):
    """RapidOCR (ONNX PaddleOCR) — CPU-local raster-image fallback."""

    is_local = True
    """Pure local inference — model + execution stay on this machine."""

    label = "rapidocr"
    # Common raster formats RapidOCR accepts via its internal image
    # loader. PDFs intentionally excluded — pymupdf / marker handle
    # those upstream. GIF excluded because RapidOCR doesn't unpack
    # multi-frame GIFs and a single-frame GIF rarely contains text
    # worth extracting; if that turns out wrong, e2e will surface it.
    supported_extensions = frozenset(
        {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".bmp",
            ".tiff",
            ".tif",
        }
    )

    def __init__(self) -> None:
        # ``rapidocr_onnxruntime`` lazy-loads its ONNX models on first
        # call; constructing the engine eagerly keeps the first-OCR
        # latency off the user's path but is still optional. We defer
        # to per-call construction inside ``_extract_sync`` so a
        # process that never paste-falls-back to rapidocr never pays
        # the model-load cost.
        self._engine = None

    async def extract_text(self, path: Path) -> ExtractResult:
        try:
            text = await asyncio.to_thread(self._extract_sync, path)
        except ImportError as exc:
            # Surface as OcrError so the chain falls through instead of
            # breaking the entire extraction. Operators see the missing
            # extra in the error string.
            raise OcrError(
                "rapidocr-onnxruntime is not installed; install with "
                "`uv sync --extra ocr-image`"
            ) from exc
        except FileNotFoundError as exc:
            raise OcrError(f"file not found: {path}") from exc
        except Exception as exc:  # noqa: BLE001
            raise OcrError(f"rapidocr extraction failed: {exc}") from exc

        if not text.strip():
            raise OcrError("rapidocr: no text extracted")
        return ExtractResult(
            text=text, provider_used=self.label, chain_tried=[self.label]
        )

    def _extract_sync(self, path: Path) -> str:
        if self._engine is None:
            # Lazy import — keeps cold ``import claritymed`` fast for
            # callers that never touch image OCR. ``RapidOCR`` writes
            # ONNX provider init notices to fd 2 on first load; silence
            # to keep Textual's alternate screen clean.
            from rapidocr_onnxruntime import RapidOCR

            with silence_fd_stderr():
                self._engine = RapidOCR()

        with silence_fd_stderr():
            result, _elapse = self._engine(str(path))
        if not result:
            return ""
        # ``result`` is a list of ``[box, text, score]``. We discard
        # boxes/scores — the chain contract is plain text, and any
        # downstream consumer that needs layout would use marker/mineru
        # not rapidocr.
        return "\n".join(line[1] for line in result if line and len(line) >= 2)
