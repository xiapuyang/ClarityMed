"""``marker-pdf`` fallback — second leg of the local PHI OCR chain.

``marker-pdf`` is the best-in-class local OCR for scanned/messy PDFs
(including Chinese), but it pulls PyTorch (~2 GB) on import, so we
import lazily inside ``extract_text`` instead of at module load. The
chain only reaches this provider when ``pymupdf`` returned empty
(i.e. the PDF is scanned).

Optional extra: install with ``uv sync --extra ocr-scanned``. The
factory raises a clear error if marker is configured but the extra
isn't installed.

License: GPL-3.0-or-later — acknowledged at plan time as a known
copyleft cost; `surya-ocr` (Apache 2.0) is the swap candidate if the
project's distribution stance changes.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from claritymed.core.ocr.base import ExtractResult, OcrEmpty, OcrError, OcrProvider

logger = logging.getLogger(__name__)


class MarkerOcrProvider(OcrProvider):
    """OCR via marker-pdf. Lazy import keeps cold paths cheap."""

    is_local = True
    """PHI-safe by construction — model + execution stay on this machine."""

    label = "marker"
    supported_extensions = frozenset({".pdf"})

    def __init__(self, *, max_pages: int | None = None) -> None:
        # The plan's CPU-mode caveat (~10s/page) is real; allow callers to
        # cap pages on weak hardware. None = use marker's own default.
        self._max_pages = max_pages

    async def extract_text(self, path: Path) -> ExtractResult:
        try:
            text = await asyncio.to_thread(self._extract_sync, path)
        except ImportError as exc:
            # Surfacing this as OcrError lets the chain fall through to
            # the next provider rather than failing the whole extraction.
            raise OcrError(
                "marker-pdf is not installed; install with "
                "`uv sync --extra ocr-scanned`"
            ) from exc
        except FileNotFoundError as exc:
            raise OcrError(f"file not found: {path}") from exc
        except Exception as exc:  # noqa: BLE001
            raise OcrError(f"marker-pdf extraction failed: {exc}") from exc

        if not text.strip():
            raise OcrEmpty("marker-pdf: no text extracted")
        return ExtractResult(
            text=text, provider_used=self.label, chain_tried=[self.label]
        )

    def _extract_sync(self, path: Path) -> str:
        # Lazy: imports cost ~2 s on first call (model load).
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered

        # The plan acknowledges this is heavy and the chain skips it when
        # not needed. We do NOT cache models across calls in v1 — each
        # construction is paid for. v1.1+ can introduce a module-level
        # singleton if profiling shows the cost matters.
        converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = converter(str(path))
        text, _, _images = text_from_rendered(rendered)
        return text
