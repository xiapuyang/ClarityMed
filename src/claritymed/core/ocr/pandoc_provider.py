"""Office-format → markdown via ``pypandoc``.

Handles ``.docx``, ``.pptx``, ``.xlsx``, ``.odt`` and friends — the
non-PDF tail of the document chain. Pure-Python wrapper around the
``pandoc`` binary; installing the ``ocr-office`` extra ships the binary
alongside (via ``pypandoc-binary``) so users without pandoc on PATH
still get extraction.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from claritymed.core.ocr.base import ExtractResult, OcrError, OcrProvider

logger = logging.getLogger(__name__)


class PandocOcrProvider(OcrProvider):
    """Office formats → markdown. Empty extract → fall through."""

    is_local = True
    """``pandoc`` runs as a local subprocess — no network hop."""

    label = "pandoc"
    # Common office/text-markup formats pandoc handles well. PDFs go to
    # pymupdf/marker; images go to llm/mineru — explicitly excluded.
    supported_extensions = frozenset(
        {
            ".doc",
            ".docx",
            ".odt",
            ".rtf",
            ".epub",
            ".ppt",
            ".pptx",
            ".xls",
            ".xlsx",
            ".html",
            ".htm",
        }
    )

    async def extract_text(self, path: Path) -> ExtractResult:
        try:
            text = await asyncio.to_thread(self._extract_sync, path)
        except ImportError as exc:
            raise OcrError(
                "pypandoc is not installed; install with `uv sync --extra ocr-office`"
            ) from exc
        except FileNotFoundError as exc:
            raise OcrError(f"file not found: {path}") from exc
        except Exception as exc:  # noqa: BLE001
            raise OcrError(f"pypandoc extraction failed: {exc}") from exc

        if not text.strip():
            raise OcrError("pypandoc: no text extracted")
        return ExtractResult(
            text=text, provider_used=self.label, chain_tried=[self.label]
        )

    @staticmethod
    def _extract_sync(path: Path) -> str:
        # Lazy: pypandoc may not be installed.
        import pypandoc

        return pypandoc.convert_file(str(path), "markdown")
