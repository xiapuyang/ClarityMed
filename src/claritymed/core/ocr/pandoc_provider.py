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

from claritymed.core.ocr.base import ExtractResult, OcrEmpty, OcrError, OcrProvider

logger = logging.getLogger(__name__)


class PandocOcrProvider(OcrProvider):
    """Office formats → markdown. Empty extract → fall through."""

    is_local = True
    """``pandoc`` runs as a local subprocess — no network hop."""

    label = "pandoc"
    # Office/text-markup formats verified against pandoc 3.x via the
    # ``tests/e2e/ocr/`` matrix. Three exclusions reflect real gaps,
    # not over-conservatism:
    #
    # * Legacy binary office (``.doc`` / ``.ppt`` / ``.xls``) — pandoc
    #   shells out to ``antiword`` / ``catdoc`` / ``libreoffice`` for
    #   these, and ``pypandoc-binary`` (our distribution) bundles only
    #   the pandoc executable. Claiming support would route those files
    #   into pandoc and fail at runtime.
    # * ``.xlsx`` — pandoc 3.9.x has a workbook-relationship parsing bug
    #   ("Entry not found: xl//xl/worksheets/sheet1.xml", doubled path
    #   prefix) on openpyxl-generated workbooks. Re-add when that's
    #   fixed upstream and the e2e test starts passing.
    supported_extensions = frozenset(
        {
            ".docx",
            ".odt",
            ".rtf",
            ".epub",
            ".pptx",
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
            raise OcrEmpty("pypandoc: no text extracted")
        return ExtractResult(
            text=text, provider_used=self.label, chain_tried=[self.label]
        )

    @staticmethod
    def _extract_sync(path: Path) -> str:
        # Lazy: pypandoc may not be installed.
        import pypandoc

        return pypandoc.convert_file(str(path), "markdown")
