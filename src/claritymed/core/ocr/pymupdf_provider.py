"""PyMuPDF text extraction — first leg of the local PHI OCR chain.

For digital PDFs (the common case for modern Chinese hospital reports),
``pymupdf`` returns extracted text in well under 100 ms with no model
load. For scanned PDFs the extract is empty; the routing layer treats
that as "this provider can't do it" and falls through to the next leg
(``marker-pdf`` → vision LLM).

License: AGPLv3 — acknowledged at plan time alongside the marker-pdf
GPL-3.0-or-later question. The project's distribution stance is
self-hosted; if that changes, ``pypdfium2`` (Apache 2.0) is the swap.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from claritymed.core.ocr.base import ExtractResult, OcrEmpty, OcrError, OcrProvider

logger = logging.getLogger(__name__)


class PyMuPDFOcrProvider(OcrProvider):
    """Digital-PDF text extraction via ``pymupdf`` (``fitz``).

    No external services, no model loads, no PHI hop. ``is_local = True``
    so the chain composer can always keep it on PHI paths even when
    ``phi_policy="local-only"`` is in effect.
    """

    is_local = True
    """PHI-safe by construction — never leaves the machine."""

    label = "pymupdf"
    supported_extensions = frozenset({".pdf"})

    async def extract_text(self, path: Path) -> ExtractResult:
        """Run blocking ``fitz`` in a thread; return concatenated text.

        Empty extract is treated as a "no text" signal so the routing
        layer can fall through to the next provider rather than
        returning a confusing empty string to the user.
        """
        try:
            text = await asyncio.to_thread(self._extract_sync, path)
        except FileNotFoundError as exc:
            raise OcrError(f"file not found: {path}") from exc
        except Exception as exc:  # noqa: BLE001
            # Any pymupdf-side failure (encrypted PDF, corrupt header) is
            # routable to the next provider — wrap into OcrError so the
            # chain handler picks it up.
            raise OcrError(f"pymupdf extraction failed: {exc}") from exc

        if not text.strip():
            # Scanned PDFs return empty extracts. Signal explicitly so
            # the routing chain falls through to marker / vision-LLM.
            raise OcrEmpty("pymupdf: no text extracted (likely scanned PDF)")
        return ExtractResult(
            text=text, provider_used=self.label, chain_tried=[self.label]
        )

    @staticmethod
    def _extract_sync(path: Path) -> str:
        # Lazy import — keeps cold ``import claritymed`` fast for callers
        # that never touch OCR.
        import fitz  # pymupdf, imported as ``fitz`` by historical convention

        pieces: list[str] = []
        with fitz.open(path) as doc:
            for page in doc:
                # ``markdown`` extractor is best on layouts with tables /
                # headings but falls back to plain on older versions; the
                # ``text`` extractor is the safe default we ship with.
                pieces.append(page.get_text("text"))
        return "\n\n".join(p for p in pieces if p)
