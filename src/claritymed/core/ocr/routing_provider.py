"""File-type-aware routing OCR provider.

Routes extraction based on file extension:
  - Document files (.pdf .doc .docx .ppt .pptx .xls .xlsx) → document_provider
  - Image files (everything else) → image_default; on OcrError → image_fallback

Emits one ``ocr.extract`` audit event per call (best-effort — no-op when the
request ContextVars are not set, e.g. in unit tests).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from claritymed.core.ocr.base import OcrError, OcrProvider

logger = logging.getLogger(__name__)

_DOCUMENT_EXTENSIONS = frozenset(
    {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}
)


def _provider_label(provider: OcrProvider) -> str:
    """Short lowercase label derived from class name, e.g. 'mineru' or 'llm'."""
    name = type(provider).__name__  # e.g. "MineRUOcrProvider"
    return (
        name.lower()
        .replace("ocrprovider", "")
        .replace("ocr", "")
        .replace("provider", "")
    )


def _emit_audit(payload: dict[str, Any]) -> None:
    """Emit ocr.extract audit event."""
    try:
        from claritymed.core.observability.audit import audit_event

        audit_event("ocr.extract", payload)
    except Exception:  # noqa: BLE001
        logger.warning("ocr.extract audit failed", exc_info=True)


class RoutingOcrProvider(OcrProvider):
    """Routes OCR to different backends based on file type."""

    def __init__(
        self,
        *,
        document_provider: OcrProvider,
        image_default: OcrProvider,
        image_fallback: OcrProvider | None = None,
    ) -> None:
        self._document = document_provider
        self._image_default = image_default
        self._image_fallback = image_fallback

    async def extract_text(self, path: Path) -> str:
        t0 = time.perf_counter()
        size_bytes = path.stat().st_size if path.exists() else 0
        used_fallback = False
        provider_label = ""

        try:
            if path.suffix.lower() in _DOCUMENT_EXTENSIONS:
                logger.debug("ocr routing: %s → document provider", path.name)
                provider_label = _provider_label(self._document)
                result = await self._document.extract_text(path)
            else:
                logger.debug("ocr routing: %s → image default provider", path.name)
                provider_label = _provider_label(self._image_default)
                try:
                    result = await self._image_default.extract_text(path)
                except OcrError as exc:
                    if self._image_fallback is None:
                        raise
                    logger.warning(
                        "ocr routing: image default failed for %s (%s), trying fallback",
                        path.name,
                        exc,
                    )
                    provider_label = (
                        f"{_provider_label(self._image_default)}"
                        f"→{_provider_label(self._image_fallback)}"
                    )
                    used_fallback = True
                    result = await self._image_fallback.extract_text(path)

            duration_ms = int((time.perf_counter() - t0) * 1000)
            _emit_audit(
                {
                    "status": "ok",
                    "provider": provider_label,
                    "file": path.name,
                    "size_bytes": size_bytes,
                    "chars": len(result),
                    "duration_ms": duration_ms,
                    "fallback": used_fallback,
                }
            )
            return result

        except OcrError as exc:
            duration_ms = int((time.perf_counter() - t0) * 1000)
            _emit_audit(
                {
                    "status": "error",
                    "provider": provider_label or "unknown",
                    "file": path.name,
                    "size_bytes": size_bytes,
                    "duration_ms": duration_ms,
                    "error": str(exc),
                    "fallback": used_fallback,
                }
            )
            raise
