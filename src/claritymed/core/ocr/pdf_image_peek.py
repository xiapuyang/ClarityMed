"""Detect and rasterize 1-page image-only PDFs for the vision pipeline.

Some upstream tools wrap a single scan (CT slice, ultrasound capture)
in a PDF container even though the payload is one raster image. Such
PDFs would otherwise fail the vision tool's PIL decode step (``PIL``'s
PDF plugin is write-only), and they also bypass the medical-clip
modality classifier because the OCR worker's ``_IMAGE_EXTS`` whitelist
is extension-based.

This module solves both problems with one cheap check: open the PDF
with PyMuPDF, and if it is exactly one page with no meaningful text
and at least one embedded raster image, rasterize the page to PNG.
The OCR worker writes the result as a ``vision.png`` sidecar and
treats the blob as image-bearing thereafter — medical-clip runs on
the PNG, the vision server gets PNG bytes, and the LLM-facing routing
rules (``modality`` / ``is_medical``) fire as if the user had pasted
the raw image to begin with.

Multi-page PDFs and PDFs with significant text fall through to the
existing MineRU path — they're documents, not image wrappers.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# A 1-page PDF with more text than this is treated as a document (report,
# requisition form, etc.) and routed through the OCR/text pipeline. The
# tolerance covers small auto-inserted artefacts like page numbers or
# DICOM-export footers (``Page 1 / 1``, timestamps, study IDs) without
# letting a thin clinical note slip through.
_TEXT_LEN_CEILING_CHARS = 50

# 200 DPI gives ~1600x2000 px for a US-letter page — large enough that
# downstream medical-clip / vision models see the original detail of a
# scan that was wrapped at typical viewer DPI, small enough that the
# rasterized PNG stays well under 1 MB for the common case (e.g. a
# 381x282 ICC-based raster in a small PDF balloons to ~50-150 KB).
_RASTER_DPI = 200


def maybe_rasterize_single_image_pdf(pdf_path: Path) -> bytes | None:
    """Return rasterized PNG bytes for a 1-page image-only PDF, else None.

    Returns ``None`` for every PDF that should keep flowing through the
    normal OCR/text path:

    * any PyMuPDF failure (corrupt header, password-protected, etc.) —
      logged at debug, treated as "not our case";
    * page count != 1;
    * page contains more than :data:`_TEXT_LEN_CEILING_CHARS` characters
      of extractable text (likely a typed report);
    * page contains zero embedded raster images (blank page or
      vector-only chart — rasterizing would produce something
      medical-clip can't usefully classify).

    Args:
        pdf_path: Path to the on-disk PDF blob.

    Returns:
        PNG-encoded bytes when the PDF qualifies; ``None`` otherwise.
        Caller writes the bytes to a ``vision.png`` sidecar and treats
        the blob as image-bearing in the OCR worker.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.debug("pymupdf not installed; skipping single-image PDF peek")
        return None

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:  # noqa: BLE001 — fitz raises many shapes
        # Corrupt headers, encrypted PDFs, and "this isn't a PDF" all land
        # here. The right move is to let the existing OCR routing handle
        # whatever it is, not to crash the worker.
        logger.debug("pymupdf failed to open %s: %s", pdf_path.name, exc)
        return None

    try:
        if len(doc) != 1:
            return None
        page = doc[0]
        text = page.get_text("text").strip()
        if len(text) > _TEXT_LEN_CEILING_CHARS:
            return None
        # Embedded raster check — distinguishes "scan wrapped in PDF"
        # from "blank page" and "vector-only chart". ``get_images()``
        # returns one tuple per ``Image`` XObject reachable from the
        # page's resource dict.
        if len(page.get_images()) < 1:
            return None
        pixmap = page.get_pixmap(dpi=_RASTER_DPI, alpha=False)
        return pixmap.tobytes("png")
    except Exception as exc:  # noqa: BLE001
        # Rasterization itself can fail on exotic encodings (JBIG2 streams
        # without a decoder, JPEG2000 with strict colorspaces). Same
        # posture as the open failure — log and fall through.
        logger.debug("pymupdf rasterize failed for %s: %s", pdf_path.name, exc)
        return None
    finally:
        doc.close()


def has_vision_sidecar(blob_dir: Path) -> bool:
    """True iff the blob dir contains a ``vision.png`` sidecar.

    Set by :func:`maybe_rasterize_single_image_pdf` after a successful
    peek; downstream consumers (``_compute_vision_tags``,
    ``vision_plugin._read_blob_bytes``) check this to decide whether
    the blob has image bytes available regardless of the original
    ``content.*`` extension.
    """
    return (blob_dir / "vision.png").exists()


def vision_sidecar_path(blob_dir: Path) -> Path:
    """Compose the ``vision.png`` path — does not assert existence."""
    return blob_dir / "vision.png"


def write_vision_sidecar(blob_dir: Path, png_bytes: bytes) -> Path:
    """Atomically write ``vision.png`` next to the original content blob.

    Mirrors :class:`BlobStore`'s atomic-by-rename posture — we never
    leave a half-written ``vision.png`` even on disk-full, so the
    sidecar's *presence* is the unambiguous "this PDF was peeked and
    rasterized" signal. Idempotent: a second call with identical bytes
    is a no-op.
    """
    blob_dir.mkdir(parents=True, exist_ok=True)
    target = blob_dir / "vision.png"
    if target.exists() and target.read_bytes() == png_bytes:
        return target
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(png_bytes)
    tmp.replace(target)
    return target


__all__ = [
    "has_vision_sidecar",
    "maybe_rasterize_single_image_pdf",
    "vision_sidecar_path",
    "write_vision_sidecar",
]
