"""Image size + dimension guard rails for vision / OCR inference paths.

Enforced at the LLM OCR boundary (``LLMOcrProvider.extract_text``) before
the raw bytes are wrapped in a ``BinaryContent`` payload — out-of-range
inputs raise :class:`claritymed.errors.ImageTooLargeError` /
:class:`claritymed.errors.ImageTooSmallError` with the specific axis that
tripped, so the LLM never gets to bill us for a 50 MB DICOM, and a 50 px
icon never gets fed to a ViT encoder that would upscale it into garbage.

Limits are loaded from ``configs/app.yaml`` ``vision.image_limits`` via
:func:`claritymed.config.vision_image_limits`. PDFs are validated on
byte-size only — page-level pixel checks happen at the rasterizer hop
(``pdf_image_peek``) where dimensions actually exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from claritymed.errors import ImageTooLargeError, ImageTooSmallError

_PDF_SUFFIX = ".pdf"


@dataclass(frozen=True)
class ImageLimits:
    """Upper + lower bounds applied to a single image before inference.

    All fields are integers; bytes is the on-disk file size,
    ``dimension`` is max of width/height, and ``pixels`` is width*height.
    Bytes is checked unconditionally; dimensions and pixels are checked
    when the file is a raster (PIL-openable) and skipped for PDFs.
    """

    max_bytes: int
    max_dimension: int
    max_pixels: int
    min_bytes: int
    min_dimension: int
    min_pixels: int


def validate_image(path: Path, limits: ImageLimits) -> None:
    """Validate ``path`` against ``limits`` or raise a typed error.

    Order: bytes → dimension → pixels. First trip wins so the error
    message names the most specific violation. PDFs are byte-checked
    only; rasterized PDF pages get a second pass at the rasterizer.

    Args:
        path: Image (or PDF) file to validate. Must exist.
        limits: Bounds to enforce.

    Raises:
        ImageTooLargeError: File bytes / dimension / pixels exceeded a max.
        ImageTooSmallError: File bytes / dimension / pixels fell below a min.
        FileNotFoundError: ``path`` does not exist (bubbles from ``stat``).
    """
    size = path.stat().st_size
    if size > limits.max_bytes:
        raise ImageTooLargeError(
            f"{path.name}: {size:,} bytes exceeds max_bytes={limits.max_bytes:,}"
        )
    if size < limits.min_bytes:
        raise ImageTooSmallError(
            f"{path.name}: {size:,} bytes below min_bytes={limits.min_bytes:,}"
        )
    if path.suffix.lower() == _PDF_SUFFIX:
        return

    # Lazy import — Pillow is in the vision extra and may not be present
    # in minimal images; the byte check above still gives us a guard.
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError:
        return

    try:
        with Image.open(path) as im:
            width, height = im.size
    except (UnidentifiedImageError, OSError):
        # Not a raster Pillow can parse (e.g. a corrupted file). Let the
        # downstream loader surface the real read error; bytes already
        # cleared min/max so the guard isn't the layer to refuse here.
        return

    if width > limits.max_dimension or height > limits.max_dimension:
        raise ImageTooLargeError(
            f"{path.name}: {width}x{height} exceeds "
            f"max_dimension={limits.max_dimension}"
        )
    if width < limits.min_dimension or height < limits.min_dimension:
        raise ImageTooSmallError(
            f"{path.name}: {width}x{height} below min_dimension={limits.min_dimension}"
        )
    pixels = width * height
    if pixels > limits.max_pixels:
        raise ImageTooLargeError(
            f"{path.name}: {pixels:,} pixels exceeds max_pixels={limits.max_pixels:,}"
        )
    if pixels < limits.min_pixels:
        raise ImageTooSmallError(
            f"{path.name}: {pixels:,} pixels below min_pixels={limits.min_pixels:,}"
        )


__all__ = ["ImageLimits", "validate_image"]
