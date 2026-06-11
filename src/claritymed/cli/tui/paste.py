"""Cross-platform clipboard reader for the Ctrl+V paste flow.

Mirrors claude-code's ``imagePaste.ts`` three-state output:

* ``ImageBytes(bytes, ext)`` — clipboard has raw image bytes (PNG, JPEG).
* ``FilePath(path)`` — clipboard text looks like a file path; we resolve
  + read it on the caller's side.
* ``LargeText(text)`` — clipboard text >= 800 chars; UI inserts a
  ``[Pasted text #id]`` placeholder and stashes the body separately.
* ``SmallText(text)`` — short text; just falls through to the Input.
* ``Empty`` — clipboard contained nothing readable.

Implementation is best-effort per platform: macOS uses NSPasteboard
when ``pyobjc`` is available else osascript; Linux uses ``xclip``
fallback ``wl-paste``; Windows uses ``pywin32``. If none of the
platform paths can read the clipboard we return ``Empty`` and surface
a status-bar hint upstream.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageBytes:
    bytes: bytes
    ext: str  # ``"png"`` / ``"jpeg"`` etc.


@dataclass(frozen=True)
class FilePath:
    path: Path


@dataclass(frozen=True)
class LargeText:
    text: str


@dataclass(frozen=True)
class SmallText:
    text: str


@dataclass(frozen=True)
class Empty:
    pass


ClipboardContent = Union[ImageBytes, FilePath, LargeText, SmallText, Empty]

LARGE_TEXT_THRESHOLD = 800
# A modest path-like regex — full URI parsing belongs upstream.
_PATH_LOOKING = re.compile(r"^(/|~|[A-Za-z]:\\)[^\n\r]+\.[A-Za-z0-9]{1,8}$")


def read_clipboard() -> ClipboardContent:
    """Best-effort cross-platform clipboard read.

    Tries image bytes first, text second. Catches every platform-side
    failure into ``Empty`` so a missing helper never crashes the TUI.
    """
    try:
        if sys.platform == "darwin":
            return _read_macos()
        if sys.platform.startswith("linux"):
            return _read_linux()
        if sys.platform == "win32":
            return _read_windows()
    except Exception:  # noqa: BLE001
        logger.warning("clipboard read failed", exc_info=True)
    return Empty()


def classify_text(text: str) -> ClipboardContent:
    """Promote a raw text string to the right ClipboardContent subtype."""
    if not text:
        return Empty()
    if _PATH_LOOKING.match(text.strip()):
        p = Path(text.strip()).expanduser()
        if p.exists():
            return FilePath(p)
    if len(text) >= LARGE_TEXT_THRESHOLD:
        return LargeText(text)
    return SmallText(text)


# --- macOS --------------------------------------------------------------


def _read_macos() -> ClipboardContent:
    # Prefer pyobjc when available; fall back to osascript.
    try:
        from AppKit import NSPasteboard  # type: ignore

        pb = NSPasteboard.generalPasteboard()
        png_data = pb.dataForType_("public.png")
        if png_data is not None:
            return ImageBytes(bytes=bytes(png_data), ext="png")
        text = pb.stringForType_("public.utf8-plain-text")
        if text:
            return classify_text(str(text))
        return Empty()
    except ImportError:
        pass

    # Fallback: osascript can read text; image bytes require pyobjc.
    try:
        text = subprocess.check_output(
            ["osascript", "-e", "the clipboard as text"], text=True
        )
        return classify_text(text)
    except subprocess.CalledProcessError:
        return Empty()


# --- Linux --------------------------------------------------------------


def _read_linux() -> ClipboardContent:
    """xclip (X11) or wl-paste (Wayland). Both are optional."""
    # Image first.
    for cmd in (
        ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"],
        ["wl-paste", "--type", "image/png"],
    ):
        try:
            out = subprocess.run(cmd, check=True, capture_output=True)
            if out.stdout:
                return ImageBytes(bytes=out.stdout, ext="png")
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    # Text fallback.
    for cmd in (["xclip", "-selection", "clipboard", "-o"], ["wl-paste"]):
        try:
            out = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if out.stdout:
                return classify_text(out.stdout)
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    return Empty()


# --- Windows ------------------------------------------------------------


def _read_windows() -> ClipboardContent:
    try:
        import win32clipboard  # type: ignore
    except ImportError:
        return Empty()
    win32clipboard.OpenClipboard()
    try:
        # CF_DIB = 8 (bitmap); convert to PNG via Pillow for cross-platform sha.
        if win32clipboard.IsClipboardFormatAvailable(8):
            try:
                from io import BytesIO

                from PIL import Image  # type: ignore
            except ImportError:
                pass
            else:
                dib = win32clipboard.GetClipboardData(8)
                buf = BytesIO()
                Image.open(BytesIO(dib)).save(buf, format="PNG")
                return ImageBytes(bytes=buf.getvalue(), ext="png")
        if win32clipboard.IsClipboardFormatAvailable(13):  # CF_UNICODETEXT
            text = win32clipboard.GetClipboardData(13)
            return classify_text(text)
    finally:
        win32clipboard.CloseClipboard()
    return Empty()
