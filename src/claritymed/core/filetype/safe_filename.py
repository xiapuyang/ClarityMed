"""Normalize an arbitrary user-supplied filename for durable storage.

The blob CAS keys by SHA-256; the original filename is otherwise only
in ``SessionAttachments`` (per-session) and the chat manifest. To give
disaster-recovery a fallback we also drop the first-observed name into
``ocr.json``. Filenames coming off Ctrl+V, drag-drop, or upload can
contain anything — control bytes, Unicode normalization variants,
absurdly long paths from buggy upload widgets — so this helper bounds
them before they hit the sentinel.

What we do, and why:

* **NFC normalize.** macOS HFS+ returns NFD (``é`` = ``e`` + U+0301);
  Linux returns NFC. Without normalization the same uploaded photo
  ends up with two different ``original_filename`` values depending on
  the operating system that wrote ``ocr.json`` first. NFC makes the
  field stable and grep-friendly across platforms.
* **Control characters → ``"_"``.** ``\\x00``-``\\x1f`` and DEL never
  belong in real filenames, but a malicious paste payload could embed
  them. Replacing keeps the string JSON-safe and prevents log injection.
* **255-char cap.** Mirrors the POSIX ``NAME_MAX`` so we don't store
  more than a filesystem could ever produce. Truncation preserves the
  extension when one's present (last ``.`` within the trailing 16
  chars) — ``"a"*1000 + ".pdf"`` becomes ``"a"*251 + ".pdf"`` not
  ``"a"*255``.

What we do NOT do:

* Strip path separators. JSON stores ``/`` and ``\\`` safely, and the
  original_filename is for human display, never for re-constructing
  filesystem paths.
* Reject unicode. Chinese, emoji, RTL — all preserved verbatim.
* Lower-case or fold. The user's display name is the user's
  display name.
"""

from __future__ import annotations

import unicodedata

MAX_FILENAME_LEN = 255
_MAX_SUFFIX_LEN = 16
"""Longest suffix we consider an "extension" for truncation-preserve.

A ``.tar.gz`` is two segments but ``rpartition('.')`` only sees the
last; that's fine — ``.gz`` is the meaningful one for an OS-side
type sniff. Cap at 16 so a malicious ``a.aaaaaaaaaaaaaaaaaa`` doesn't
get treated as having a "real" extension."""


def safe_filename(name: str | None) -> str | None:
    """Return a JSON-safe, length-bounded, NFC-normalized filename.

    ``None`` and empty / whitespace-only inputs return ``None`` so the
    sentinel can omit the field rather than carrying noise.
    """
    if not name:
        return None
    stripped = name.strip()
    if not stripped:
        return None

    normalized = unicodedata.normalize("NFC", stripped)
    cleaned = "".join(
        ch if (ord(ch) >= 0x20 and ord(ch) != 0x7F) else "_" for ch in normalized
    )
    if len(cleaned) <= MAX_FILENAME_LEN:
        return cleaned

    # Truncation that preserves an extension when one looks real.
    stem, dot, suffix = cleaned.rpartition(".")
    if dot and stem and 0 < len(suffix) < _MAX_SUFFIX_LEN:
        keep = MAX_FILENAME_LEN - len(suffix) - 1
        return stem[:keep] + "." + suffix
    return cleaned[:MAX_FILENAME_LEN]
