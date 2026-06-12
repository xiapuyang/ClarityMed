"""``silence_fd_stderr`` — suppress C-level stderr inside a block.

Some native libraries write diagnostics directly to file descriptor 2,
bypassing Python's ``sys.stderr``. Two places have hit this:

* ``cli/tui/paste.py``: macOS CoreGraphics' ImageIO emits
  ``cannot create jp2 color space, fallback to sRGB`` when canonicalising
  screenshot PNGs pulled off ``NSPasteboard``.
* ``core/scrub/service.py``: onnxruntime's CoreML execution provider
  prints initialization notices when the privacy-filter session loads.

Both garbled Textual's alternate-screen render. ``sys.stderr`` redirects
don't help because the writes go to fd 2 underneath Python. This
helper duplicates and restores fd 2 around the block, leaving
``sys.stderr`` (and the ``LazyStderrHandler``) untouched.

Defensive on ``os.dup(2)`` failure (fd 2 closed, rare test fixtures) —
yields without redirecting rather than raising, so a hardened test
environment doesn't crash code that was just trying to suppress noise.
"""

from __future__ import annotations

import contextlib
import os
from typing import Iterator


@contextlib.contextmanager
def silence_fd_stderr() -> Iterator[None]:
    """Redirect OS-level stderr (fd 2) to ``/dev/null`` for the block.

    Yields without doing anything when fd 2 isn't dupable (unusual
    test envs where stderr was closed). Always restores the original
    fd on exit, including when the wrapped block raises.
    """
    try:
        saved = os.dup(2)
    except OSError:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
    finally:
        os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
