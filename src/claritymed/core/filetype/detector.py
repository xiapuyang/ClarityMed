"""Magika-backed filetype detection for extension-less paste targets.

The clipboard / drag-drop path normally trusts ``path.suffix`` to decide
how to ingest a file. That falls apart for two real cases:

* No suffix at all — ``Dockerfile``, ``README``, ``Makefile``.
* Wrong suffix — a file renamed from ``.jpg`` to ``.bin`` because some
  upstream sanitiser stripped the extension.

In both cases ``_ingest_clipboard_bytes`` would fall back to ``ext="bin"``
and the paste-time gate would reject the file as unsupported. Magika
recovers the true extension from the bytes via a small (~1 MB) ONNX
model, which we already host onnxruntime for (privacy filter).

This module is a thin wrapper: one process-wide ``Magika`` engine
constructed lazily, one ``detect`` entry point, no caller-side
configuration. Magika failures are non-fatal — the caller falls back
to the original extension and the gate decides downstream.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claritymed.core.observability.silence import silence_fd_stderr

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectResult:
    """Outcome of a single Magika identification.

    ``ext`` is the canonical leading-dot extension (e.g. ``".pdf"``)
    when Magika is confident, else ``None``. ``label`` and ``score``
    let audit logs reason about why the gate accepted or rejected the
    bytes.
    """

    ext: str | None
    label: str
    score: float
    mime_type: str


_engine: Any | None = None
_engine_lock = threading.Lock()


def _get_engine() -> Any | None:
    """Return the process-wide Magika engine, building it on demand.

    Returns ``None`` when ``magika`` isn't importable so the caller
    can fall back to the original extension cleanly. The construction
    is silenced through ``silence_fd_stderr`` because the ONNX session
    init writes provider notices to fd 2.
    """
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is not None:
            return _engine
        try:
            from magika import Magika

            with silence_fd_stderr():
                _engine = Magika()
        except Exception:  # noqa: BLE001
            logger.warning(
                "magika unavailable; filetype detection disabled", exc_info=True
            )
            _engine = None
        return _engine


def detect(data: bytes) -> DetectResult | None:
    """Identify *data* and return a ``DetectResult`` or ``None``.

    ``None`` means the detector itself failed (import error, model
    load error, inference error). A successful run with low confidence
    still returns a ``DetectResult`` — let the caller decide whether
    the score is high enough to act on, because the right threshold
    depends on the situation (gate acceptance vs audit-only logging).
    """
    engine = _get_engine()
    if engine is None:
        return None
    try:
        with silence_fd_stderr():
            result = engine.identify_bytes(data)
    except Exception:  # noqa: BLE001
        logger.warning("magika identify_bytes failed", exc_info=True)
        return None

    if not getattr(result, "ok", True):
        return None

    out = result.output
    extensions = list(getattr(out, "extensions", []) or [])
    # Prefer the first canonical extension. Magika returns the
    # vendor-canonical one first (e.g. ``["jpg", "jpeg"]`` → ``"jpg"``).
    ext = f".{extensions[0]}" if extensions else None
    return DetectResult(
        ext=ext,
        label=str(out.label),
        score=float(getattr(result, "score", 0.0)),
        mime_type=str(getattr(out, "mime_type", "application/octet-stream")),
    )


def detect_path(path: Path) -> DetectResult | None:
    """Convenience wrapper for callers holding a ``Path`` instead of bytes.

    Reads the file with ``read_bytes`` — fine for paste/drag-drop sizes
    (clipboard payloads are bounded by the OS). Falls back to ``None``
    on OS errors so the caller treats a read failure the same as a
    detector failure.
    """
    try:
        return detect(path.read_bytes())
    except OSError:
        logger.warning("filetype detect: cannot read %s", path, exc_info=True)
        return None
