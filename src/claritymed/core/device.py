"""Torch device resolution shared across the codebase.

Single source of truth for the mps → cuda → cpu probe. Both the RAG
servers and the scrub service use this so a future backend preference
(e.g. ROCm) is a one-line change.
"""

from __future__ import annotations


def resolve_device(device: str = "auto") -> str:
    """Return a concrete torch device string.

    ``"auto"`` probes for the best available backend: mps → cuda → cpu.
    Any explicit string (``"cpu"``, ``"mps"``, ``"cuda"``) is returned
    unchanged. Falls back to ``"cpu"`` on any probe failure (no torch,
    broken backend, etc.) so callers never crash at import time.
    """
    if device != "auto":
        return device
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"
