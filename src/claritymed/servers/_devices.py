"""Best-available torch device probe shared by the RAG servers.

Both ``embedder`` and ``reranker`` pick a device at startup with the same
preference order (mps → cuda → cpu); centralising the probe keeps the
two server lifespans in lockstep so a future change (e.g. preferring
ROCm) lands in one place.
"""

from __future__ import annotations


def default_device() -> str:
    """Return ``"mps"`` / ``"cuda"`` / ``"cpu"`` based on torch's view.

    Catches any probe failure (no torch, broken backend, etc.) and falls
    back to ``cpu`` rather than crashing the server at import time.
    """
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001 — any probe failure → cpu
        pass
    return "cpu"
