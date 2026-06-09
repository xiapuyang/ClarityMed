"""Device probe for the RAG servers — delegates to ``core.device``."""

from __future__ import annotations

from claritymed.core.device import resolve_device


def default_device() -> str:
    """Return ``"mps"`` / ``"cuda"`` / ``"cpu"`` based on torch's view."""
    return resolve_device("auto")
