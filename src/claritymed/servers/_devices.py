"""Shared utilities for the RAG servers."""

from __future__ import annotations

from claritymed.core.device import resolve_device

# Passed as log_config to uvicorn.run() in every server so that uvicorn's own
# loggers (which default to propagate=False) also emit %(asctime)s timestamps.
LOG_CONFIG: dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stderr",
        },
    },
    "root": {"handlers": ["default"], "level": "INFO"},
    "loggers": {
        "uvicorn": {"propagate": True},
        "uvicorn.error": {"propagate": True},
        "uvicorn.access": {"propagate": True},
    },
}


def default_device() -> str:
    """Return ``"mps"`` / ``"cuda"`` / ``"cpu"`` based on torch's view."""
    return resolve_device("auto")
