"""Symptoms FastAPI server — multi-dataset typed-BASD inference.

Public surface is intentionally small: callers import :data:`app` for
ASGI mounting or testing, and :func:`main` is the entry point wired to
``claritymed-symptoms-server`` in ``pyproject.toml``.

Per the plan's KTD-7 the server is data-only: it returns raw clinical
data (differential with per-disease severity, evidence Q&A, turn count).
Tier-appropriate safety wording is composed by the LLM downstream from
``symptoms_final_reply.yaml`` — no ``safety_sentence`` field on the wire.

``app`` / ``main`` are exposed via PEP 562 lazy attribute access so pure
sibling modules (``wire``, ``state``, ``loader``, ``differential``) can be
imported without the ``symptoms-server`` extra installed — which is what
unit tests that only touch the wire schemas need.
"""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING, Any

__all__ = ["app", "main"]

if TYPE_CHECKING:
    from claritymed.servers.symptoms.app import app, main


def __getattr__(name: str) -> Any:
    if name in {"app", "main"}:
        module = importlib.import_module("claritymed.servers.symptoms.app")
        value = getattr(module, name)
        # importlib.import_module also binds the submodule as
        # ``claritymed.servers.symptoms.app`` (the module object) on this
        # package. Cache the real object under the same name so subsequent
        # ``from claritymed.servers.symptoms import app`` calls return the
        # FastAPI instance instead of the submodule.
        setattr(sys.modules[__name__], name, value)
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
