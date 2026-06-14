"""Symptoms FastAPI server — multi-dataset typed-BASD inference.

Public surface is intentionally small: callers import :data:`app` for
ASGI mounting or testing, and :func:`main` is the entry point wired to
``claritymed-symptoms-server`` in ``pyproject.toml``.

Per the plan's KTD-7 the server is data-only: it returns raw clinical
data (differential with per-disease severity, evidence Q&A, turn count).
Tier-appropriate safety wording is composed by the LLM downstream from
``symptoms_final_reply.yaml`` — no ``safety_sentence`` field on the wire.
"""

from claritymed.servers.symptoms.app import app, main

__all__ = ["app", "main"]
