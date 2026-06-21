"""ClarityMed web layer — FastAPI app + JWT cookie auth + chat SSE.

The web package mirrors the CLI/TUI surface for browser clients. All
business primitives (AskService, ChatSession, PhiGuard, AccountStore)
are reused unchanged; the web layer only wires HTTP request handling,
cookie-based auth, CSRF, and SSE-formatted streaming around them.

Install with ``uv sync --extra web``. Boot with ``claritymed-web``.
"""
