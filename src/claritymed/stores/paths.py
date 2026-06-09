"""Single source of truth for runtime file paths.

All store code asks ``paths.user_root("alice")`` rather than concatenating
``DATA_DIR / "users" / ...`` inline — this keeps the per-user / shared split
in one place and makes it impossible to typo a sibling user directory.

``DATA_DIR`` and ``SHARED_DIR`` are read from ``claritymed.config`` as a
module reference so tests that reload the config see the redirected path.
"""

from __future__ import annotations

import re
from pathlib import Path

from claritymed import config as _cfg
from claritymed.errors import InvalidUserIdError

USER_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,32}$")


def validate_user_id(user_id: str) -> str:
    """Reject ``..``, slashes, length overflow. Returns the same id on success."""
    if not isinstance(user_id, str) or not USER_ID_RE.match(user_id):
        raise InvalidUserIdError(f"invalid user_id: {user_id!r}")
    return user_id


# --- per-user paths -----------------------------------------------------


def user_root(user_id: str) -> Path:
    return _cfg.DATA_DIR / "users" / validate_user_id(user_id)


def user_db_path(user_id: str) -> Path:
    return user_root(user_id) / "profile.db"


def user_sessions_dir(user_id: str) -> Path:
    """One JSONL per chat session under ``<user_root>/sessions/``."""
    return user_root(user_id) / "sessions"


def user_uploads_dir(user_id: str) -> Path:
    return user_root(user_id) / "uploads"


def user_settings_path(user_id: str) -> Path:
    return user_root(user_id) / "settings.yaml"


def user_rag_qdrant_dir(user_id: str) -> Path:
    """Per-user Qdrant local-mode storage directory.

    user_rag uses ``qdrant-client``'s local file-locked SQLite mode (each
    user gets their own ``<user_root>/qdrant/storage.sqlite``) rather
    than the shared server Docker that holds system collections. The
    split is intentional: PHI never leaves the per-user filesystem
    scope, OS file-mode bits enforce isolation, and a code bug picking
    the wrong path is structurally impossible since the path is derived
    from ``user_id``. Trade-off: the same user can't concurrently
    upload (``rag add``) and query (TUI) — file lock is exclusive — but
    cross-user reads/writes are physically separate.

    See ``docs/rag-setup.md`` §user_rag for the why and the caveat.
    """
    return user_root(user_id) / "qdrant"


def user_parent_docstore_path(user_id: str) -> Path:
    """LlamaIndex ``SimpleDocumentStore`` JSON for a user's RAG uploads.

    Holds parent-chunk text keyed by id; child chunks live in Qdrant. Per-
    user PHI: kept under ``data/users/<id>/`` so file isolation does the
    work that a cross-user filter would otherwise have to. Mirrors the
    foundation §04 lesson (filter-only isolation is unreliable).
    """
    return user_root(user_id) / "parent_docstore.json"


def list_user_ids() -> list[str]:
    """Return ids of users that have a ``settings.yaml`` on disk.

    Scanning for ``settings.yaml`` rather than directory presence is the
    'first user becomes admin' bootstrap correctness fix — a dangling empty
    directory must not trick the next account into being treated as second.
    """
    users_root = _cfg.DATA_DIR / "users"
    if not users_root.exists():
        return []
    return sorted(
        p.name
        for p in users_root.iterdir()
        if p.is_dir() and USER_ID_RE.match(p.name) and (p / "settings.yaml").exists()
    )


# --- shared paths (admin-managed) ---------------------------------------


def shared_root() -> Path:
    return _cfg.SHARED_DIR


def shared_knowledge_raw_dir() -> Path:
    return _cfg.SHARED_DIR / "knowledge" / "raw"


def shared_knowledge_normalized_dir() -> Path:
    return _cfg.SHARED_DIR / "knowledge" / "normalized"


def shared_parent_docstore_path() -> Path:
    """LlamaIndex ``SimpleDocumentStore`` JSON for system RAG collections.

    Holds parent-chunk text for shared corpora (StatPearls, ...). Admin-
    managed: only ingest CLI writes here. Read path is open to any
    authenticated user — system parent text is not PHI by construction.
    """
    return _cfg.SHARED_DIR / "parent_docstore.json"


def shared_terminology_dir() -> Path:
    """Directory holding the UMLS+CMeKG terminology export.

    ``concepts.jsonl`` lives here. Admin-managed, cross-user, non-PHI —
    same category as ``shared_knowledge_*_dir()``. Operators populate it
    via ``scripts/init_terminology.py``.
    """
    return _cfg.SHARED_DIR / "terminology"


def shared_terminology_jsonl() -> Path:
    """Canonical ``concepts.jsonl`` location under ``shared/terminology/``."""
    return shared_terminology_dir() / "concepts.jsonl"


def shared_vision_models_dir(
    disease: str | None = None, version: str | None = None
) -> Path:
    base = _cfg.SHARED_DIR / "vision_models"
    if disease is None:
        return base
    if version is None:
        return base / disease
    return base / disease / version
