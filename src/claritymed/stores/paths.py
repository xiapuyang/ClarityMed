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


def user_rag_qdrant_dir() -> Path:
    """Qdrant directory for per-user RAG collections.

    All user_rag chunks live in one Qdrant instance here; isolation is
    structural via collection naming (``user_rag_<user_id>``). The directory
    is under DATA_DIR (per-user PHI scope), not SHARED_DIR.
    """
    return _cfg.DATA_DIR / "qdrant" / "user_rag"


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


def shared_qdrant_dir() -> Path:
    return _cfg.SHARED_DIR / "qdrant"


def shared_parent_docstore_path() -> Path:
    """LlamaIndex ``SimpleDocumentStore`` JSON for system RAG collections.

    Holds parent-chunk text for shared corpora (StatPearls, ...). Admin-
    managed: only ingest CLI writes here. Read path is open to any
    authenticated user — system parent text is not PHI by construction.
    """
    return _cfg.SHARED_DIR / "parent_docstore.json"


def shared_vision_models_dir(
    disease: str | None = None, version: str | None = None
) -> Path:
    base = _cfg.SHARED_DIR / "vision_models"
    if disease is None:
        return base
    if version is None:
        return base / disease
    return base / disease / version
