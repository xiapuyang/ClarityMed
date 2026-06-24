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
# Lowercase 64-char hex (sha256). Tools and the BlobStore both validate at the
# boundary so a typo never produces a sibling-directory collision.
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
# Category and slug occupy two real filesystem directory components. We allow
# the same alphabet as user_id (alnum, ``_``, ``-``) plus a dot — but no dot
# *prefix* (``.evil``) and no traversal segment. One regex, two components,
# zero room for path injection.
CATEGORY_RE = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9_\-]{0,63}$")
SLUG_RE = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9_\-]{0,127}$")


def _validate_sha256(sha: str) -> str:
    if not isinstance(sha, str) or not SHA256_RE.match(sha):
        raise ValueError(f"invalid sha256: {sha!r}")
    return sha


def _validate_category(category: str) -> str:
    if not isinstance(category, str) or not CATEGORY_RE.match(category):
        raise ValueError(f"invalid category: {category!r}")
    return category


def _validate_slug(slug: str) -> str:
    if not isinstance(slug, str) or not SLUG_RE.match(slug):
        raise ValueError(f"invalid slug: {slug!r}")
    return slug


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


def user_parent_docstore_phi_path(user_id: str) -> Path:
    """LlamaIndex parent-chunk JSON for the user's PHI Qdrant collection.

    PHI records (manifest-shaped events ingested via ``save_record``) embed
    into ``user_phi_<id>``; their parent-chunk text lives here, kept
    physically distinct from the library docstore so the two collections
    never share a backing file.
    """
    return user_root(user_id) / "parent_docstore_phi.json"


def user_parent_docstore_library_path(user_id: str) -> Path:
    """LlamaIndex parent-chunk JSON for the user's library Qdrant collection.

    Successor to ``user_parent_docstore_path`` once Unit 4 lands. Kept as a
    separate helper rather than a rename so the legacy filename isn't
    accidentally overwritten by a half-migrated developer install.
    """
    return user_root(user_id) / "parent_docstore_library.json"


# --- per-event manifest layout (records + library) ---------------------
#
# Records hold PHI events (one-checkup-per-directory). Library holds user-
# curated reference material (papers, articles, notes). They share the same
# directory shape so ManifestStore can target either with a ``scope`` arg.


def user_records_dir(user_id: str, category: str | None = None) -> Path:
    """``<user_root>/records/`` or ``<user_root>/records/<category>/``."""
    base = user_root(user_id) / "records"
    if category is None:
        return base
    return base / _validate_category(category)


def user_record_dir(user_id: str, category: str, slug: str) -> Path:
    """``<user_root>/records/<category>/<slug>/`` — one event directory.

    Holds ``manifest.yaml`` (event metadata) and its lockfile; attachments
    live in ``blobs/`` (CAS), referenced from the manifest by sha256.
    """
    return user_records_dir(user_id, category) / _validate_slug(slug)


def user_library_dir(user_id: str, category: str | None = None) -> Path:
    """``<user_root>/library/`` or ``<user_root>/library/<category>/``."""
    base = user_root(user_id) / "library"
    if category is None:
        return base
    return base / _validate_category(category)


def user_library_record_dir(user_id: str, category: str, slug: str) -> Path:
    """``<user_root>/library/<category>/<slug>/`` — one library entry."""
    return user_library_dir(user_id, category) / _validate_slug(slug)


# --- content-addressable blob store ------------------------------------
#
# The blob pool is two-level (sha[:2] / sha[64]) so a single directory never
# exceeds the few-thousand-entries point where filesystem dirent scans start
# to drag. ``content.<ext>`` is the original bytes; ``ocr.md`` / ``ocr.json``
# are sibling files written by the OCR worker.


def user_blobs_dir(user_id: str) -> Path:
    """``<user_root>/blobs/``."""
    return user_root(user_id) / "blobs"


def user_blob_dir(user_id: str, sha256: str) -> Path:
    """``<user_root>/blobs/<sha[:2]>/<sha>/`` — one blob's directory.

    Holds ``content.<ext>``, ``ocr.md``, ``ocr.json``. The directory is
    the unit of deduplication: the same PDF uploaded twice resolves to one
    directory and one OCR cache entry.
    """
    sha = _validate_sha256(sha256)
    return user_blobs_dir(user_id) / sha[:2] / sha


def user_blob_path(user_id: str, sha256: str, ext: str) -> Path:
    """``<blob_dir>/content.<ext>``. ``ext`` is the file's natural extension."""
    if not ext or "/" in ext or ext.startswith("."):
        raise ValueError(f"invalid blob extension: {ext!r}")
    return user_blob_dir(user_id, sha256) / f"content.{ext}"


# --- session-scoped state ----------------------------------------------
#
# A session_id is uuid4 per TUI launch. ChatSession already writes
# ``sessions/<sid>.jsonl``; attachments use ``session/<sid>/`` (note the
# singular) so the two are siblings rather than nested.


def user_session_dir(user_id: str, session_id: str) -> Path:
    """``<user_root>/session/<sid>/`` — per-session ephemeral state.

    Singular ``session`` (not ``sessions``) intentionally distinguishes the
    new per-session directory tree from the existing ``sessions/<sid>.jsonl``
    chat log.
    """
    if not session_id or "/" in session_id or ".." in session_id:
        raise ValueError(f"invalid session_id: {session_id!r}")
    return user_root(user_id) / "session" / session_id


def user_session_attachments_path(user_id: str, session_id: str) -> Path:
    """``<user_root>/session/<sid>/attachments.json`` — the SessionAttachments file."""
    return user_session_dir(user_id, session_id) / "attachments.json"


# --- audit-payload side-channel (mode 0600) ----------------------------
#
# ``audit.log`` keeps the one-line-per-event invariant with non-PHI fields;
# PHI text (titles, notes, extracted_labs values) goes to per-request files
# under ``audit_payloads/`` with owner-only mode bits.


def user_audit_payloads_dir(user_id: str) -> Path:
    """``<user_root>/audit_payloads/`` — owner-only PHI text side-channel."""
    return user_root(user_id) / "audit_payloads"


def user_audit_payload_path(user_id: str, request_id: str) -> Path:
    """``<user_root>/audit_payloads/<request_id>.json``."""
    if not request_id or "/" in request_id or ".." in request_id:
        raise ValueError(f"invalid request_id: {request_id!r}")
    return user_audit_payloads_dir(user_id) / f"{request_id}.json"


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


# --- admin paths --------------------------------------------------------


def jobs_dir() -> Path:
    """``DATA_DIR / "jobs"`` — on-disk mirror of the admin JobRegistry.

    Each running / finished job has a ``<job_id>.json`` here so the
    registry survives a process restart. Cleanup keeps the last 100
    finished jobs plus the rolling 30-day window.
    """
    return _cfg.DATA_DIR / "jobs"


def job_path(job_id: str) -> Path:
    """``DATA_DIR / "jobs" / "<job_id>.json"``.

    ``job_id`` is uuid4-shaped so traversal segments cannot reach this
    function via API input. The guard below is belt-and-suspenders.
    """
    if not job_id or "/" in job_id or ".." in job_id:
        raise ValueError(f"invalid job_id: {job_id!r}")
    return jobs_dir() / f"{job_id}.json"


def shared_vision_models_dir(
    disease: str | None = None, version: str | None = None
) -> Path:
    base = _cfg.SHARED_DIR / "vision_models"
    if disease is None:
        return base
    if version is None:
        return base / disease
    return base / disease / version
