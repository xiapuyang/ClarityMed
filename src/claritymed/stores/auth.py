"""``PasswordStore`` — bcrypt hash storage at ``data/users/<id>/auth.yaml``.

Single field on disk (``password_hash``). Atomic write via temp-file +
rename under a ``filelock`` advisory lock so a concurrent reader cannot
observe a partial hash.

Bcrypt work factor is the library default (12 rounds in bcrypt 4.x).
Acceptable for the project's single-user / small-team scope — revisit
if a multi-tenant deployment surfaces.

Passwords are SHA-256-prehashed before bcrypt sees them. Two reasons:

* bcrypt 4+ rejects inputs longer than 72 bytes; prehash gives a fixed
  32-byte input that always fits.
* Unicode-heavy passwords whose UTF-8 representation exceeds 72 bytes
  silently lose tail characters under bcrypt 3's auto-truncation;
  prehash makes every character contribute.

The prehash + bcrypt composition is the canonical mitigation
recommended in OpenWall's bcrypt docs and used by Dropbox / Django.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import bcrypt
import yaml

from claritymed.core.locks import file_lock
from claritymed.stores.paths import user_root, validate_user_id

PASSWORD_FILE_NAME = "auth.yaml"
PASSWORD_FIELD = "password_hash"


def _prehash(plaintext: str) -> bytes:
    """Return ``sha256(plaintext)`` raw bytes (32 bytes; bcrypt-safe)."""
    return hashlib.sha256(plaintext.encode("utf-8")).digest()


def _hash(plaintext: str) -> str:
    """Return the bcrypt hash of ``plaintext`` as an ASCII string."""
    salted = bcrypt.hashpw(_prehash(plaintext), bcrypt.gensalt())
    return salted.decode("ascii")


def _verify(plaintext: str, hashed: str) -> bool:
    """Constant-time ``bcrypt.checkpw`` over the prehashed plaintext."""
    try:
        return bcrypt.checkpw(_prehash(plaintext), hashed.encode("ascii"))
    except (ValueError, UnicodeEncodeError):
        # Hash bytes were corrupt or non-ASCII — treat as no password.
        return False


def _auth_path(user_id: str) -> Path:
    return user_root(user_id) / PASSWORD_FILE_NAME


def _lock_path(user_id: str) -> Path:
    return user_root(user_id) / f"{PASSWORD_FILE_NAME}.lock"


class PasswordStore:
    """Static interface for set / verify on a per-user bcrypt hash."""

    @staticmethod
    def set_password(user_id: str, plaintext: str) -> None:
        """Hash ``plaintext`` with bcrypt and write atomically to disk.

        ``validate_user_id`` runs first so a bogus id (path traversal,
        oversize, non-ASCII) fails at the boundary rather than producing
        a sibling-directory write. The temp-file + rename idiom mirrors
        ``AccountStore.save``.
        """
        uid = validate_user_id(user_id)
        path = _auth_path(uid)
        path.parent.mkdir(parents=True, exist_ok=True)
        hashed = _hash(plaintext)
        with file_lock(_lock_path(uid)):
            tmp = path.with_suffix(path.suffix + ".tmp")
            try:
                with tmp.open("w", encoding="utf-8") as fh:
                    yaml.safe_dump({PASSWORD_FIELD: hashed}, fh, sort_keys=False)
                tmp.replace(path)
            except OSError:
                tmp.unlink(missing_ok=True)
                raise

    @staticmethod
    def verify_password(user_id: str, plaintext: str) -> bool:
        """Return True iff a hash exists for ``user_id`` and matches.

        Returns False — never raises — for missing file, malformed YAML,
        or a corrupt hash so the auth router has one clean path:
        "verify returned False ⇒ 401 with a generic body". The auth
        router is responsible for emitting a ``web.auth.login_failed``
        audit event so post-hoc analysis can separate enumeration from
        password-spray.
        """
        uid = validate_user_id(user_id)
        path = _auth_path(uid)
        if not path.exists():
            return False
        try:
            with path.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
        except OSError:
            return False
        if not isinstance(raw, dict):
            return False
        hashed = raw.get(PASSWORD_FIELD)
        if not isinstance(hashed, str) or not hashed:
            return False
        return _verify(plaintext, hashed)
