"""Per-event ``manifest.yaml`` storage with optimistic-lock revision counter.

Each PHI record lives in ``records/<category>/<slug>/manifest.yaml`` and
each curated library entry in ``library/<category>/<slug>/manifest.yaml``.
ManifestStore is parameterized on ``scope`` so the two share one class —
the directory shape and locking discipline are identical.

The lock discipline is the same one ``ChatSession`` already uses for its
JSONL append: read current, mutate in memory, atomically rename a ``.tmp``
in place. The new piece is the ``revision`` field — every update must
declare what revision it read and the store rejects with
``RevisionConflict`` if the on-disk revision moved. That makes the
classic concurrent-edit race ("TUI + headless CLI both ran ``update_record``
on the same slug") loudly visible instead of last-writer-wins silent.
"""

from __future__ import annotations

import logging
import os
import secrets
from datetime import date as date_cls, datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Literal

import yaml

from claritymed.core.locks import file_lock
from claritymed.core.schemas.records import Manifest
from claritymed.errors import RecordNotFound, RevisionConflict
from claritymed.stores.paths import (
    user_library_dir,
    user_library_record_dir,
    user_record_dir,
    user_records_dir,
    validate_user_id,
)

logger = logging.getLogger(__name__)

Scope = Literal["records", "library"]
# 8-char base32-ish suffix; same alphabet as user_id so the slug regex
# applied at the path layer doesn't reject anything generated here.
_SLUG_ALPHABET = "abcdefghijkmnpqrstuvwxyz23456789"


def make_slug(when: date_cls | None = None) -> str:
    """Return ``<YYYY-MM-DD>-<8-char-shortuuid>``.

    Date prefix sorts naturally in directory listings; no ``kind`` field
    avoids the zh/en translation flakiness of having Chinese or English
    type words in the path. ``when=None`` uses today's UTC date.
    """
    d = when or datetime.now(timezone.utc).date()
    suffix = "".join(secrets.choice(_SLUG_ALPHABET) for _ in range(8))
    return f"{d.isoformat()}-{suffix}"


def _dump_yaml(data: dict) -> str:
    """Canonical YAML for manifests — explicit settings so re-reads round-trip."""
    return yaml.safe_dump(
        data,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )


def _atomic_write_text(target: Path, content: str) -> None:
    """``.tmp`` + os.rename — power-loss safe single-file write."""
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.rename(tmp, target)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


class ManifestStore:
    """One per (user, scope). The scope picks records/ vs library/.

    The brainstorm uses one model for both manifests (``Manifest`` in
    ``schemas/records``); a few library-only fields (``authors``, ``year``,
    ``public``) live on the same model and stay at defaults for records.
    Keeping one model and one store class avoids two parallel codepaths
    that would have to be kept in sync forever.
    """

    def __init__(self, user_id: str, scope: Scope) -> None:
        self.user_id = validate_user_id(user_id)
        if scope not in ("records", "library"):
            raise ValueError(f"invalid scope: {scope!r}")
        self.scope = scope

    # --- internal helpers --------------------------------------------

    def _scope_root(self, category: str | None = None) -> Path:
        if self.scope == "records":
            return user_records_dir(self.user_id, category)
        return user_library_dir(self.user_id, category)

    def _record_dir(self, category: str, slug: str) -> Path:
        if self.scope == "records":
            return user_record_dir(self.user_id, category, slug)
        return user_library_record_dir(self.user_id, category, slug)

    def _manifest_path(self, category: str, slug: str) -> Path:
        return self._record_dir(category, slug) / "manifest.yaml"

    def _lock_path(self, category: str, slug: str) -> Path:
        # Lock files live OUTSIDE the record dir. The previous layout
        # (``<record_dir>/manifest.yaml.lock``) made ``delete()`` race
        # itself: the ``with file_lock(...)`` block held a handle to a
        # file the same block then unlinked. POSIX got away with it
        # (silently); Windows refused with "file in use". A separate
        # ``.locks`` dir at the scope root keeps locks decoupled from
        # the data they protect and makes ``delete()`` safe everywhere.
        if self.scope == "records":
            locks_dir = user_records_dir(self.user_id) / ".locks"
        else:
            locks_dir = user_library_dir(self.user_id) / ".locks"
        locks_dir.mkdir(parents=True, exist_ok=True)
        return locks_dir / f"{category}__{slug}.lock"

    # --- API ---------------------------------------------------------

    def create(self, category: str, slug: str, data: dict) -> Path:
        """Write a fresh manifest at revision=1; refuse if slug already exists.

        ``data`` is whatever ``Manifest`` accepts — typically tool-args
        carried over from ``save_record`` / ``save_to_library`` plus a
        freshly generated slug. Date/category/slug must already be
        consistent with the on-disk path.
        """
        manifest_path = self._manifest_path(category, slug)
        if manifest_path.exists():
            # Idempotency-violation guard: someone called create twice for the
            # same slug, or two processes raced. ``create`` is the API for
            # genuinely-new manifests; ``update`` is the API for changes.
            raise FileExistsError(f"manifest already exists: {manifest_path}")

        payload = dict(data)
        payload.setdefault("revision", 1)
        payload.setdefault("schema_version", 1)
        payload["category"] = category
        payload["slug"] = slug

        # Validate via pydantic before writing — guarantees the on-disk file
        # is round-trippable to a Manifest model.
        manifest = Manifest.model_validate(payload)

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(self._lock_path(category, slug)):
            if manifest_path.exists():
                raise FileExistsError(f"manifest already exists: {manifest_path}")
            _atomic_write_text(
                manifest_path,
                _dump_yaml(manifest.model_dump(by_alias=True, mode="json")),
            )
        return manifest_path

    def read(self, category: str, slug: str) -> Manifest:
        """Parse the manifest into a pydantic ``Manifest``.

        Raises ``RecordNotFound`` (``LookupError`` subclass) when the slug
        does not exist; bubbles ``ValidationError`` when the on-disk file
        is corrupt (caller decides whether to surface or recover).
        """
        manifest_path = self._manifest_path(category, slug)
        if not manifest_path.exists():
            raise RecordNotFound(f"no manifest at {manifest_path}")
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        return Manifest.model_validate(raw)

    def update(
        self,
        category: str,
        slug: str,
        *,
        expected_revision: int,
        mutator: Callable[[dict], dict],
    ) -> Manifest:
        """File-lock-guarded R-M-W with monotonic revision counter.

        ``mutator`` receives the current manifest as a dict (with aliases
        applied — ``date``, not ``event_date``) and returns the new dict.
        Revision++ is handled inside this method, not by the mutator, so
        callers cannot accidentally skip or repeat a revision.
        """
        manifest_path = self._manifest_path(category, slug)
        if not manifest_path.exists():
            raise RecordNotFound(f"no manifest at {manifest_path}")

        with file_lock(self._lock_path(category, slug)):
            raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            current_revision = int(raw.get("revision", 0))
            if current_revision != expected_revision:
                raise RevisionConflict(
                    f"manifest revision={current_revision} but caller expected "
                    f"{expected_revision}: {manifest_path}"
                )

            new_data = mutator(dict(raw))
            new_data["revision"] = current_revision + 1
            new_data["updated_at"] = datetime.now(timezone.utc).isoformat()
            # Preserve category/slug — mutator must not move the record.
            new_data["category"] = category
            new_data["slug"] = slug

            manifest = Manifest.model_validate(new_data)
            _atomic_write_text(
                manifest_path,
                _dump_yaml(manifest.model_dump(by_alias=True, mode="json")),
            )
        return manifest

    def delete(self, category: str, slug: str) -> None:
        """Remove the slug directory entirely.

        Caller is responsible for the upstream Qdrant delete (see Unit 6's
        ``delete_record`` tool — Qdrant first, manifest second, fail-stop
        ordering). Direct callers of ``ManifestStore.delete`` are expected
        to know the cascade.

        Records can carry attachment sub-directories (e.g. ``attachments/``
        for blob copies the user pasted); recursive removal here so the
        whole record tree disappears in one atomic-from-the-caller's-view
        step. ``shutil.rmtree`` instead of ``rmdir`` so non-empty subdirs
        don't crash the call.
        """
        import shutil

        target = self._record_dir(category, slug)
        if not target.exists():
            raise RecordNotFound(f"no record dir at {target}")
        with file_lock(self._lock_path(category, slug)):
            shutil.rmtree(target)

    def list(self, category: str | None = None) -> Iterator[Path]:
        """Yield every ``manifest.yaml`` path under the scope (optionally a category)."""
        root = self._scope_root(category)
        if not root.exists():
            return
        if category is None:
            # Walk records/<category>/<slug>/manifest.yaml.
            for cat_dir in sorted(root.iterdir()):
                if not cat_dir.is_dir():
                    continue
                for slug_dir in sorted(cat_dir.iterdir()):
                    manifest_path = slug_dir / "manifest.yaml"
                    if manifest_path.exists():
                        yield manifest_path
            return
        for slug_dir in sorted(root.iterdir()):
            manifest_path = slug_dir / "manifest.yaml"
            if manifest_path.exists():
                yield manifest_path
