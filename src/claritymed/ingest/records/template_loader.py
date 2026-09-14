"""Load + validate a template directory; emit a content-canonical ``import_id``.

Contract overview (mirror of plan §"Template directory shape"):

* The directory is **flat** — one ``_meta.yaml`` plus one ``<user_id>.yaml``
  per uid listed in ``_meta.user_ids``. Subdirs, hidden files, ``.DS_Store``,
  symlinks → ``TemplateValidationError``.
* Every ``case.case_id`` is globally unique across all per-user files; the
  loader enforces this BEFORE any writer touches disk.
* Every ``case.kind`` is required (no defaulting from ``category``); the
  loader honors the model-level guarantee from ``template_schema``.

``import_id`` is the sha256 prefix of *content-canonical* bytes, not raw
file bytes — see ``_canonicalize_for_hash``. Two skill runs at different
timestamps over identical case content produce the same id; ``.DS_Store``
contamination cannot move the id (such files are loader errors).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from claritymed.errors import TemplateValidationError
from claritymed.ingest.records.template_schema import (
    CaseEntry,
    MetaConfig,
    UserBundle,
)
from claritymed.stores.paths import USER_ID_RE

META_FILENAME = "_meta.yaml"
IMPORT_ID_LENGTH = 12
CLEANUP_FAILED_MARKER = ".cleanup_failed"


@dataclass(frozen=True)
class LoadedTemplate:
    """Validated template snapshot ready for the orchestrator.

    ``bundles`` is keyed by the per-user filename stem (Unit 4 derives
    user_id from the filename — the orchestrator must always use the
    dict key, never any in-body ``user_id``). ``warnings`` carries
    non-fatal advisory messages — e.g. a case whose ``kind`` looks like
    a plural folder name. The CLI surfaces these in dry-run output.
    """

    meta: MetaConfig
    bundles: dict[str, UserBundle]
    import_id: str
    warnings: list[str]


def load_template(template_dir: Path) -> LoadedTemplate:
    """Validate a template directory; return the ``LoadedTemplate``.

    Fail-loud at the first invariant violation. The orchestrator copies
    the directory into ``data/_imports/<id>/template/`` only after this
    function returns cleanly, so a failure here leaves no on-disk state
    behind.
    """
    if not template_dir.is_dir():
        raise TemplateValidationError(
            f"template directory does not exist or is not a directory: {template_dir}"
        )

    if (template_dir / CLEANUP_FAILED_MARKER).exists():
        raise TemplateValidationError(
            f"template directory has a {CLEANUP_FAILED_MARKER} marker — a "
            "prior cleanup failed and the directory is in an unknown state. "
            "Inspect the marker contents and resolve manually before "
            f"resuming: {template_dir}"
        )

    accepted_files = _enumerate_directory(template_dir)
    meta_path = template_dir / META_FILENAME
    meta = _load_meta(meta_path)
    bundles_by_uid = _load_user_bundles(template_dir, accepted_files, meta)
    warnings: list[str] = []
    _apply_case_defaults(bundles_by_uid, meta, warnings)
    _check_cross_file_case_id_uniqueness(bundles_by_uid)
    _check_attachment_paths(bundles_by_uid)

    import_id = _compute_import_id(template_dir, accepted_files)

    return LoadedTemplate(
        meta=meta,
        bundles=bundles_by_uid,
        import_id=import_id,
        warnings=warnings,
    )


# --- step 1: enumerate -------------------------------------------------


def _enumerate_directory(template_dir: Path) -> list[Path]:
    """Return the sorted list of accepted file paths.

    Allowed:
      * ``_meta.yaml`` (exactly one).
      * ``<user_id>.yaml`` where ``<user_id>`` matches ``USER_ID_RE``.

    Anything else (subdir, hidden file, ``.DS_Store``, symlink, file with
    a non-YAML suffix or a malformed stem) is rejected. ``META_FILENAME``
    is required.
    """
    accepted: list[Path] = []
    saw_meta = False
    for entry in sorted(template_dir.iterdir()):
        if entry.is_symlink():
            raise TemplateValidationError(
                f"template directory contains a symlink (not allowed): {entry.name}"
            )
        if entry.is_dir():
            raise TemplateValidationError(
                f"template directory must be flat — subdirs not allowed: {entry.name}"
            )
        if entry.name == META_FILENAME:
            saw_meta = True
            accepted.append(entry)
            continue
        if not entry.name.endswith(".yaml"):
            raise TemplateValidationError(
                f"unexpected file in template directory (only _meta.yaml + "
                f"<user_id>.yaml allowed): {entry.name}"
            )
        stem = entry.name[: -len(".yaml")]
        if not USER_ID_RE.match(stem):
            raise TemplateValidationError(
                f"per-user file stem must match USER_ID_RE: {entry.name}"
            )
        accepted.append(entry)
    if not saw_meta:
        raise TemplateValidationError(
            f"template directory missing required {META_FILENAME}"
        )
    return accepted


# --- step 2: load _meta.yaml ------------------------------------------


def _load_meta(meta_path: Path) -> MetaConfig:
    raw = _read_yaml(meta_path)
    try:
        return MetaConfig.model_validate(raw)
    except ValidationError as exc:
        raise TemplateValidationError(f"{meta_path.name}: {exc}") from exc


# --- step 3 + 4: load each per-user file ------------------------------


def _load_user_bundles(
    template_dir: Path,
    accepted_files: list[Path],
    meta: MetaConfig,
) -> dict[str, UserBundle]:
    bundles_by_uid: dict[str, UserBundle] = {}
    per_user_files = [p for p in accepted_files if p.name != META_FILENAME]
    per_user_uids = {p.name[: -len(".yaml")] for p in per_user_files}
    declared_uids = set(meta.user_ids)

    # Loader-level mismatch: every declared uid needs a file; every file
    # needs to be declared. Two distinct messages because the remediation
    # differs (write the missing file vs delete the stray file vs add
    # the stray uid to ``_meta.user_ids``).
    missing = declared_uids - per_user_uids
    if missing:
        raise TemplateValidationError(
            f"per-user file missing for uids in _meta.user_ids: {sorted(missing)}"
        )
    stray = per_user_uids - declared_uids
    if stray:
        raise TemplateValidationError(
            f"per-user file present but not listed in _meta.user_ids: {sorted(stray)}"
        )

    for path in per_user_files:
        uid = path.name[: -len(".yaml")]
        raw = _read_yaml(path)
        try:
            bundles_by_uid[uid] = UserBundle.model_validate(raw)
        except ValidationError as exc:
            raise TemplateValidationError(f"{path.name}: {exc}") from exc

    return bundles_by_uid


# --- step 5: case_id cross-file uniqueness ---------------------------


def _check_cross_file_case_id_uniqueness(
    bundles_by_uid: dict[str, UserBundle],
) -> None:
    seen: dict[str, str] = {}
    for uid, bundle in bundles_by_uid.items():
        for case in bundle.cases:
            other = seen.get(case.case_id)
            if other is not None:
                raise TemplateValidationError(
                    f"case_id collision across files: {case.case_id!r} "
                    f"appears in both {other}.yaml and {uid}.yaml"
                )
            seen[case.case_id] = uid


# --- step 6: backfill category + warn on suspicious kind --------------


def _apply_case_defaults(
    bundles_by_uid: dict[str, UserBundle],
    meta: MetaConfig,
    warnings: list[str],
) -> None:
    """Backfill ``case.category`` from ``meta.default_category``.

    Bundles are frozen, so we rebuild the case with the resolved
    category in place. Bundles dict is mutated by reference; the
    UserBundle object is replaced.
    """
    for uid, bundle in list(bundles_by_uid.items()):
        rebuilt_cases: list[CaseEntry] = []
        for case in bundle.cases:
            resolved_category = case.category or meta.default_category
            if resolved_category is None:
                raise TemplateValidationError(
                    f"{uid}.yaml: case {case.case_id!r} has no category and "
                    "no _meta.default_category to fall back on"
                )
            # Heuristic: kind == category likely means a skill author
            # confused the two (kind is singular like 'lab-report'; category
            # is plural like 'lab-reports'). Not a hard reject — surface in
            # warnings so the human reviewer can flag it during dry-run.
            if case.kind == resolved_category:
                warnings.append(
                    f"{uid}.yaml: case {case.case_id!r} has kind == category "
                    f"({resolved_category!r}); kind should be singular "
                    "(e.g. 'lab-report') while category is plural"
                )
            if case.category != resolved_category:
                rebuilt_cases.append(
                    case.model_copy(update={"category": resolved_category})
                )
            else:
                rebuilt_cases.append(case)
        bundles_by_uid[uid] = bundle.model_copy(update={"cases": rebuilt_cases})


# --- step 7: attachment paths exist + not symlinks --------------------


def _check_attachment_paths(bundles_by_uid: dict[str, UserBundle]) -> None:
    for uid, bundle in bundles_by_uid.items():
        for case in bundle.cases:
            for att in case.attachments:
                attachment_path = Path(att.path)
                if attachment_path.is_symlink():
                    raise TemplateValidationError(
                        f"{uid}.yaml: case {case.case_id!r} attachment "
                        f"{att.original_filename!r} is a symlink (not allowed)"
                    )
                if not attachment_path.is_file():
                    raise TemplateValidationError(
                        f"{uid}.yaml: case {case.case_id!r} attachment "
                        f"path does not exist: {att.path}"
                    )


# --- step 8: import_id canonicalization -------------------------------


def _compute_import_id(
    template_dir: Path,
    accepted_files: list[Path],
) -> str:
    """Hash content-canonical bytes (not raw bytes).

    Why content-canonical: two skill runs at different ``created_at``
    timestamps over identical case content produce the same id. Two
    YAML files written with different key ordering / quote style /
    trailing newline produce the same id. The hash is also invariant
    to which parent directory the template lives under (we hash only
    the relative path and the canonical body).
    """
    sorted_files = sorted(accepted_files, key=lambda p: p.name)
    hasher = hashlib.sha256()
    for entry in sorted_files:
        rel = entry.relative_to(template_dir).as_posix()
        raw = _read_yaml(entry)
        canonical = _canonicalize_for_hash(raw, entry.name)
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(canonical)
        hasher.update(b"\x00")
    return hasher.hexdigest()[:IMPORT_ID_LENGTH]


def _canonicalize_for_hash(raw: Any, filename: str) -> bytes:
    """Strip volatile fields, re-dump with stable sort + encoding.

    For ``_meta.yaml`` the ``created_at`` field is volatile (changes
    between skill runs over identical content) so we strip it before
    hashing. All other fields hash as-is. ``sort_keys=True`` defeats
    YAML key reordering; ``allow_unicode=True`` defeats Unicode escape
    drift; ``default_flow_style=False`` pins block style.
    """
    if filename == META_FILENAME and isinstance(raw, dict):
        clone = {k: v for k, v in raw.items() if k != "created_at"}
    else:
        clone = raw
    return yaml.safe_dump(
        clone,
        sort_keys=True,
        allow_unicode=True,
        default_flow_style=False,
    ).encode("utf-8")


# --- shared helpers ---------------------------------------------------


def _read_yaml(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TemplateValidationError(f"cannot read {path.name}: {exc}") from exc
    try:
        return yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise TemplateValidationError(f"{path.name}: YAML parse failed: {exc}") from exc
