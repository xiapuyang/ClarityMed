"""Write-side helpers for ``~/.claritymed/.env``.

Read-side already lives in :func:`claritymed.config.load_env_file`. This
module adds the bits the admin UI needs:

* :func:`atomic_write_env` — apply a ``{KEY: value}`` update map and rewrite
  the file under ``os.replace`` so a torn write never leaves a half-file on
  disk. Mode is forced to ``0o600`` so the file is unreadable by other
  unix users even if it lived through a prior umask quirk.
* :func:`masked_view` — given the manifest, classify each known key as
  ``set`` / ``missing`` and report whether the active value came from the
  env file or a pre-existing shell env var. The value itself is never
  returned; the editor only sees ``{key, set, source}`` triples.

Secrets edits never go through ``load_env_file`` — that function applies
``override=False`` semantics for startup. The admin write helper here
unconditionally rewrites the on-disk file; whether new values reach
provider clients on this process depends on whether they re-read
``os.environ`` (most don't; restart is the user-facing contract).
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from claritymed import config as _cfg
from claritymed.web.admin.secrets_manifest import EXPECTED_SECRETS, SecretSpec

logger = logging.getLogger(__name__)

ENV_FILE_MODE = 0o600

SecretSource = Literal["env", "file", "missing"]


@dataclass(frozen=True)
class SecretStatus:
    """One row in the masked view returned to the admin UI."""

    key: str
    label: str
    hint: str
    category: str
    required: bool
    is_set: bool
    source: SecretSource


def env_file_path() -> Path:
    """Return ``CLARITYMED_HOME / '.env'`` (the runtime env file).

    Resolved each call so tests can rebind ``CLARITYMED_HOME`` via env var
    and see the result without reloading the module.
    """
    return _cfg.CLARITYMED_HOME / ".env"


def _read_env_file(path: Path) -> dict[str, str]:
    """Parse a ``KEY=VALUE`` env file into a dict. Returns ``{}`` if absent.

    Same parser semantics as :func:`claritymed.config.load_env_file` so the
    write path round-trips with the read path: ``#``-comment lines skipped,
    optional ``export`` prefix tolerated, surrounding quotes stripped.
    """
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        out[key] = value.strip().strip('"').strip("'")
    return out


def atomic_write_env(updates: dict[str, str]) -> dict[str, str]:
    """Apply ``updates`` to the env file and rewrite atomically.

    Reads the existing file (if any), overwrites the keys in ``updates``,
    writes to a tempfile in the same dir, and ``os.replace``s. The result
    is set to mode ``0o600``.

    Args:
        updates: ``{KEY: value}`` map. Keys must appear in
            :data:`EXPECTED_SECRETS` — unknown keys raise ``ValueError``
            before the file is touched. Empty values are written as
            ``KEY=`` so the operator can clear a secret without deleting
            the line manually.

    Returns:
        The full ``{KEY: value}`` map that was written, including pre-existing
        unrelated keys carried over from the previous file. Useful for tests.

    Raises:
        ValueError: A key in ``updates`` is not in the manifest.
        OSError: The file or its parent directory cannot be written.
    """
    unknown = set(updates) - EXPECTED_SECRETS.keys()
    if unknown:
        raise ValueError(
            f"Unknown secret key(s) {sorted(unknown)!r}; extend "
            "claritymed.web.admin.secrets_manifest.EXPECTED_SECRETS first."
        )
    path = env_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = _read_env_file(path)
    merged.update(updates)
    body = "\n".join(f"{k}={v}" for k, v in merged.items()) + "\n"
    # Tempfile in the same directory so ``os.replace`` is atomic across
    # the rename (cross-mount renames would not be).
    fd, tmp_name = tempfile.mkstemp(prefix=".env.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(body)
        os.replace(tmp_name, path)
    except Exception:
        # Best-effort cleanup; if the tempfile is still on disk after the
        # replace failed, the operator can remove it.
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    try:
        os.chmod(path, ENV_FILE_MODE)
    except OSError:
        # Windows or odd filesystem — don't fail the write.
        logger.warning("could not chmod %s to 0o600", path)
    return merged


def masked_view(specs: dict[str, SecretSpec] | None = None) -> list[SecretStatus]:
    """Return a list of ``SecretStatus`` rows for the admin UI.

    Source attribution: when both the env file and ``os.environ`` carry a
    value, ``source="env"`` (because shell-level env wins per the project
    rule). When only the file has it, ``source="file"``. Otherwise
    ``source="missing"``.
    """
    specs = specs or EXPECTED_SECRETS
    file_keys = _read_env_file(env_file_path())
    rows: list[SecretStatus] = []
    for spec in specs.values():
        in_env = bool(os.environ.get(spec.key))
        in_file = bool(file_keys.get(spec.key))
        if in_env:
            source: SecretSource = "env"
            is_set = True
        elif in_file:
            source = "file"
            is_set = True
        else:
            source = "missing"
            is_set = False
        rows.append(
            SecretStatus(
                key=spec.key,
                label=spec.label,
                hint=spec.hint,
                category=spec.category,
                required=spec.required,
                is_set=is_set,
                source=source,
            )
        )
    return rows
