"""Atomic YAML write helper for admin edits.

Read-side stays on :func:`claritymed.config.load_yaml`; the write side
lives here so it can:

1. Refuse paths not in the editable-configs allowlist.
2. Atomic-write via tempfile + ``os.replace``.
3. Invalidate the ``lru_cache`` on :func:`claritymed.config.load_yaml` so
   the next read sees the new contents.

The allowlist check is intentionally on the *name only* — never on a
joined path — so a payload like ``"../etc/passwd"`` is rejected before
:class:`Path` ever touches the file system.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from claritymed import config as _cfg
from claritymed.web.admin.allowlists import (
    EDITABLE_CONFIGS,
    EDITABLE_MODEL_CATALOGS,
)

logger = logging.getLogger(__name__)


def save_yaml(name: str, data: dict[str, Any]) -> Path:
    """Write ``data`` to ``CONFIGS_DIR / name`` atomically.

    Args:
        name: Bare filename (e.g. ``"app.yaml"``). Must be in
            :data:`EDITABLE_CONFIGS` ∪ :data:`EDITABLE_MODEL_CATALOGS`.
        data: Dict to serialize. ``yaml.safe_dump`` is used so only
            primitive Python types make it to disk.

    Returns:
        The absolute path that was written.

    Raises:
        ValueError: ``name`` not in either allowlist.
        OSError: Tempfile or replace failed.
    """
    if name not in EDITABLE_CONFIGS and name not in EDITABLE_MODEL_CATALOGS:
        raise ValueError(
            f"{name!r} is not in EDITABLE_CONFIGS or EDITABLE_MODEL_CATALOGS"
        )
    target = _cfg.CONFIGS_DIR / name
    # Belt-and-suspenders: assert the resolved path is still a child of
    # CONFIGS_DIR. Should be impossible after the allowlist check, but
    # the cost is one syscall.
    target_resolved = target.resolve()
    if _cfg.CONFIGS_DIR.resolve() not in target_resolved.parents:
        raise ValueError(f"resolved target escaped CONFIGS_DIR: {target_resolved!r}")
    body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(body)
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    _invalidate_cache()
    logger.info("admin.yaml_io.save_yaml wrote %s (%d bytes)", name, len(body))
    return target


def _invalidate_cache() -> None:
    """Blow away ``load_yaml``'s lru_cache so the next read is fresh.

    ``load_yaml`` is the ONLY cached read in the config layer that the
    admin write surface touches — clearing it is enough. Re-deriving
    fancier caches (validated Pydantic models in feature loaders) is the
    caller's responsibility.
    """
    cached = getattr(_cfg.load_yaml, "cache_clear", None)
    if cached is not None:
        cached()
