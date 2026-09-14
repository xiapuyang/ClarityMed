"""Admin system-config CRUD endpoints.

Read every editable config, edit each one through an allowlisted dotted
path. The allowlist guarantees the surface stays narrow even as the
underlying YAMLs grow: a freshly added knob has zero attack surface
until an admin (us) explicitly registers it.

PATCH body: ``{"path": "i18n.default_lang", "value": "zh"}``. The
endpoint mutates the in-memory dict at ``path``, re-serializes via
``yaml.safe_dump`` through :func:`save_yaml`, and the underlying loader
``lru_cache`` is invalidated automatically so the next read is fresh.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from claritymed import config as _cfg
from claritymed.core.observability.audit import audit_event
from claritymed.web.admin.allowlists import (
    EDITABLE_CONFIGS,
    EDITABLE_KEYS_BY_CONFIG,
    is_editable_config,
    is_editable_config_key,
)
from claritymed.web.admin.yaml_io import save_yaml

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/configs", tags=["admin", "configs"])


class ConfigPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    value: Any


@router.get("")
async def list_editable_configs() -> dict[str, Any]:
    """Return the editable-config catalog so the SPA renders without
    hardcoding the list of files or per-key allowlists.
    """
    return {
        "configs": sorted(EDITABLE_CONFIGS),
        "editable_keys": {
            name: list(EDITABLE_KEYS_BY_CONFIG.get(name, ()))
            for name in sorted(EDITABLE_CONFIGS)
        },
    }


@router.get("/{name}")
async def read_config(name: str) -> dict[str, Any]:
    if not is_editable_config(name):
        raise HTTPException(
            status_code=400,
            detail=f"{name!r} is not in the editable configs allowlist",
        )
    try:
        data = _cfg.load_yaml(name)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"{name!r} not found on disk"
        ) from exc
    audit_event(
        "admin.config.read",
        payload={"name": name},
    )
    return {
        "name": name,
        "data": data,
        "editable_keys": list(EDITABLE_KEYS_BY_CONFIG.get(name, ())),
    }


@router.patch("/{name}")
async def patch_config(name: str, patch: ConfigPatch) -> dict[str, Any]:
    if not is_editable_config(name):
        raise HTTPException(
            status_code=400,
            detail=f"{name!r} is not in the editable configs allowlist",
        )
    if not is_editable_config_key(name, patch.path):
        raise HTTPException(
            status_code=400,
            detail=f"key {patch.path!r} is not editable in {name!r}",
        )
    try:
        data = dict(_cfg.load_yaml(name))
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"{name!r} not found on disk"
        ) from exc
    old_value = _get_dotted(data, patch.path)
    _set_dotted(data, patch.path, patch.value)
    try:
        save_yaml(name, data)
    except ValueError as exc:
        # save_yaml's allowlist guard. Should be unreachable after the
        # check above, but surface explicitly.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    audit_event(
        "admin.config.write",
        payload={
            "name": name,
            "path": patch.path,
            "old_value": old_value,
            "new_value": patch.value,
        },
    )
    return {"name": name, "data": data}


def _get_dotted(data: dict[str, Any], dotted: str) -> Any:
    cursor: Any = data
    for part in dotted.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cursor: Any = data
    for part in parts[:-1]:
        existing = cursor.get(part) if isinstance(cursor, dict) else None
        if not isinstance(existing, dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value
