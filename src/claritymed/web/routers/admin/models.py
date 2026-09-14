"""Admin model catalogs + secrets endpoints.

Two surfaces under ``/admin/models``:

* Catalogs (``models.yaml``, ``vision.yaml``, ``medical_clip.yaml``,
  ``symptoms.yaml``) — read + full-document overwrite, validated
  against existing Pydantic schemas before commit.
* Secrets — view the manifest with masked values (``set`` / ``missing``,
  never the value itself) and write to ``~/.claritymed/.env`` via
  ``atomic_write_env``.

Catalog writes are accepted as a full YAML body so the SPA can drive
the Versions & Flow panel without the backend caring about the per-
disease panel UX — the panel constructs the right YAML mutation and
submits the whole document.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from claritymed import config as _cfg
from claritymed.core.observability.audit import audit_event
from claritymed.web.admin.allowlists import (
    EDITABLE_MODEL_CATALOGS,
    is_editable_model_catalog,
)
from claritymed.web.admin.secrets import atomic_write_env, masked_view
from claritymed.web.admin.secrets_manifest import EXPECTED_SECRETS
from claritymed.web.admin.yaml_io import save_yaml

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["admin", "models"])


class CatalogReplace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: dict[str, Any]


class SecretsPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    updates: dict[str, str]


@router.get("/catalogs")
async def list_catalogs() -> dict[str, Any]:
    return {"catalogs": sorted(EDITABLE_MODEL_CATALOGS)}


@router.get("/catalogs/{name}")
async def read_catalog(name: str) -> dict[str, Any]:
    if not is_editable_model_catalog(name):
        raise HTTPException(
            status_code=400,
            detail=f"{name!r} is not in the editable catalogs allowlist",
        )
    try:
        data = _cfg.load_yaml(name)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"{name!r} not found on disk"
        ) from exc
    return {"name": name, "data": data}


@router.patch("/catalogs/{name}")
async def patch_catalog(name: str, body: CatalogReplace) -> dict[str, Any]:
    if not is_editable_model_catalog(name):
        raise HTTPException(
            status_code=400,
            detail=f"{name!r} is not in the editable catalogs allowlist",
        )
    try:
        old = _cfg.load_yaml(name)
    except FileNotFoundError:
        old = {}
    # Re-validate via the existing per-catalog schema before commit
    # whenever we know about it. New catalogs without a registered
    # schema fall through with a structural-only check (dict at top).
    _validate_catalog(name, body.data)
    save_yaml(name, body.data)
    audit_event(
        "admin.models.write",
        payload={
            "name": name,
            "previous_top_level_keys": sorted(old.keys())
            if isinstance(old, dict)
            else [],
            "new_top_level_keys": sorted(body.data.keys()),
        },
    )
    return {"name": name, "data": body.data}


@router.get("/secrets")
async def read_secrets() -> dict[str, Any]:
    return {
        "manifest": [
            {
                "key": s.key,
                "label": s.label,
                "hint": s.hint,
                "category": s.category,
                "required": s.required,
            }
            for s in EXPECTED_SECRETS.values()
        ],
        "status": [
            {
                "key": row.key,
                "label": row.label,
                "hint": row.hint,
                "category": row.category,
                "required": row.required,
                "is_set": row.is_set,
                "source": row.source,
            }
            for row in masked_view()
        ],
    }


@router.patch("/secrets")
async def patch_secrets(body: SecretsPatch) -> dict[str, Any]:
    if not body.updates:
        return {"keys_changed": [], "restart_required": False}
    try:
        atomic_write_env(body.updates)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    audit_event(
        "admin.models.secrets.write",
        payload={"keys_changed": sorted(body.updates.keys())},
    )
    return {
        "keys_changed": sorted(body.updates.keys()),
        "restart_required": True,
    }


def _validate_catalog(name: str, data: dict[str, Any]) -> None:
    """Best-effort re-validation against the existing config schema.

    The check is intentionally lenient: a new top-level key admins
    want to introduce shouldn't be rejected just because the schema
    doesn't know it yet. We only fail when the Pydantic validator
    raises a structural error against a known schema.
    """
    if not isinstance(data, dict):
        raise HTTPException(
            status_code=422, detail="catalog must be a YAML mapping at top level"
        )
    try:
        if name == "vision.yaml":
            from claritymed.core.vision.schemas import VisionConfig

            VisionConfig.model_validate(data)
        elif name == "models.yaml":
            from claritymed.stores.models import ModelsConfig

            ModelsConfig.model_validate(data)
    except ImportError:
        return
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422,
            detail=f"catalog re-validation failed: {exc}",
        ) from exc
