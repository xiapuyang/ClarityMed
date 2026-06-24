"""Admin i18n editing endpoints.

Two surfaces:

* Backend strings (``configs/i18n/<lang>.yaml``) — global UI copy used
  by chat/TUI/audit messages. Editable per language. Writes go through
  the same atomic-tempfile + cache invalidation as the other configs.
* Admin UI strings (``src/claritymed/web/admin_ui/src/i18n/<lang>.json``)
  — the SPA's own copy. Writes take effect on the next ``make admin-ui``
  rebuild; with Vite HMR running they're live. The endpoint surfaces
  that via the ``needs_rebuild`` flag.

The backend YAML loader supports both ``<lang>.yaml`` and a per-domain
``<lang>/<domain>.yaml`` set; this endpoint only edits the global
``<lang>.yaml`` for simplicity. Per-domain edits remain CLI-only.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict

from claritymed import config as _cfg
from claritymed.core.observability.audit import audit_event

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/i18n", tags=["admin", "i18n"])

SUPPORTED_LANGUAGES = ("en", "zh")


class I18nPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    updates: dict[str, Any]


def _validate_lang(lang: str) -> str:
    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"unsupported language: {lang!r}")
    return lang


def _backend_path(lang: str) -> Path:
    return _cfg.I18N_DIR / f"{lang}.yaml"


def _admin_ui_path(lang: str) -> Path:
    # The admin_ui source lives at
    # ``src/claritymed/web/admin_ui/src/i18n/<lang>.json`` relative to
    # the package root. Resolving via __file__ keeps it correct even
    # when CLARITYMED_HOME is redirected for tests.
    return (
        Path(__file__).resolve().parents[2]
        / "admin_ui"
        / "src"
        / "i18n"
        / f"{lang}.json"
    )


@router.get("/backend-strings")
async def read_backend_strings(
    lang: str = Query(default="en"),
) -> dict[str, Any]:
    _validate_lang(lang)
    path = _backend_path(lang)
    if not path.exists():
        return {"lang": lang, "data": {}}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise HTTPException(
            status_code=500, detail=f"invalid YAML on disk: {exc}"
        ) from exc
    return {"lang": lang, "data": data}


@router.patch("/backend-strings")
async def patch_backend_strings(
    body: I18nPatch, lang: str = Query(default="en")
) -> dict[str, Any]:
    _validate_lang(lang)
    path = _backend_path(lang)
    path.parent.mkdir(parents=True, exist_ok=True)
    current: dict[str, Any] = {}
    if path.exists():
        current = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    _deep_merge(current, body.updates)
    _atomic_write(path, yaml.safe_dump(current, allow_unicode=True, sort_keys=False))
    audit_event(
        "admin.i18n.write",
        payload={
            "surface": "backend",
            "lang": lang,
            "keys_changed": sorted(_flatten_keys(body.updates)),
        },
    )
    return {"lang": lang, "data": current, "needs_rebuild": False}


@router.get("/admin-strings")
async def read_admin_strings(
    lang: str = Query(default="en"),
) -> dict[str, Any]:
    _validate_lang(lang)
    path = _admin_ui_path(lang)
    if not path.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "admin_ui locale missing — install the SPA source tree before "
                "editing admin strings"
            ),
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500, detail=f"invalid JSON on disk: {exc}"
        ) from exc
    return {"lang": lang, "data": data}


@router.patch("/admin-strings")
async def patch_admin_strings(
    body: I18nPatch, lang: str = Query(default="en")
) -> dict[str, Any]:
    _validate_lang(lang)
    path = _admin_ui_path(lang)
    if not path.exists():
        raise HTTPException(
            status_code=500,
            detail="admin_ui locale missing on disk",
        )
    current = json.loads(path.read_text(encoding="utf-8"))
    _deep_merge(current, body.updates)
    _atomic_write(path, json.dumps(current, ensure_ascii=False, indent=2))
    audit_event(
        "admin.i18n.write",
        payload={
            "surface": "admin_ui",
            "lang": lang,
            "keys_changed": sorted(_flatten_keys(body.updates)),
        },
    )
    return {"lang": lang, "data": current, "needs_rebuild": True}


def _deep_merge(target: dict[str, Any], updates: dict[str, Any]) -> None:
    """Merge ``updates`` into ``target`` in place. Nested dicts compose."""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            _deep_merge(target[k], v)
        else:
            target[k] = v


def _flatten_keys(data: dict[str, Any], prefix: str = "") -> list[str]:
    out: list[str] = []
    for k, v in data.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.extend(_flatten_keys(v, full))
        else:
            out.append(full)
    return out


def _atomic_write(path: Path, body: str) -> None:
    fd, tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(body)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
