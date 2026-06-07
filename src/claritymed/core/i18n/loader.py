"""i18n loader (mirrors ``fin/fin/i18n.py``, YAML instead of JSON).

``t(key, lang=None, **fmt)`` is the only public surface. Translations live in
``configs/i18n/{en,zh}.yaml`` and reload automatically when the file's mtime
changes — admins editing copy do not need to restart the app.

Language resolution order:

1. Explicit ``lang=`` argument.
2. ``language_ctx`` ContextVar (set by CLI entry / FastAPI middleware).
3. ``configs/app.yaml`` ``i18n.default_lang``.
4. ``"en"``.

Unlike fin we do not consult OS locale — language is a per-request setting in
ClarityMed, not a machine preference.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any

import yaml

from claritymed.config import I18N_DIR, default_lang
from claritymed.context import language_ctx

logger = logging.getLogger(__name__)

_cache: dict[str, dict[str, Any]] = {}
_mtime: dict[str, float] = {}
_lock = Lock()


def resolve_lang(explicit: str | None = None) -> str:
    """Return the active language code for this call."""
    if explicit:
        return explicit.lower()
    from_ctx = language_ctx.get()
    if from_ctx:
        return from_ctx.lower()
    fallback = default_lang().lower()
    return fallback if fallback in ("en", "zh") else "en"


def _load(lang: str) -> dict[str, Any]:
    """Read ``configs/i18n/<lang>.yaml`` with mtime-based caching.

    Returns an empty dict when the file is missing — better to fall back to
    the bare key than crash mid-request because someone is editing a YAML.
    """
    path = I18N_DIR / f"{lang}.yaml"
    if not path.exists():
        return {}

    try:
        mtime = path.stat().st_mtime
    except OSError:
        return _cache.get(lang, {})

    with _lock:
        if _mtime.get(lang) == mtime and lang in _cache:
            return _cache[lang]
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except (OSError, yaml.YAMLError) as exc:
            logger.warning("i18n: failed to load %s: %s", path, exc)
            return _cache.get(lang, {})
        flat = _flatten(data)
        _cache[lang] = flat
        _mtime[lang] = mtime
        return flat


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Convert nested YAML to dotted keys: ``{"ui": {"x": "y"}}`` -> ``{"ui.x": "y"}``."""
    out: dict[str, str] = {}
    for key, value in data.items():
        full = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten(value, full))
        elif value is not None:
            out[full] = str(value)
    return out


def t(key: str, lang: str | None = None, **fmt: Any) -> str:
    """Translate ``key`` for the active language.

    Lookup order: active locale -> English -> the bare key string. ``**fmt`` is
    passed through ``str.format``; a malformed template returns the unformatted
    string and logs a warning rather than raising.
    """
    active = resolve_lang(lang)
    value = _load(active).get(key)
    if value is None and active != "en":
        value = _load("en").get(key)
    if value is None:
        return key
    if fmt:
        try:
            return value.format(**fmt)
        except (KeyError, IndexError, ValueError) as exc:
            logger.warning("i18n: format failed for %r: %s", key, exc)
            return value
    return value


def _reset_for_tests() -> None:
    """Test hook: clear cache so a fixture YAML edit is visible immediately."""
    with _lock:
        _cache.clear()
        _mtime.clear()
