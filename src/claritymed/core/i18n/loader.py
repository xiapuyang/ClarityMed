"""i18n loader (mirrors ``fin/fin/i18n.py``, YAML instead of JSON).

``t(key, lang=None, **fmt)`` is the only public surface. Translations live in:

* ``configs/i18n/<lang>.yaml`` — the global namespace (app-wide UI copy,
  red-flag wording, mode labels, …).
* ``configs/i18n/<lang>/*.yaml`` — per-domain namespace files merged into
  the same flat key dict (introduced so the symptoms feature can ship a
  large per-dataset translation table without bloating the global YAML).

All files for a given language are loaded + flattened + merged into one
``key -> string`` dict per language. Reload is mtime-driven: a fingerprint
combining every file's mtime invalidates the cache when any file changes,
so admins editing copy do not need to restart the app. Subdirectory keys
later in the merge order win over earlier ones; the global ``<lang>.yaml``
is loaded first and per-domain files override it — useful when a feature
needs to rebrand a global key for its surface without touching the global.

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
from pathlib import Path
from threading import Lock
from typing import Any

import yaml

from claritymed.config import I18N_DIR, default_lang
from claritymed.context import language_ctx

logger = logging.getLogger(__name__)

_cache: dict[str, dict[str, Any]] = {}
_fingerprint: dict[str, tuple[tuple[str, float], ...]] = {}
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


def _candidate_paths(lang: str) -> list[Path]:
    """Return every file that contributes to ``lang``'s flat dict.

    Order:

    1. ``configs/i18n/<lang>.yaml`` (global) — loaded first so per-domain
       files can override.
    2. ``configs/i18n/<lang>/*.yaml`` (per-domain) sorted by filename so
       the merge order is deterministic across hosts.
    """
    paths: list[Path] = []
    base = I18N_DIR / f"{lang}.yaml"
    if base.exists():
        paths.append(base)
    domain_dir = I18N_DIR / lang
    if domain_dir.is_dir():
        paths.extend(sorted(domain_dir.glob("*.yaml")))
    return paths


def _current_fingerprint(paths: list[Path]) -> tuple[tuple[str, float], ...]:
    """Build an mtime fingerprint that changes when any file's mtime moves."""
    out: list[tuple[str, float]] = []
    for path in paths:
        try:
            out.append((str(path), path.stat().st_mtime))
        except OSError:
            out.append((str(path), -1.0))
    return tuple(out)


def _load(lang: str) -> dict[str, Any]:
    """Read every contributing YAML for ``lang`` and return the merged flat dict.

    Returns an empty dict when no files exist — better to fall back to the
    bare key than crash mid-request because someone is editing a YAML.
    """
    paths = _candidate_paths(lang)
    if not paths:
        return {}

    fingerprint = _current_fingerprint(paths)

    with _lock:
        if _fingerprint.get(lang) == fingerprint and lang in _cache:
            return _cache[lang]
        merged: dict[str, str] = {}
        for path in paths:
            try:
                with path.open("r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
            except (OSError, yaml.YAMLError) as exc:
                logger.warning("i18n: failed to load %s: %s", path, exc)
                continue
            merged.update(_flatten(data))
        _cache[lang] = merged
        _fingerprint[lang] = fingerprint
        return merged


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Convert nested YAML to dotted keys: ``{"ui": {"x": "y"}}`` -> ``{"ui.x": "y"}``.

    Strings are stored as-is, lists are preserved (consumed by
    :func:`t_list`). Scalars other than strings/lists are coerced via
    ``str()`` so numbers in YAML still address as text via :func:`t`.
    """
    out: dict[str, Any] = {}
    for key, value in data.items():
        full = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten(value, full))
        elif isinstance(value, list):
            out[full] = list(value)
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
    if isinstance(value, list):
        logger.warning("i18n: t() called on list key %r; use t_list() instead", key)
        return key
    if fmt:
        try:
            return value.format(**fmt)
        except (KeyError, IndexError, ValueError) as exc:
            logger.warning("i18n: format failed for %r: %s", key, exc)
            return value
    return value


def t_list(key: str, lang: str | None = None) -> list[str]:
    """Translate ``key`` to a list-valued entry.

    Lookup order matches :func:`t`: active locale -> English -> empty
    list. A scalar at the requested key is a type mismatch — log a
    warning and return an empty list so the caller doesn't crash.
    Lists are copied so callers can't mutate the cached dict.
    """
    active = resolve_lang(lang)
    value = _load(active).get(key)
    if value is None and active != "en":
        value = _load("en").get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        logger.warning("i18n: t_list() called on scalar key %r", key)
        return []
    return [str(item) for item in value]


def _reset_for_tests() -> None:
    """Test hook: clear cache so a fixture YAML edit is visible immediately."""
    with _lock:
        _cache.clear()
        _fingerprint.clear()
