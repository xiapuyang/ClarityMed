"""Tests for ``claritymed.core.i18n.loader``."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pytest
import yaml

from claritymed.context import language_ctx
from claritymed.core.i18n import loader as i18n_loader
from claritymed.core.i18n.loader import _reset_for_tests, resolve_lang, t


@pytest.fixture
def i18n_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the loader at a fresh per-test YAML dir."""
    d = tmp_path / "i18n"
    d.mkdir()
    monkeypatch.setattr(i18n_loader, "I18N_DIR", d)
    _reset_for_tests()
    return d


def _write(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_happy_lookup(i18n_dir):
    _write(i18n_dir / "en.yaml", {"ui": {"greeting": "Hi"}})
    assert t("ui.greeting", lang="en") == "Hi"


def test_format_kwargs(i18n_dir):
    _write(i18n_dir / "en.yaml", {"ui": {"welcome": "Welcome, {name}"}})
    assert t("ui.welcome", lang="en", name="Alice") == "Welcome, Alice"


def test_active_lang_from_context_var(i18n_dir):
    _write(i18n_dir / "en.yaml", {"ui": {"hi": "hi"}})
    _write(i18n_dir / "zh.yaml", {"ui": {"hi": "你好"}})
    token = language_ctx.set("zh")
    try:
        assert t("ui.hi") == "你好"
    finally:
        language_ctx.reset(token)


def test_zh_missing_key_falls_back_to_en(i18n_dir):
    _write(i18n_dir / "en.yaml", {"ui": {"only_en": "english"}})
    _write(i18n_dir / "zh.yaml", {"ui": {"unrelated": "..."}})
    assert t("ui.only_en", lang="zh") == "english"


def test_unknown_key_returns_key_itself(i18n_dir):
    _write(i18n_dir / "en.yaml", {"ui": {"greeting": "Hi"}})
    assert t("ui.unknown_key", lang="en") == "ui.unknown_key"


def test_bad_format_template_returns_unformatted(i18n_dir, caplog):
    _write(i18n_dir / "en.yaml", {"ui": {"x": "Hi, {missing}"}})
    with caplog.at_level(logging.WARNING):
        result = t("ui.x", lang="en", name="Alice")
    assert result == "Hi, {missing}"
    assert any("format failed" in rec.message for rec in caplog.records)


def test_missing_yaml_file_returns_key(i18n_dir):
    # No zh.yaml created.
    assert t("ui.anything", lang="zh") == "ui.anything"


def test_mtime_hot_reload(i18n_dir):
    _write(i18n_dir / "en.yaml", {"ui": {"x": "first"}})
    assert t("ui.x", lang="en") == "first"
    _write(i18n_dir / "en.yaml", {"ui": {"x": "second"}})
    # Force a different mtime even if filesystem resolution is coarse.
    new_time = time.time() + 5
    os.utime(i18n_dir / "en.yaml", (new_time, new_time))
    assert t("ui.x", lang="en") == "second"


def test_unsafe_yaml_tag_rejected(i18n_dir, caplog):
    (i18n_dir / "en.yaml").write_text(
        "ui:\n  x: !!python/object/apply:os.system ['echo hi']\n"
    )
    with caplog.at_level(logging.WARNING):
        result = t("ui.x", lang="en")
    # safe_load rejected the tag; loader logged the failure and returned the
    # key string (no cached value to fall back to).
    assert result == "ui.x"
    assert any("failed to load" in rec.message for rec in caplog.records)


def _flat_keys_from_path(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    out: set[str] = set()

    def walk(node, prefix=""):
        for k, v in node.items():
            full = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                walk(v, full)
            else:
                out.add(full)

    walk(data)
    return out


def test_key_sets_must_match_in_real_yamls():
    """The shipped global i18n YAMLs must have identical key sets.

    Bilingual-equivalence enforcement — if zh.yaml is missing a disclaimer
    key, the orchestrator would silently render the en string.
    """
    from claritymed.config import I18N_DIR as REAL_I18N_DIR

    en_keys = _flat_keys_from_path(REAL_I18N_DIR / "en.yaml")
    zh_keys = _flat_keys_from_path(REAL_I18N_DIR / "zh.yaml")
    assert en_keys == zh_keys, (
        f"en/zh key sets differ — only in en: {en_keys - zh_keys}; "
        f"only in zh: {zh_keys - en_keys}"
    )


def test_per_domain_yaml_key_sets_must_match():
    """Per-domain i18n files under configs/i18n/<lang>/ must also pair up.

    Every file under ``en/`` must have a matching ``zh/`` sibling with
    the same flat key set, so a domain (e.g. ``symptoms_ddxplus``) can
    never silently fall back to English just because the zh file is
    incomplete.
    """
    from claritymed.config import I18N_DIR as REAL_I18N_DIR

    en_dir = REAL_I18N_DIR / "en"
    zh_dir = REAL_I18N_DIR / "zh"
    if not en_dir.is_dir() and not zh_dir.is_dir():
        pytest.skip("no per-domain i18n files shipped yet")
    en_files = {p.name for p in en_dir.glob("*.yaml")} if en_dir.is_dir() else set()
    zh_files = {p.name for p in zh_dir.glob("*.yaml")} if zh_dir.is_dir() else set()
    assert en_files == zh_files, (
        f"per-domain file sets differ — only in en/: {en_files - zh_files}; "
        f"only in zh/: {zh_files - en_files}"
    )
    for name in en_files:
        en_keys = _flat_keys_from_path(en_dir / name)
        zh_keys = _flat_keys_from_path(zh_dir / name)
        assert en_keys == zh_keys, (
            f"{name}: en/zh key sets differ — only in en: "
            f"{en_keys - zh_keys}; only in zh: {zh_keys - en_keys}"
        )


def test_per_domain_yaml_merged_into_lang_dict(i18n_dir):
    """A file in ``configs/i18n/<lang>/`` adds its keys to ``t()``'s namespace."""
    _write(i18n_dir / "en.yaml", {"ui": {"x": "global"}})
    (i18n_dir / "en").mkdir()
    _write(i18n_dir / "en" / "symptoms.yaml", {"symptoms": {"q": "domain"}})
    assert t("ui.x", lang="en") == "global"
    assert t("symptoms.q", lang="en") == "domain"


def test_per_domain_file_overrides_global_on_collision(i18n_dir):
    """When the same key appears in both, the per-domain file wins.

    Useful when a feature wants to rebrand a global label for its
    surface without touching the global YAML.
    """
    _write(i18n_dir / "en.yaml", {"label": "original"})
    (i18n_dir / "en").mkdir()
    _write(i18n_dir / "en" / "override.yaml", {"label": "rebranded"})
    assert t("label", lang="en") == "rebranded"


def test_multiple_per_domain_files_merge_deterministically(i18n_dir):
    """Files load in sorted-by-name order; the later file wins on collision."""
    _write(i18n_dir / "en.yaml", {"ui": {"x": "global"}})
    (i18n_dir / "en").mkdir()
    _write(i18n_dir / "en" / "a_first.yaml", {"shared": "from_a"})
    _write(i18n_dir / "en" / "b_second.yaml", {"shared": "from_b"})
    assert t("shared", lang="en") == "from_b"


def test_new_per_domain_file_invalidates_cache(i18n_dir):
    """Adding a per-domain file after a lookup should be picked up."""
    _write(i18n_dir / "en.yaml", {"ui": {"x": "g"}})
    assert t("ui.x", lang="en") == "g"
    (i18n_dir / "en").mkdir()
    _write(i18n_dir / "en" / "added.yaml", {"new_key": "added"})
    # Force a different mtime so the fingerprint changes deterministically.
    new_time = time.time() + 5
    os.utime(i18n_dir / "en", (new_time, new_time))
    os.utime(i18n_dir / "en" / "added.yaml", (new_time, new_time))
    assert t("new_key", lang="en") == "added"
    assert t("ui.x", lang="en") == "g"


def test_per_domain_dir_without_global_yaml_still_works(i18n_dir):
    """The base ``<lang>.yaml`` can be absent if only per-domain files exist."""
    (i18n_dir / "en").mkdir()
    _write(i18n_dir / "en" / "only.yaml", {"only_key": "value"})
    assert t("only_key", lang="en") == "value"


def test_resolve_lang_prefers_explicit_over_ctx():
    token = language_ctx.set("zh")
    try:
        assert resolve_lang("en") == "en"
    finally:
        language_ctx.reset(token)
