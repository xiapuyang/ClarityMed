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


def test_key_sets_must_match_in_real_yamls():
    """All shipped i18n YAMLs must have identical key sets.

    This is the bilingual-equivalence enforcement — if zh.yaml is missing a
    disclaimer key, the orchestrator would silently render the en string.
    """
    from claritymed.config import I18N_DIR as REAL_I18N_DIR

    def _flat_keys(path: Path) -> set[str]:
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

    en_keys = _flat_keys(REAL_I18N_DIR / "en.yaml")
    zh_keys = _flat_keys(REAL_I18N_DIR / "zh.yaml")
    assert en_keys == zh_keys, (
        f"en/zh key sets differ — only in en: {en_keys - zh_keys}; "
        f"only in zh: {zh_keys - en_keys}"
    )


def test_resolve_lang_prefers_explicit_over_ctx():
    token = language_ctx.set("zh")
    try:
        assert resolve_lang("en") == "en"
    finally:
        language_ctx.reset(token)
