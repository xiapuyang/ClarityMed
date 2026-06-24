"""Tests for ``claritymed.web.admin.yaml_io.save_yaml``."""

from __future__ import annotations


import pytest
import yaml

from claritymed import config as _cfg
from claritymed.web.admin import yaml_io


def test_save_yaml_rejects_unknown_name():
    """A name not in EDITABLE_CONFIGS / EDITABLE_MODEL_CATALOGS → ValueError."""
    with pytest.raises(ValueError, match="not in EDITABLE_CONFIGS"):
        yaml_io.save_yaml("not_in_allowlist.yaml", {"k": "v"})


def test_save_yaml_rejects_traversal_payload():
    """A payload like '../etc/passwd' must be refused before path resolve."""
    with pytest.raises(ValueError, match="not in EDITABLE_CONFIGS"):
        yaml_io.save_yaml("../etc/passwd", {})


def test_save_yaml_writes_app_yaml_atomically(tmp_configs_dir):
    """A valid name produces an atomic rewrite under CONFIGS_DIR/<name>."""
    data = {
        "i18n": {"default_lang": "en"},
        "tracing": {"enabled": False},
    }
    target = yaml_io.save_yaml("app.yaml", data)
    assert target == _cfg.CONFIGS_DIR / "app.yaml"
    assert target.exists()
    written = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert written == data


def test_save_yaml_invalidates_load_yaml_cache(tmp_configs_dir):
    """The lru_cache on load_yaml must be cleared so the next read is fresh."""
    yaml_io.save_yaml("app.yaml", {"i18n": {"default_lang": "en"}})
    first = _cfg.load_yaml("app.yaml")
    yaml_io.save_yaml("app.yaml", {"i18n": {"default_lang": "zh"}})
    second = _cfg.load_yaml("app.yaml")
    assert first != second
    assert second["i18n"]["default_lang"] == "zh"


@pytest.fixture
def tmp_configs_dir(tmp_path, monkeypatch):
    """Redirect CONFIGS_DIR to a tmp dir so the test does not stomp on the
    real repo configs/ tree.
    """
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    # Seed an app.yaml so save_yaml's atomic replace has a real
    # parent dir to write into.
    (cfg_dir / "app.yaml").write_text("i18n:\n  default_lang: en\n", encoding="utf-8")
    monkeypatch.setattr(_cfg, "CONFIGS_DIR", cfg_dir)
    _cfg.load_yaml.cache_clear()
    yield cfg_dir
    _cfg.load_yaml.cache_clear()
