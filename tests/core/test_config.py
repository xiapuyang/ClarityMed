"""Tests for ``claritymed.config``: env layering, YAML loading, runtime dirs."""

from __future__ import annotations

import importlib

import pytest
import yaml


def _reload_config():
    """Reload after env mutation. Returns the fresh module."""
    from claritymed import config as _cfg

    return importlib.reload(_cfg)


def test_default_lang_reads_app_yaml(monkeypatch):
    cfg = _reload_config()
    assert cfg.default_lang() == "en"


def test_load_yaml_missing_returns_empty(monkeypatch):
    cfg = _reload_config()
    assert cfg.load_yaml("does_not_exist.yaml") == {}


def test_load_yaml_safe_load_rejects_python_object(tmp_path, monkeypatch):
    cfg = _reload_config()
    fake = cfg.CONFIGS_DIR / "fake_unsafe.yaml"
    try:
        fake.write_text("!!python/object/apply:os.system ['echo hi']\n")
        with pytest.raises(yaml.YAMLError):
            cfg.reload_configs()
            cfg.load_yaml("fake_unsafe.yaml")
    finally:
        if fake.exists():
            fake.unlink()


def test_reload_clears_cache(monkeypatch, tmp_path):
    cfg = _reload_config()
    fake = cfg.CONFIGS_DIR / "tmp_reload.yaml"
    try:
        fake.write_text("value: first\n")
        assert cfg.load_yaml("tmp_reload.yaml") == {"value": "first"}
        fake.write_text("value: second\n")
        # No reload yet -> stale cache.
        assert cfg.load_yaml("tmp_reload.yaml") == {"value": "first"}
        cfg.reload_configs()
        assert cfg.load_yaml("tmp_reload.yaml") == {"value": "second"}
    finally:
        if fake.exists():
            fake.unlink()


def test_home_env_drives_all_three_dirs(monkeypatch, tmp_path):
    monkeypatch.setenv("CLARITYMED_HOME", str(tmp_path / "home"))
    cfg = _reload_config()
    assert cfg.DATA_DIR == tmp_path / "home" / "data"
    assert cfg.SHARED_DIR == tmp_path / "home" / "shared"
    assert cfg.LOG_DIR == tmp_path / "home" / "logs"


def test_specific_env_overrides_home(monkeypatch, tmp_path):
    monkeypatch.setenv("CLARITYMED_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CLARITYMED_DATA_DIR", str(tmp_path / "elsewhere"))
    cfg = _reload_config()
    assert cfg.DATA_DIR == tmp_path / "elsewhere"
    # SHARED_DIR / LOG_DIR still fall back to HOME children.
    assert cfg.SHARED_DIR == tmp_path / "home" / "shared"
    assert cfg.LOG_DIR == tmp_path / "home" / "logs"


def test_data_and_shared_dirs_do_not_overlap(monkeypatch, tmp_path):
    cfg = _reload_config()
    data = cfg.DATA_DIR.resolve()
    shared = cfg.SHARED_DIR.resolve()
    assert not data.is_relative_to(shared)
    assert not shared.is_relative_to(data)


def test_ensure_runtime_dirs_creates_top_three(monkeypatch, tmp_path):
    cfg = _reload_config()
    assert not cfg.DATA_DIR.exists()
    cfg.ensure_runtime_dirs()
    assert cfg.DATA_DIR.is_dir()
    assert cfg.SHARED_DIR.is_dir()
    assert cfg.LOG_DIR.is_dir()
    # Subdirs are not created here.
    assert not (cfg.DATA_DIR / "users").exists()
    assert not (cfg.SHARED_DIR / "qdrant").exists()


def test_ensure_runtime_dirs_idempotent(monkeypatch, tmp_path):
    cfg = _reload_config()
    cfg.ensure_runtime_dirs()
    cfg.ensure_runtime_dirs()  # second call must not raise
    assert cfg.DATA_DIR.is_dir()
