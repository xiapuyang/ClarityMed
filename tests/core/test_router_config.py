"""Tests for ``configs/router.yaml`` loading."""

from __future__ import annotations

import importlib

import pytest
import yaml
from pydantic import ValidationError

from claritymed.core.schemas.router import RouterConfig


def _reload_config():
    from claritymed import config as _cfg

    return importlib.reload(_cfg)


def test_load_router_config_returns_config():
    cfg = _reload_config()
    router = cfg.load_router_config()

    assert isinstance(router, RouterConfig)


def test_router_thresholds_and_fallback():
    cfg = _reload_config()
    router = cfg.load_router_config()

    assert router.high_threshold == 0.9
    assert router.low_threshold == 0.5
    assert router.llm_fallback.provider_id == "ollama"
    assert len(router.rules.imperative_verbs_ingest) > 0


def test_missing_required_field_raises(tmp_path, monkeypatch):
    """A malformed router.yaml fails fast at load time."""
    cfg = _reload_config()
    bad = tmp_path / "router.yaml"
    bad.write_text(
        yaml.safe_dump(
            {
                "rules": {},
                # missing llm_fallback
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cfg, "CONFIGS_DIR", tmp_path)
    cfg.reload_configs()
    with pytest.raises(ValidationError):
        cfg.load_router_config()


def test_missing_file_raises(tmp_path, monkeypatch):
    cfg = _reload_config()
    monkeypatch.setattr(cfg, "CONFIGS_DIR", tmp_path)
    cfg.reload_configs()
    with pytest.raises(FileNotFoundError):
        cfg.load_router_config()
