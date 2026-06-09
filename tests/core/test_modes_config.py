"""Tests for ``configs/modes.yaml`` loading and i18n integration."""

from __future__ import annotations

import importlib

import pytest
import yaml
from pydantic import ValidationError

from claritymed.context import language_ctx
from claritymed.core.i18n.loader import _reset_for_tests, t
from claritymed.core.schemas.modes import ModesConfig


def _reload_config():
    from claritymed import config as _cfg

    return importlib.reload(_cfg)


def test_load_modes_config_returns_three_modes():
    cfg = _reload_config()
    modes = cfg.load_modes_config()

    assert isinstance(modes, ModesConfig)
    assert set(modes.modes.keys()) == {"ingest", "ask", "rag"}


def test_mode_llm_inference_flags():
    cfg = _reload_config()
    modes = cfg.load_modes_config()

    assert modes.get("ingest").allow_llm_inference is False
    assert modes.get("ask").allow_llm_inference is True
    assert modes.get("rag").allow_llm_inference is False


def test_mode_has_required_fields():
    cfg = _reload_config()
    modes = cfg.load_modes_config()

    ask = modes.get("ask")
    assert ask.prompt_key == "ask"
    assert ask.audit_event_type == "mode.ask"
    assert "retrieve_medical_literature" in ask.tools


def test_router_thresholds_and_fallback():
    cfg = _reload_config()
    modes = cfg.load_modes_config()

    assert modes.router.high_threshold == 0.9
    assert modes.router.low_threshold == 0.5
    assert modes.router.llm_fallback.provider_id == "ollama"
    assert len(modes.router.rules.imperative_verbs_ingest) > 0


def test_missing_required_field_raises(tmp_path, monkeypatch):
    """A malformed modes.yaml fails fast at load time."""
    cfg = _reload_config()
    bad = tmp_path / "modes.yaml"
    bad.write_text(
        yaml.safe_dump(
            {
                "modes": {
                    "ingest": {
                        # missing audit_event_type, allow_llm_inference, tools
                        "prompt_key": "ingest",
                    },
                },
                "router": {
                    "rules": {},
                    "llm_fallback": {"provider_id": "x"},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cfg, "CONFIGS_DIR", tmp_path)
    cfg.reload_configs()
    with pytest.raises(ValidationError):
        cfg.load_modes_config()


def test_missing_file_raises(tmp_path, monkeypatch):
    cfg = _reload_config()
    monkeypatch.setattr(cfg, "CONFIGS_DIR", tmp_path)
    cfg.reload_configs()
    with pytest.raises(FileNotFoundError):
        cfg.load_modes_config()


def test_i18n_mode_labels_present_both_langs():
    _reload_config()
    _reset_for_tests()

    token = language_ctx.set("en")
    try:
        assert t("modes.ingest.label") == "My Records"
        assert t("modes.ask.label") == "Ask"
        assert t("modes.rag.label") == "My Library"
    finally:
        language_ctx.reset(token)

    token = language_ctx.set("zh")
    try:
        assert t("modes.ingest.label") == "我的档案"
        assert t("modes.ask.label") == "咨询"
        assert t("modes.rag.label") == "我的资料库"
    finally:
        language_ctx.reset(token)


def test_i18n_phi_redaction_markers_present():
    _reload_config()
    _reset_for_tests()

    token = language_ctx.set("en")
    try:
        assert t("phi.redaction_marker.phone") == "[REDACTED:PHONE]"
        assert t("phi.redaction_marker.email") == "[REDACTED:EMAIL]"
    finally:
        language_ctx.reset(token)

    token = language_ctx.set("zh")
    try:
        assert "脱敏" in t("phi.redaction_marker.phone")
    finally:
        language_ctx.reset(token)
