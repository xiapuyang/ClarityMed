"""Tests for ``EvalsConfig`` and ``configs/evals.yaml`` loading."""

from __future__ import annotations

import importlib

import pytest
import yaml
from pydantic import ValidationError

from claritymed.core.schemas import EvalsConfig


def _reload_config():
    from claritymed import config as _cfg

    return importlib.reload(_cfg)


# ---------- schema: happy paths ----------


def test_minimal_valid_config():
    cfg = EvalsConfig.model_validate(
        {
            "tasks": ["medqa"],
            "output_dir": "data/evals/results",
        }
    )
    assert cfg.tasks == ["medqa"]
    assert cfg.output_dir == "data/evals/results"
    assert cfg.judge_provider_id is None
    assert cfg.default_limit is None


def test_judge_provider_id_null_parses_as_none():
    cfg = EvalsConfig.model_validate(
        {
            "tasks": ["medqa"],
            "output_dir": "data/evals/results",
            "judge_provider_id": None,
        }
    )
    assert cfg.judge_provider_id is None


def test_default_limit_accepts_positive_integer():
    cfg = EvalsConfig.model_validate(
        {
            "tasks": ["medqa"],
            "output_dir": "data/evals/results",
            "default_limit": 50,
        }
    )
    assert cfg.default_limit == 50


def test_default_limit_accepts_zero():
    cfg = EvalsConfig.model_validate(
        {
            "tasks": ["medqa"],
            "output_dir": "data/evals/results",
            "default_limit": 0,
        }
    )
    assert cfg.default_limit == 0


# ---------- schema: error paths ----------


def test_missing_tasks_raises():
    with pytest.raises(ValidationError, match="tasks"):
        EvalsConfig.model_validate({"output_dir": "data/evals/results"})


def test_empty_tasks_list_raises():
    with pytest.raises(ValidationError, match="tasks"):
        EvalsConfig.model_validate({"tasks": [], "output_dir": "data/evals/results"})


def test_missing_output_dir_raises():
    with pytest.raises(ValidationError, match="output_dir"):
        EvalsConfig.model_validate({"tasks": ["medqa"]})


def test_negative_default_limit_raises():
    with pytest.raises(ValidationError, match="default_limit"):
        EvalsConfig.model_validate(
            {
                "tasks": ["medqa"],
                "output_dir": "data/evals/results",
                "default_limit": -1,
            }
        )


def test_extra_field_rejected():
    """``extra="forbid"`` catches typos before they silently no-op a run."""
    with pytest.raises(ValidationError):
        EvalsConfig.model_validate(
            {
                "tasks": ["medqa"],
                "output_dir": "data/evals/results",
                "ouput_directory": "typo",  # codespell:ignore
            }
        )


def test_frozen_instance():
    cfg = EvalsConfig.model_validate(
        {"tasks": ["medqa"], "output_dir": "data/evals/results"}
    )
    with pytest.raises(ValidationError):
        cfg.tasks = ["other"]  # type: ignore[misc]


# ---------- loader: happy + error ----------


def test_load_evals_config_from_repo():
    """Real ``configs/evals.yaml`` parses without surprises."""
    cfg = _reload_config()
    evals = cfg.load_evals_config()

    assert isinstance(evals, EvalsConfig)
    assert "medqa" in evals.tasks
    assert evals.output_dir == "data/evals/results"


def test_load_evals_config_missing_file_raises(tmp_path, monkeypatch):
    cfg = _reload_config()
    monkeypatch.setattr(cfg, "CONFIGS_DIR", tmp_path)
    cfg.reload_configs()
    with pytest.raises(FileNotFoundError):
        cfg.load_evals_config()


def test_load_evals_config_malformed_raises(tmp_path, monkeypatch):
    cfg = _reload_config()
    bad = tmp_path / "evals.yaml"
    bad.write_text(
        yaml.safe_dump({"tasks": ["medqa"]}),  # missing output_dir
        encoding="utf-8",
    )
    monkeypatch.setattr(cfg, "CONFIGS_DIR", tmp_path)
    cfg.reload_configs()
    with pytest.raises(ValidationError):
        cfg.load_evals_config()
