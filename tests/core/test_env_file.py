"""Tests for ``config.load_env_file`` — runtime .env loader."""

from __future__ import annotations

import importlib
import os
from pathlib import Path


def _reload_config():
    from claritymed import config as _cfg

    return importlib.reload(_cfg)


def test_loads_simple_kv(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLARITYMED_HOME", str(tmp_path))
    monkeypatch.delenv("OMLX_API_KEY", raising=False)
    cfg = _reload_config()
    (tmp_path / ".env").write_text("OMLX_API_KEY=Abcd1234\n", encoding="utf-8")
    applied = cfg.load_env_file()
    assert applied["OMLX_API_KEY"] == "Abcd1234"
    assert os.environ["OMLX_API_KEY"] == "Abcd1234"


def test_skips_comments_and_blanks(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLARITYMED_HOME", str(tmp_path))
    monkeypatch.delenv("FOO", raising=False)
    monkeypatch.delenv("BAR", raising=False)
    cfg = _reload_config()
    (tmp_path / ".env").write_text(
        "# comment\n\nFOO=bar\n   # indented comment\nBAR=baz\n",
        encoding="utf-8",
    )
    applied = cfg.load_env_file()
    assert applied == {"FOO": "bar", "BAR": "baz"}


def test_does_not_override_existing_env(tmp_path: Path, monkeypatch):
    """Existing env vars (e.g. shell-exported) win over the file."""
    monkeypatch.setenv("CLARITYMED_HOME", str(tmp_path))
    monkeypatch.setenv("ALREADY_SET", "from_shell")
    cfg = _reload_config()
    (tmp_path / ".env").write_text("ALREADY_SET=from_file\n", encoding="utf-8")
    applied = cfg.load_env_file()
    assert "ALREADY_SET" not in applied
    assert os.environ["ALREADY_SET"] == "from_shell"


def test_strips_quotes(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLARITYMED_HOME", str(tmp_path))
    monkeypatch.delenv("QUOTED", raising=False)
    cfg = _reload_config()
    (tmp_path / ".env").write_text('QUOTED="has spaces"\n', encoding="utf-8")
    applied = cfg.load_env_file()
    assert applied["QUOTED"] == "has spaces"


def test_missing_file_returns_empty(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLARITYMED_HOME", str(tmp_path))
    cfg = _reload_config()
    assert cfg.load_env_file() == {}


def test_strips_leading_export(tmp_path: Path, monkeypatch):
    """Lines copy-pasted from a shell-sourceable .env keep ``export `` —
    the loader must strip it so the key isn't stored as ``"export NAME"``."""
    monkeypatch.setenv("CLARITYMED_HOME", str(tmp_path))
    monkeypatch.delenv("CLARITYMED_ALLOW_MINERU", raising=False)
    monkeypatch.delenv("EXPORTED_QUOTED", raising=False)
    cfg = _reload_config()
    (tmp_path / ".env").write_text(
        'export CLARITYMED_ALLOW_MINERU=1\nexport EXPORTED_QUOTED="has spaces"\n',
        encoding="utf-8",
    )
    applied = cfg.load_env_file()
    assert applied["CLARITYMED_ALLOW_MINERU"] == "1"
    assert applied["EXPORTED_QUOTED"] == "has spaces"
    assert os.environ["CLARITYMED_ALLOW_MINERU"] == "1"
    assert "export CLARITYMED_ALLOW_MINERU" not in os.environ
