"""Tests for ``claritymed.web.admin.secrets``.

The read-side helper (``load_env_file``) lives in
``claritymed.config`` and already has coverage elsewhere; this file
covers the admin-specific write + masked-view helpers.
"""

from __future__ import annotations

import os
import stat

import pytest

from claritymed.web.admin import secrets
from claritymed.web.admin.secrets_manifest import EXPECTED_SECRETS


def test_atomic_write_env_writes_only_known_keys(tmp_claritymed_home):
    secrets.atomic_write_env({"OPENAI_API_KEY": "sk-test"})
    path = secrets.env_file_path()
    assert path.exists()
    body = path.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=sk-test" in body


def test_atomic_write_env_preserves_unrelated_keys(tmp_claritymed_home):
    secrets.atomic_write_env({"OPENAI_API_KEY": "sk-a", "ANTHROPIC_API_KEY": "sk-b"})
    secrets.atomic_write_env({"OPENAI_API_KEY": "sk-c"})
    body = secrets.env_file_path().read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=sk-c" in body
    assert "ANTHROPIC_API_KEY=sk-b" in body


def test_atomic_write_env_rejects_unknown_key(tmp_claritymed_home):
    with pytest.raises(ValueError, match="Unknown secret key"):
        secrets.atomic_write_env({"NOT_IN_MANIFEST": "x"})
    # No file is created when the call is rejected.
    assert not secrets.env_file_path().exists()


def test_atomic_write_env_sets_0o600(tmp_claritymed_home):
    secrets.atomic_write_env({"OPENAI_API_KEY": "sk-x"})
    mode = stat.S_IMODE(os.stat(secrets.env_file_path()).st_mode)
    assert mode == 0o600


def test_masked_view_reports_missing_when_absent(tmp_claritymed_home, monkeypatch):
    for spec in EXPECTED_SECRETS.values():
        monkeypatch.delenv(spec.key, raising=False)
    rows = {row.key: row for row in secrets.masked_view()}
    openai = rows["OPENAI_API_KEY"]
    assert openai.is_set is False
    assert openai.source == "missing"


def test_masked_view_reports_file_when_only_file(tmp_claritymed_home, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    secrets.atomic_write_env({"OPENAI_API_KEY": "sk-on-disk"})
    rows = {row.key: row for row in secrets.masked_view()}
    openai = rows["OPENAI_API_KEY"]
    assert openai.is_set is True
    assert openai.source == "file"


def test_masked_view_reports_env_when_shell_env_present(
    tmp_claritymed_home, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-shell")
    rows = {row.key: row for row in secrets.masked_view()}
    openai = rows["OPENAI_API_KEY"]
    assert openai.is_set is True
    assert openai.source == "env"


@pytest.fixture
def tmp_claritymed_home(tmp_path, monkeypatch):
    """Rebind CLARITYMED_HOME so the env file lives in the test tmp dir."""
    home = tmp_path / "claritymed_home"
    home.mkdir()
    monkeypatch.setenv("CLARITYMED_HOME", str(home))
    # Reload config so the module-level CLARITYMED_HOME constant picks up
    # the new env var.
    import importlib

    from claritymed import config as _cfg

    importlib.reload(_cfg)
    yield home
    importlib.reload(_cfg)
