"""Tests for ``skills/import-medical-record/scripts/session_plan.py``."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SCRIPT = (
    Path(__file__).parent.parent.parent
    / "skills"
    / "import-medical-record"
    / "scripts"
    / "session_plan.py"
)


@pytest.fixture
def redirected_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CLARITYMED_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


def _run(args: list[str], data_dir: Path) -> tuple[int, dict]:
    env = os.environ.copy()
    env["CLARITYMED_DATA_DIR"] = str(data_dir)
    res = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(res.stdout.strip().splitlines()[-1])
    return res.returncode, payload


SRC = "abcdef1234567890"


def test_read_missing_plan_returns_exists_false(redirected_data_dir):
    code, payload = _run(["read", SRC], redirected_data_dir)
    assert code == 0
    assert payload == {"exists": False}


def test_init_creates_plan(redirected_data_dir):
    code, payload = _run(
        [
            "init",
            SRC,
            "--description",
            "test source",
            "--split",
            "by_user",
            "--users",
            "alice:patient,bob:patient",
            "--remaining-scopes",
            "alice,bob",
        ],
        redirected_data_dir,
    )
    assert code == 0, payload
    assert payload["ok"] is True
    plan_path = redirected_data_dir / "_skill_sessions" / SRC / "plan.yaml"
    assert plan_path.exists()
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    assert plan["split_strategy"] == "by_user"
    assert plan["users"] == {"alice": "patient", "bob": "patient"}
    assert plan["remaining_scopes"] == ["alice", "bob"]


def test_init_refuses_when_plan_exists_without_force(redirected_data_dir):
    _run(
        ["init", SRC, "--split", "by_user", "--remaining-scopes", "alice"],
        redirected_data_dir,
    )
    code, payload = _run(
        ["init", SRC, "--split", "by_user", "--remaining-scopes", "alice"],
        redirected_data_dir,
    )
    assert code == 1
    assert payload["error"] == "plan_exists"


def test_complete_moves_scope_to_sessions(redirected_data_dir):
    _run(
        [
            "init",
            SRC,
            "--split",
            "by_user",
            "--remaining-scopes",
            "alice,bob",
        ],
        redirected_data_dir,
    )
    code, payload = _run(
        [
            "complete",
            SRC,
            "--scope",
            "alice",
            "--import-id",
            "0123456789ab",
            "--case-count",
            "5",
            "--fact-count",
            "2",
        ],
        redirected_data_dir,
    )
    assert code == 0
    assert payload["remaining"] == ["bob"]

    code, plan = _run(["read", SRC], redirected_data_dir)
    assert plan["sessions"][0]["scope"] == "alice"
    assert plan["sessions"][0]["import_id"] == "0123456789ab"


def test_complete_on_unknown_scope_errors(redirected_data_dir):
    _run(
        ["init", SRC, "--split", "by_user", "--remaining-scopes", "alice"],
        redirected_data_dir,
    )
    code, payload = _run(
        [
            "complete",
            SRC,
            "--scope",
            "charlie",
            "--import-id",
            "0123456789ab",
            "--case-count",
            "1",
            "--fact-count",
            "0",
        ],
        redirected_data_dir,
    )
    assert code == 1
    assert payload["error"] == "scope_unknown"


def test_complete_on_already_complete_scope_errors(redirected_data_dir):
    _run(
        ["init", SRC, "--split", "by_user", "--remaining-scopes", "alice"],
        redirected_data_dir,
    )
    _run(
        [
            "complete",
            SRC,
            "--scope",
            "alice",
            "--import-id",
            "0123456789ab",
            "--case-count",
            "1",
            "--fact-count",
            "0",
        ],
        redirected_data_dir,
    )
    code, payload = _run(
        [
            "complete",
            SRC,
            "--scope",
            "alice",
            "--import-id",
            "ffffffffffff",
            "--case-count",
            "1",
            "--fact-count",
            "0",
        ],
        redirected_data_dir,
    )
    assert code == 1
    assert payload["error"] == "scope_already_complete"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_plan_file_mode_is_0600(redirected_data_dir):
    _run(
        ["init", SRC, "--split", "single_shot", "--remaining-scopes", "all"],
        redirected_data_dir,
    )
    plan_path = redirected_data_dir / "_skill_sessions" / SRC / "plan.yaml"
    mode = stat.S_IMODE(os.stat(plan_path).st_mode)
    assert mode == 0o600


def test_invalid_source_id_rejected(redirected_data_dir):
    code, _ = _run(["read", "../escape"], redirected_data_dir)
    assert code != 0
