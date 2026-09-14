"""Tests for ``skills/import-medical-record/scripts/validate_draft.py``."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).parent.parent.parent
    / "skills"
    / "import-medical-record"
    / "scripts"
    / "validate_draft.py"
)

FIXTURES = (
    Path(__file__).parent.parent / "ingest" / "records" / "fixtures" / "templates"
)


def _run(target: Path) -> tuple[int, dict]:
    res = subprocess.run(
        [sys.executable, str(SCRIPT), str(target)],
        capture_output=True,
        text=True,
    )
    return res.returncode, json.loads(res.stdout.strip().splitlines()[-1])


def test_valid_template_returns_ok():
    code, payload = _run(FIXTURES / "minimal_single_user")
    assert code == 0
    assert payload["ok"] is True
    assert "import_id" in payload
    assert payload["user_ids"] == ["test"]


def test_invalid_template_returns_error():
    code, payload = _run(FIXTURES / "invalid_extra_file")
    assert code == 1
    assert payload["ok"] is False
    assert "error" in payload


def test_missing_dir_returns_error(tmp_path):
    code, payload = _run(tmp_path / "no-such-dir")
    assert code == 1
    assert payload["ok"] is False


def test_no_args_returns_usage(tmp_path):
    res = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 2
    payload = json.loads(res.stdout.strip())
    assert payload["ok"] is False
    assert "usage" in payload["error"].lower()
