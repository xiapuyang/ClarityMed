"""End-to-end smoke for ``claritymed audit scan`` / ``audit list-rules``.

Writes a tiny fake ``audit.log`` under a tmp LOG_DIR (via env var),
runs the command, and checks the report surfaces the expected
findings. Covers the typer wiring without depending on a real
production audit log.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from claritymed.cli.main import app


@pytest.fixture
def runner():
    return CliRunner()


def _write_log(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    # 10 mode.ask, 3 of them flagged as announced-but-skipped → 30% rate.
    for _ in range(10):
        lines.append(
            json.dumps(
                {
                    "kind": "mode.ask",
                    "payload": {"provider_id": "ollama", "model": "qwen3:14b"},
                    "user_id": "alice",
                    "created_at": "2026-06-09T10:00:00Z",
                }
            )
        )
    for i in range(3):
        lines.append(
            json.dumps(
                {
                    "kind": "mode.ask.tool_announced_but_skipped",
                    "payload": {
                        "provider_id": "ollama",
                        "model": "qwen3:14b",
                        "snippet": f"我将检索sample-{i}",
                        "tool": "retrieve_medical_literature",
                    },
                    "user_id": "alice",
                    "created_at": "2026-06-09T10:00:00Z",
                }
            )
        )
    (log_dir / "audit.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_audit_list_rules_prints_registered_rules(runner):
    result = runner.invoke(app, ["audit", "list-rules"])
    assert result.exit_code == 0
    assert "tool_announced_but_skipped" in result.stdout


def test_audit_scan_reports_findings(runner, tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    _write_log(log_dir)
    monkeypatch.setenv("CLARITYMED_LOG_DIR", str(log_dir))
    # Re-import config so LOG_DIR picks up the env override.
    import importlib

    from claritymed import config as cfg_mod

    importlib.reload(cfg_mod)

    result = runner.invoke(app, ["audit", "scan"])
    assert result.exit_code == 0, result.stdout
    assert "tool_announced_but_skipped" in result.stdout
    # 30% should trigger the "switch to deterministic mode" finding.
    assert "deterministic" in result.stdout


def test_audit_scan_json_output(runner, tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    _write_log(log_dir)
    monkeypatch.setenv("CLARITYMED_LOG_DIR", str(log_dir))
    import importlib

    from claritymed import config as cfg_mod

    importlib.reload(cfg_mod)

    result = runner.invoke(app, ["audit", "scan", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["events_consumed"] == 13
    rules = {r["name"]: r for r in payload["rules"]}
    assert "tool_announced_but_skipped" in rules
    assert rules["tool_announced_but_skipped"]["counts"]["ollama/qwen3:14b"] == 3


def test_audit_scan_unknown_rule_errors(runner, tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CLARITYMED_LOG_DIR", str(log_dir))
    import importlib

    from claritymed import config as cfg_mod

    importlib.reload(cfg_mod)

    result = runner.invoke(app, ["audit", "scan", "--rule", "does_not_exist"])
    assert result.exit_code == 2


def test_audit_scan_missing_log_dir_errors(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("CLARITYMED_LOG_DIR", str(tmp_path / "nope"))
    import importlib

    from claritymed import config as cfg_mod

    importlib.reload(cfg_mod)

    result = runner.invoke(app, ["audit", "scan"])
    assert result.exit_code == 1


def teardown_module(_module) -> None:
    """Restore real LOG_DIR after the env-mutating tests above."""
    import importlib

    from claritymed import config as cfg_mod

    # Drop the env var we set, then reload so the rest of the suite gets
    # the actual user-config path back.
    os.environ.pop("CLARITYMED_LOG_DIR", None)
    importlib.reload(cfg_mod)
