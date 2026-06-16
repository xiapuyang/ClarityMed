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


def _write_blob_sentinel(
    users_root: Path,
    user_id: str,
    sha256: str,
    *,
    ocr_has_report: bool,
    text: str = "",
    modality: str = "ultrasound",
) -> None:
    blob_dir = users_root / user_id / "blobs" / sha256[:2] / sha256
    blob_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "done",
        "kind": "ocr",
        "ext": "png",
        "provider": "bench-seed",
        "chain_tried": ["bench-seed"],
        "reason": None,
        "chars": len(text),
        "original_filename": f"{user_id}-report.png",
        "modality": modality,
        "modality_confidence": 0.9,
        "is_medical": True,
        "ocr_has_report": ocr_has_report,
    }
    (blob_dir / "ocr.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    if text:
        (blob_dir / "ocr.md").write_text(text, encoding="utf-8")


def _reload_with_data_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CLARITYMED_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CLARITYMED_LOG_DIR", str(tmp_path / "logs"))
    import importlib

    from claritymed import config as cfg_mod

    importlib.reload(cfg_mod)
    # The audit module bound `_cfg` at import time — reload it too so the
    # subcommand reads the patched DATA_DIR / LOG_DIR.
    from claritymed.cli.commands import audit as audit_mod

    importlib.reload(audit_mod)
    from claritymed.cli import main as cli_main

    importlib.reload(cli_main)


def test_audit_ocr_overrides_filters_true(runner, tmp_path, monkeypatch):
    _reload_with_data_dir(tmp_path, monkeypatch)
    users_root = tmp_path / "data" / "users"
    # 1 override, 1 non-override → only the override is listed.
    _write_blob_sentinel(
        users_root,
        "alice",
        "a" * 64,
        ocr_has_report=True,
        text="FINDINGS: …",
    )
    _write_blob_sentinel(
        users_root,
        "alice",
        "b" * 64,
        ocr_has_report=False,
    )

    from claritymed.cli.main import app as fresh_app

    result = runner.invoke(fresh_app, ["audit", "ocr-overrides", "--json"])
    assert result.exit_code == 0, result.stdout
    rows = [json.loads(line) for line in result.stdout.strip().splitlines() if line]
    assert len(rows) == 1
    assert rows[0]["sha256"] == "a" * 64
    assert rows[0]["user_id"] == "alice"
    assert "ocr_text" not in rows[0]


def test_audit_ocr_overrides_with_text_inlines_md(runner, tmp_path, monkeypatch):
    _reload_with_data_dir(tmp_path, monkeypatch)
    users_root = tmp_path / "data" / "users"
    _write_blob_sentinel(
        users_root,
        "bob",
        "c" * 64,
        ocr_has_report=True,
        text="IMPRESSION: cyst",
    )

    from claritymed.cli.main import app as fresh_app

    result = runner.invoke(
        fresh_app, ["audit", "ocr-overrides", "--json", "--with-text"]
    )
    assert result.exit_code == 0, result.stdout
    row = json.loads(result.stdout.strip().splitlines()[0])
    assert row["ocr_text"] == "IMPRESSION: cyst"


def test_audit_ocr_overrides_empty_returns_nonzero(runner, tmp_path, monkeypatch):
    _reload_with_data_dir(tmp_path, monkeypatch)
    from claritymed.cli.main import app as fresh_app

    result = runner.invoke(fresh_app, ["audit", "ocr-overrides"])
    assert result.exit_code == 1


def teardown_module(_module) -> None:
    """Restore real LOG_DIR after the env-mutating tests above."""
    import importlib

    from claritymed import config as cfg_mod

    # Drop the env var we set, then reload so the rest of the suite gets
    # the actual user-config path back.
    os.environ.pop("CLARITYMED_LOG_DIR", None)
    os.environ.pop("CLARITYMED_DATA_DIR", None)
    importlib.reload(cfg_mod)
    from claritymed.cli.commands import audit as audit_mod

    importlib.reload(audit_mod)
