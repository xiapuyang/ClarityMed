"""``--emergency-sensitivity`` CLI flag accepts {strict, balanced, lenient} only.

``off`` is intentionally rejected at the parser level. Rationale lives in
``cli/common.py::CLIEmergencySensitivity``: the CLI flag bypasses the
``settings.yaml`` ``off_acknowledged_at`` two-step safeguard, so the only
operator paths to disable the gate are the master env switch
(``CLARITYMED_FORCE_EMERGENCY_GATE=off``, which downgrades to ``lenient``)
or a hand-edit of ``data/users/<uid>/settings.yaml`` with the ack
timestamp present.

These tests assert the parser-level rejection only; they do not exercise
the runtime gate. If a future refactor exposes ``off`` through CLI, this
file is the regression net.
"""

from __future__ import annotations

from typer.testing import CliRunner

from claritymed.cli.main import app

runner = CliRunner()


def test_ask_rejects_emergency_sensitivity_off():
    result = runner.invoke(
        app,
        ["ask", "--emergency-sensitivity", "off", "anything"],
    )
    assert result.exit_code != 0
    # typer/click's standard "invalid choice" error mentions the value.
    combined = (result.stdout + (result.stderr or "")).lower()
    assert "off" in combined
    assert "emergency-sensitivity" in combined or "invalid" in combined


def test_tui_rejects_emergency_sensitivity_off():
    result = runner.invoke(
        app,
        ["tui", "--emergency-sensitivity", "off"],
    )
    assert result.exit_code != 0
    combined = (result.stdout + (result.stderr or "")).lower()
    assert "off" in combined
    assert "emergency-sensitivity" in combined or "invalid" in combined


def test_ask_rejects_emergency_sensitivity_typo():
    """Any value outside the Enum fails; not specific to ``off``."""
    result = runner.invoke(
        app,
        ["ask", "--emergency-sensitivity", "balanced_typo", "anything"],
    )
    assert result.exit_code != 0


def test_ask_help_lists_three_choices_and_documents_off_exclusion():
    """``--help`` should advertise the restriction so operators don't guess."""
    result = runner.invoke(app, ["ask", "--help"])
    assert result.exit_code == 0
    out = result.stdout
    # Enum choices appear in help output as typer's "[strict|balanced|lenient]".
    assert "strict" in out
    assert "balanced" in out
    assert "lenient" in out
    # The help text explicitly explains why 'off' is excluded and where
    # operators should go instead.
    assert "CLARITYMED_FORCE_EMERGENCY_GATE" in out or "settings.yaml" in out


def test_tui_help_lists_three_choices_and_documents_off_exclusion():
    result = runner.invoke(app, ["tui", "--help"])
    assert result.exit_code == 0
    out = result.stdout
    assert "strict" in out
    assert "balanced" in out
    assert "lenient" in out
    assert "CLARITYMED_FORCE_EMERGENCY_GATE" in out or "settings.yaml" in out
