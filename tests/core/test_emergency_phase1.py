"""Phase 1 tests for the emergency triage gate.

Covers the schema extensions, off-mode safeguards, sensitivity
resolver, and end-to-end audit emission for the off-path turn.

The triage service itself (Phase 2-3 rule matcher + composer) is not
exercised here — Phase 1 ships a routine-only stub. These tests pin
the scaffolding so the Phase 2/3 changes don't drift the off-path
guarantees.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pytest
import yaml
from pydantic import ValidationError

from claritymed.core.emergency import (
    EmergencyAssessment,
    EmergencyTriage,
    resolve_sensitivity,
)
from claritymed.core.emergency.config import (
    EmergencyConfig,
    load_emergency_config,
)
from claritymed.core.emergency.schemas import MatchedRule
from claritymed.core.emergency.sensitivity import ResolvedSensitivity
from claritymed.core.i18n.loader import t as i18n_t
from claritymed.core.schemas import (
    Account,
    EmergencySettings,
)


# --- schema extensions -----------------------------------------------


def test_routine_noop_has_routine_level_and_no_findings():
    a = EmergencyAssessment.routine_noop()
    assert a.level == "routine"
    assert a.matched_rules == []
    assert a.suggested_action_i18n_key is None
    assert a.missing_qualifiers == []
    assert a.reasoning == ""


# --- Account validator ----------------------------------------------


def test_account_emergency_off_requires_acknowledgement():
    with pytest.raises(ValidationError) as excinfo:
        Account(
            user_id="test",
            display_name="Test",
            emergency=EmergencySettings(sensitivity="off"),
        )
    assert "off_acknowledged_at" in str(excinfo.value)


def test_account_emergency_off_with_ack_validates():
    ack = datetime.now(timezone.utc)
    acc = Account(
        user_id="test",
        display_name="Test",
        emergency=EmergencySettings(sensitivity="off", off_acknowledged_at=ack),
    )
    assert acc.emergency.sensitivity == "off"
    assert acc.emergency.off_acknowledged_at == ack


def test_account_emergency_default_is_none_sensitivity():
    acc = Account(user_id="test", display_name="Test")
    assert acc.emergency.sensitivity is None
    assert acc.emergency.off_acknowledged_at is None


def test_account_emergency_field_unknown_value_rejected():
    with pytest.raises(ValidationError):
        EmergencySettings(sensitivity="paranoid")  # type: ignore[arg-type]


# --- EmergencyConfig loader -----------------------------------------


def test_emergency_config_default_off_accepted():
    """``default_sensitivity: off`` is now valid — the loader used to
    reject it, but that safeguard was removed on operator request. The
    upstream ``CLARITYMED_FORCE_EMERGENCY_GATE`` env override (default
    ``on``) still upgrades a resolved ``off`` back to ``lenient``
    unless disabled explicitly."""
    cfg = EmergencyConfig(default_sensitivity="off")
    assert cfg.default_sensitivity == "off"


def test_emergency_config_default_balanced_ok():
    cfg = EmergencyConfig(default_sensitivity="balanced")
    assert cfg.default_sensitivity == "balanced"


def test_load_emergency_config_uses_repo_yaml():
    """The repo's ``configs/emergency.yaml`` must parse + validate."""
    cfg = load_emergency_config()
    assert cfg.default_sensitivity in {"strict", "balanced", "lenient", "off"}
    assert "off" in cfg.sensitivity_profiles
    off_profile = cfg.sensitivity_profiles["off"]
    assert off_profile.reply_footer_i18n_key == "emergency.footer.gate_disabled"


def test_load_emergency_config_accepts_yaml_with_default_off(tmp_path):
    """Round-trip: an operator-authored ``default_sensitivity: off``
    YAML is accepted end-to-end (no loader rejection)."""
    good = tmp_path / "emergency.yaml"
    good.write_text(
        yaml.safe_dump({"default_sensitivity": "off", "sensitivity_profiles": {}})
    )
    cfg = load_emergency_config(good)
    assert cfg.default_sensitivity == "off"


def test_emergency_config_coerces_yaml_off_bool_to_string():
    """Defense against the YAML 1.1 bareword footgun.

    Unquoted ``default_sensitivity: off`` in YAML parses as boolean
    ``False``, which the Literal validator would then reject with a
    confusing "Input should be 'strict'…" message. The field validator
    coerces the two YAML-bool spellings back to their string
    equivalents so the operator's intent survives."""
    cfg = EmergencyConfig.model_validate({"default_sensitivity": False})
    assert cfg.default_sensitivity == "off"


def test_emergency_config_unquoted_off_in_yaml_round_trips(tmp_path):
    """End-to-end: an operator who writes literal `default_sensitivity: off`
    without quotes in emergency.yaml still gets `off` at runtime."""
    bad_ish = tmp_path / "emergency.yaml"
    bad_ish.write_text("default_sensitivity: off\n")  # UNQUOTED — the footgun
    cfg = load_emergency_config(bad_ish)
    assert cfg.default_sensitivity == "off"


# --- Sensitivity resolver -------------------------------------------


def test_resolver_priority_cli_over_user(monkeypatch):
    monkeypatch.setenv("CLARITYMED_FORCE_EMERGENCY_GATE", "off")
    resolved = resolve_sensitivity(
        cli_override="strict",
        user_pref="lenient",
        app_default="balanced",
    )
    assert resolved.requested == "strict"
    assert resolved.effective == "strict"
    assert resolved.reason == "cli_override"
    assert resolved.env_override_applied is False


def test_resolver_priority_user_over_default(monkeypatch):
    monkeypatch.setenv("CLARITYMED_FORCE_EMERGENCY_GATE", "off")
    resolved = resolve_sensitivity(
        cli_override=None,
        user_pref="strict",
        app_default="balanced",
    )
    assert resolved.requested == "strict"
    assert resolved.reason == "user_preference"


def test_resolver_priority_app_default_fallback(monkeypatch):
    monkeypatch.setenv("CLARITYMED_FORCE_EMERGENCY_GATE", "off")
    resolved = resolve_sensitivity(
        cli_override=None,
        user_pref=None,
        app_default="balanced",
    )
    assert resolved.requested == "balanced"
    assert resolved.reason == "app_default"


def test_resolver_env_override_downgrades_off_to_lenient(monkeypatch):
    monkeypatch.setenv("CLARITYMED_FORCE_EMERGENCY_GATE", "on")
    resolved = resolve_sensitivity(
        cli_override=None,
        user_pref="off",
        app_default="balanced",
    )
    assert isinstance(resolved, ResolvedSensitivity)
    assert resolved.requested == "off"
    assert resolved.effective == "lenient"
    assert resolved.env_override_applied is True


def test_resolver_env_override_default_on_when_unset(monkeypatch):
    monkeypatch.delenv("CLARITYMED_FORCE_EMERGENCY_GATE", raising=False)
    resolved = resolve_sensitivity(
        cli_override="off",
        user_pref=None,
        app_default="balanced",
    )
    # Default is on → off downgrades.
    assert resolved.effective == "lenient"
    assert resolved.env_override_applied is True


def test_resolver_env_off_allows_user_off(monkeypatch):
    """Operator turned the override OFF — user 'off' is honored."""
    monkeypatch.setenv("CLARITYMED_FORCE_EMERGENCY_GATE", "off")
    resolved = resolve_sensitivity(
        cli_override=None,
        user_pref="off",
        app_default="balanced",
    )
    assert resolved.effective == "off"
    assert resolved.env_override_applied is False


# --- EmergencyTriage service ----------------------------------------


@pytest.mark.asyncio
async def test_triage_returns_routine_for_balanced():
    triage = EmergencyTriage()
    result = await triage.assess("我头疼", history=None, sensitivity="balanced")
    assert result.level == "routine"
    assert result.matched_rules == []


@pytest.mark.asyncio
async def test_triage_off_short_circuits_and_audits(
    caplog: pytest.LogCaptureFixture, monkeypatch
):
    """``sensitivity='off'`` must emit ``redflag.gate_disabled`` and
    return routine_noop. We don't have a real audit handler bound in
    unit tests, so we patch ``audit_event`` to capture the call.
    """
    from claritymed.core.emergency import service as service_module

    calls: list[tuple[str, dict]] = []

    def fake_audit(kind, payload=None):
        calls.append((kind, payload or {}))
        return None

    monkeypatch.setattr(service_module, "audit_event", fake_audit)
    triage = EmergencyTriage()
    result = await triage.assess("x", history=None, sensitivity="off")
    assert result.level == "routine"
    assert any(
        kind == "redflag.gate_disabled"
        and payload.get("requested") == "off"
        and payload.get("effective") == "off"
        for kind, payload in calls
    ), f"expected redflag.gate_disabled in {calls}"


@pytest.mark.asyncio
async def test_triage_off_audit_failure_does_not_propagate(
    monkeypatch, caplog: pytest.LogCaptureFixture
):
    """Gate downtime never denies the user. If the audit emit raises,
    the service still returns ``routine_noop``."""
    from claritymed.core.emergency import service as service_module

    def boom(*_a, **_kw):
        raise RuntimeError("audit broken")

    monkeypatch.setattr(service_module, "audit_event", boom)
    with caplog.at_level(logging.ERROR):
        result = await EmergencyTriage().assess("x", history=None, sensitivity="off")
    assert result.level == "routine"
    assert any("audit emit failed" in r.message for r in caplog.records)


# --- i18n footer wiring ---------------------------------------------


def test_i18n_footer_keys_present_in_en():
    assert i18n_t("emergency.footer.gate_disabled", lang="en") != (
        "emergency.footer.gate_disabled"
    )
    assert i18n_t("emergency.footer.strict_mode_active", lang="en") != (
        "emergency.footer.strict_mode_active"
    )


def test_i18n_footer_keys_present_in_zh():
    assert i18n_t("emergency.footer.gate_disabled", lang="zh") != (
        "emergency.footer.gate_disabled"
    )
    assert i18n_t("emergency.footer.strict_mode_active", lang="zh") != (
        "emergency.footer.strict_mode_active"
    )


# --- Matched-rule audit smoke test (via AskService finalize path) ---


def test_matched_rule_carries_required_fields():
    """``MatchedRule`` is the bridge between the rule engine (Phase 2)
    and the audit row; locking the shape here keeps the Phase 2 work
    honest."""
    m = MatchedRule(
        rule_id="acs_acute_coronary_syndrome",
        level="critical",
        suggested_action_i18n_key="emergency.action.call_ems_cardiac",
        citations=["AHA/ACC 2021 Chest Pain Guideline"],
        matched_qualifiers=["radiation_left_arm", "diaphoresis"],
    )
    assert m.rule_id == "acs_acute_coronary_syndrome"
    assert m.level == "critical"
