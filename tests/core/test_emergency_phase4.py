"""Phase 4 tests: output validator + post-stream audit tripwire.

Locks the R8 / R10 contracts:

* Validator rejects non-critical replies that omit the rule's action
  wording (1 retry budget, then audits + accepts).
* Tripwire scans the persisted final text and emits
  ``redflag.reply_missing_action`` when the validator passed but the
  reply still drifted.
* Both layers no-op for routine / critical / off / no-action-key cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from claritymed.core.emergency import EmergencyAssessment, MatchedRule
from claritymed.core.emergency.validators import (
    audit_reply_missing_action_if_needed,
    extract_action_fingerprint,
    make_triage_output_validator,
    reply_honors_triage,
)


# --- fingerprint extraction -----------------------------------------


def test_fingerprint_extracts_bolded_prefix_en():
    text = "**Call your local emergency number now (US: 911).** More."
    fp = extract_action_fingerprint(text)
    assert fp == "call your local em"


def test_fingerprint_extracts_bolded_prefix_zh():
    text = "**请立即拨打 120 急救电话——这可能是心肌梗死。**不要驾车。"
    fp = extract_action_fingerprint(text)
    # First 18 normalized chars of the bolded sentence.
    assert "请立即拨打" in fp
    assert "120" in fp


def test_fingerprint_empty_when_no_bold_block():
    assert extract_action_fingerprint("plain text no bold") == ""


def test_fingerprint_nfkc_normalizes_fullwidth_digits():
    """Fullwidth `１２０` collapses to ASCII `120` post-NFKC."""
    text = "**请立即拨打 １２０ 急救电话。**"
    fp = extract_action_fingerprint(text)
    assert "120" in fp


# --- reply_honors_triage -------------------------------------------


def _triage(level, action_key="emergency.action.call_ems_cardiac"):
    return EmergencyAssessment(
        level=level,
        matched_rules=[
            MatchedRule(
                rule_id="acs",
                level=level if level != "routine" else "routine",
                suggested_action_i18n_key=action_key,
                citations=["x"],
            )
        ],
        suggested_action_i18n_key=action_key,
    )


def test_honors_triage_passes_when_routine():
    triage = EmergencyAssessment.routine_noop()
    assert reply_honors_triage("anything", triage, language="en") is True


def test_honors_triage_passes_when_critical():
    """Critical never goes through the agent — the validator should
    pass it through if it ever sees one (defensive)."""
    triage = _triage("critical")
    assert reply_honors_triage("no action here", triage, language="en") is True


def test_honors_triage_passes_when_no_action_key():
    """A triage carrying matched_rules but no action key — nothing to enforce."""
    triage = EmergencyAssessment(
        level="urgent",
        matched_rules=[
            MatchedRule(
                rule_id="x",
                level="urgent",
                suggested_action_i18n_key="real.but.missing",
                citations=["c"],
            )
        ],
        suggested_action_i18n_key=None,
    )
    assert reply_honors_triage("free text", triage, language="en") is True


def test_honors_triage_passes_when_reply_contains_action():
    triage = _triage("urgent", "emergency.action.urgent_eval_chest_pain")
    reply = (
        "Chest pain needs evaluation today — please go to an urgent-care clinic. "
        "Meanwhile, note when it started."
    )
    assert reply_honors_triage(reply, triage, language="en") is True


def test_honors_triage_fails_when_reply_omits_action():
    triage = _triage("urgent", "emergency.action.urgent_eval_chest_pain")
    reply = "Try some rest and see how you feel tomorrow."
    assert reply_honors_triage(reply, triage, language="en") is False


def test_honors_triage_passes_when_action_key_missing_from_i18n():
    """Operator misconfig (action_key not in i18n) should pass-through."""
    triage = _triage("urgent", "emergency.action.this_key_does_not_exist")
    assert reply_honors_triage("anything", triage, language="en") is True


def test_honors_triage_zh_reply_matches_zh_action():
    triage = _triage("urgent", "emergency.action.urgent_eval_chest_pain")
    reply = "**胸痛需要今天就接受评估——请在数小时内前往急诊**…"
    assert reply_honors_triage(reply, triage, language="zh") is True


# --- output validator ----------------------------------------------


@dataclass
class _FakeDeps:
    triage: Any


@dataclass
class _FakeCtx:
    deps: Any


def test_validator_passes_when_no_triage():
    v = make_triage_output_validator(language="en")
    out = v(_FakeCtx(deps=_FakeDeps(triage=None)), "anything")
    assert out == "anything"


def test_validator_passes_when_routine():
    v = make_triage_output_validator(language="en")
    triage = EmergencyAssessment.routine_noop()
    out = v(_FakeCtx(deps=_FakeDeps(triage=triage)), "anything")
    assert out == "anything"


def test_validator_passes_when_reply_honors_urgent():
    v = make_triage_output_validator(language="en")
    triage = _triage("urgent", "emergency.action.urgent_eval_chest_pain")
    reply = "Chest pain needs evaluation today — go now."
    out = v(_FakeCtx(deps=_FakeDeps(triage=triage)), reply)
    assert out == reply


def test_validator_raises_model_retry_on_first_miss():
    from pydantic_ai.exceptions import ModelRetry

    v = make_triage_output_validator(language="en", retry_budget=1)
    triage = _triage("urgent", "emergency.action.urgent_eval_chest_pain")
    with pytest.raises(ModelRetry) as excinfo:
        v(_FakeCtx(deps=_FakeDeps(triage=triage)), "Try rest.")
    assert "Chest pain needs evaluation today" in str(excinfo.value)


def test_validator_passes_after_budget_exhausted(monkeypatch):
    from claritymed.core.emergency import validators as v_mod
    from pydantic_ai.exceptions import ModelRetry

    audit_calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        v_mod,
        "audit_event",
        lambda kind, payload=None: audit_calls.append((kind, payload or {})),
    )
    v = make_triage_output_validator(language="en", retry_budget=1)
    triage = _triage("urgent", "emergency.action.urgent_eval_chest_pain")
    ctx = _FakeCtx(deps=_FakeDeps(triage=triage))
    # 1st miss → ModelRetry.
    with pytest.raises(ModelRetry):
        v(ctx, "ignore the rule")
    # 2nd miss → audit + pass through (no exception).
    out = v(ctx, "still ignore")
    assert out == "still ignore"
    assert any(c[0] == "redflag.validator_unrecoverable" for c in audit_calls)


def test_validator_per_build_state_resets():
    """Two separate make_*() calls have independent attempt counters."""
    from pydantic_ai.exceptions import ModelRetry

    v1 = make_triage_output_validator(language="en", retry_budget=1)
    v2 = make_triage_output_validator(language="en", retry_budget=1)
    triage = _triage("urgent", "emergency.action.urgent_eval_chest_pain")
    ctx = _FakeCtx(deps=_FakeDeps(triage=triage))
    with pytest.raises(ModelRetry):
        v1(ctx, "x")
    # v2 should still get its retry — counter is separate.
    with pytest.raises(ModelRetry):
        v2(ctx, "y")


# --- post-stream tripwire ------------------------------------------


def test_tripwire_quiet_when_triage_none(monkeypatch):
    from claritymed.core.emergency import validators as v_mod

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        v_mod,
        "audit_event",
        lambda kind, payload=None: calls.append((kind, payload or {})),
    )
    audit_reply_missing_action_if_needed("text", None, language="en")
    assert calls == []


def test_tripwire_quiet_when_reply_honors_triage(monkeypatch):
    from claritymed.core.emergency import validators as v_mod

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        v_mod,
        "audit_event",
        lambda kind, payload=None: calls.append((kind, payload or {})),
    )
    triage = _triage("urgent", "emergency.action.urgent_eval_chest_pain")
    audit_reply_missing_action_if_needed(
        "Chest pain needs evaluation today — go now.",
        triage,
        language="en",
    )
    assert calls == []


def test_tripwire_fires_when_reply_omits_action(monkeypatch):
    from claritymed.core.emergency import validators as v_mod

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        v_mod,
        "audit_event",
        lambda kind, payload=None: calls.append((kind, payload or {})),
    )
    triage = _triage("urgent", "emergency.action.urgent_eval_chest_pain")
    audit_reply_missing_action_if_needed(
        "Just rest and drink water.",
        triage,
        language="en",
    )
    assert len(calls) == 1
    assert calls[0][0] == "redflag.reply_missing_action"
    assert calls[0][1]["level"] == "urgent"
    assert calls[0][1]["rule_ids"] == ["acs"]


def test_tripwire_quiet_when_routine(monkeypatch):
    from claritymed.core.emergency import validators as v_mod

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        v_mod,
        "audit_event",
        lambda kind, payload=None: calls.append((kind, payload or {})),
    )
    triage = EmergencyAssessment.routine_noop()
    audit_reply_missing_action_if_needed("whatever", triage, language="en")
    assert calls == []
