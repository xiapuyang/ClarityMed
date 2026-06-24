"""Unit tests for ``evals.emergency.runner``.

Focus: the runner correctly routes structured vs. text cases, skips
text cases when no extractor is wired, and propagates rule-engine
output into :class:`Prediction`. The gate itself is tested in
``tests/core/test_emergency_phase*``; this file pins the eval-harness
contract on top of it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from textwrap import dedent

import pytest

from claritymed.core.emergency import EmergencyTriage
from claritymed.core.emergency.config import EmergencyConfig, SensitivityProfile
from claritymed.core.emergency.rules import Rule, RuleTriggers

from evals.emergency.runner import load_cases, run_profile


def _write_case_file(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "cases.yaml"
    p.write_text(dedent(body), encoding="utf-8")
    return p


def _acs_rule_pack() -> tuple[list[Rule], EmergencyConfig]:
    """Minimal rule pack matching the ACS test cases below."""
    rule = Rule(
        id="acs_acute_coronary_syndrome",
        triggers=RuleTriggers(
            primary="chest_pain",
            any_of=["radiation_left_arm", "diaphoresis"],
            min_qualifier_matches=1,
            age_min=35,
        ),
        level="critical",
        action_key="emergency.action.call_ems_cardiac",
        citations=["AHA/ACC 2021"],
    )
    cfg = EmergencyConfig(
        default_sensitivity="balanced",
        sensitivity_profiles={
            "strict": SensitivityProfile(
                rule_overrides={}, ambiguous_rules_enabled=True
            ),
            "balanced": SensitivityProfile(
                rule_overrides={}, ambiguous_rules_enabled=True
            ),
            "lenient": SensitivityProfile(
                rule_overrides={}, ambiguous_rules_enabled=False
            ),
            "off": SensitivityProfile(rule_overrides={}, ambiguous_rules_enabled=False),
        },
    )
    return [rule], cfg


def test_load_cases_validates_yaml(tmp_path: Path):
    path = _write_case_file(
        tmp_path,
        """
        cases:
          - id: t1
            source: public_vignettes
            language: en
            symptoms:
              primary_complaint: chest_pain
              qualifiers: [radiation_left_arm, diaphoresis]
              age: 55
            ground_truth_level: critical
            ground_truth_rule_id: acs_acute_coronary_syndrome
            citation: "test"
        """,
    )
    cases = load_cases([path])
    assert len(cases) == 1
    assert cases[0].symptoms is not None
    assert cases[0].symptoms.primary_complaint == "chest_pain"


def test_load_cases_rejects_case_with_neither_input(tmp_path: Path):
    path = _write_case_file(
        tmp_path,
        """
        cases:
          - id: bad
            source: public_vignettes
            language: en
            ground_truth_level: routine
        """,
    )
    with pytest.raises(ValueError, match="exactly one"):
        load_cases([path])


def test_load_cases_rejects_case_with_both_inputs(tmp_path: Path):
    path = _write_case_file(
        tmp_path,
        """
        cases:
          - id: bad
            source: public_vignettes
            language: en
            symptoms: { primary_complaint: chest_pain }
            turns:
              - { role: user, text: hi }
            ground_truth_level: routine
        """,
    )
    with pytest.raises(ValueError, match="exactly one"):
        load_cases([path])


def test_run_profile_structured_case_fires_critical(tmp_path: Path):
    rules, cfg = _acs_rule_pack()
    triage = EmergencyTriage(rules=rules, config=cfg)
    path = _write_case_file(
        tmp_path,
        """
        cases:
          - id: acs_textbook
            source: public_vignettes
            language: en
            symptoms:
              primary_complaint: chest_pain
              qualifiers: [radiation_left_arm, diaphoresis]
              age: 55
            ground_truth_level: critical
            ground_truth_rule_id: acs_acute_coronary_syndrome
        """,
    )
    cases = load_cases([path])
    preds = asyncio.run(
        run_profile(cases, triage, sensitivity="balanced", has_extractor=False)
    )
    assert len(preds) == 1
    assert preds[0].predicted_level == "critical"
    assert "acs_acute_coronary_syndrome" in preds[0].matched_rule_ids
    assert preds[0].skipped is False


def test_run_profile_text_case_skipped_without_extractor(tmp_path: Path):
    rules, cfg = _acs_rule_pack()
    triage = EmergencyTriage(rules=rules, config=cfg)
    path = _write_case_file(
        tmp_path,
        """
        cases:
          - id: text_only
            source: public_vignettes
            language: en
            turns:
              - { role: user, text: "I have crushing chest pain." }
            ground_truth_level: critical
        """,
    )
    cases = load_cases([path])
    preds = asyncio.run(
        run_profile(cases, triage, sensitivity="balanced", has_extractor=False)
    )
    assert preds[0].skipped is True
    assert "extractor" in preds[0].skip_reason


def test_run_profile_off_short_circuits_to_routine(tmp_path: Path):
    """Even a textbook ACS presentation comes back routine under ``off``."""
    rules, cfg = _acs_rule_pack()
    triage = EmergencyTriage(rules=rules, config=cfg)
    path = _write_case_file(
        tmp_path,
        """
        cases:
          - id: acs_textbook
            source: public_vignettes
            language: en
            symptoms:
              primary_complaint: chest_pain
              qualifiers: [radiation_left_arm, diaphoresis]
              age: 55
            ground_truth_level: critical
            ground_truth_rule_id: acs_acute_coronary_syndrome
        """,
    )
    cases = load_cases([path])
    preds = asyncio.run(
        run_profile(cases, triage, sensitivity="off", has_extractor=False)
    )
    assert preds[0].predicted_level == "routine"
    assert preds[0].matched_rule_ids == []


def test_run_profile_demographic_gate_excludes_young_patient(tmp_path: Path):
    """Age 28 < 35 → ACS rule should not fire even with classic qualifiers."""
    rules, cfg = _acs_rule_pack()
    triage = EmergencyTriage(rules=rules, config=cfg)
    path = _write_case_file(
        tmp_path,
        """
        cases:
          - id: young_chest_pain
            source: public_vignettes
            language: en
            symptoms:
              primary_complaint: chest_pain
              qualifiers: [radiation_left_arm, diaphoresis]
              age: 28
            ground_truth_level: routine
        """,
    )
    cases = load_cases([path])
    preds = asyncio.run(
        run_profile(cases, triage, sensitivity="balanced", has_extractor=False)
    )
    # Demographic gate excludes the rule; no rule fired → routine.
    assert preds[0].predicted_level == "routine"
