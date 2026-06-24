"""Phase 2 tests: rule pack, rule engine, profile floor enforcement, composer.

Pins the deterministic-decision invariant (KTD-E2): every emergency
call must trace to a real rule, and every profile override must
respect each rule's ``minimum_sensitivity_floor`` at config-load.
"""

from __future__ import annotations

from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from claritymed.core.emergency import EmergencyTriage
from claritymed.core.emergency.composer import Composer, build_assessment
from claritymed.core.emergency.config import (
    EmergencyConfig,
    SensitivityProfile,
)
from claritymed.core.emergency.rule_engine import match
from claritymed.core.emergency.rules import (
    EMERGENCY_RULES_FILENAME,
    FloorViolationError,
    Rule,
    RulePack,
    RuleTriggers,
    apply_profile_to_rules,
    enforce_floors,
    load_rules,
    load_validated_emergency_config,
)
from claritymed.core.emergency.schemas import ExtractedSymptoms, MatchedRule
from claritymed.core.prompts.registry import PromptRegistry


# --- rule pack + ambiguous validator --------------------------------


def test_load_rules_parses_repo_yaml():
    rules = load_rules()
    ids = {r.id for r in rules}
    # Phase 2 rule pack: 5 high-yield + 1 ambiguous catch-all.
    assert "acs_acute_coronary_syndrome" in ids
    assert "stroke_fast_positive" in ids
    assert "anaphylaxis" in ids
    assert "sah_thunderclap_headache" in ids
    assert "active_suicidal_ideation" in ids
    assert "chest_pain_ambiguous_high_risk" in ids


def test_anaphylaxis_floor_is_lenient():
    rules = {r.id: r for r in load_rules()}
    assert rules["anaphylaxis"].minimum_sensitivity_floor == "lenient"
    assert rules["active_suicidal_ideation"].minimum_sensitivity_floor == "lenient"


def test_ambiguous_rule_requires_missing_qualifiers_hint():
    with pytest.raises(ValidationError):
        Rule(
            id="bad_ambiguous",
            triggers=RuleTriggers(primary="chest_pain", min_qualifier_matches=0),
            level="urgent",
            action_key="emergency.action.x",
            ambiguous=True,
            missing_qualifiers_hint=[],
            citations=["cite"],
        )


def test_load_rules_missing_file_returns_empty(tmp_path):
    assert load_rules(tmp_path / "nope.yaml") == []


def test_rule_pack_requires_at_least_one_rule():
    with pytest.raises(ValidationError):
        RulePack.model_validate({"rules": []})


# --- floor enforcement ----------------------------------------------


def _baseline_rules() -> list[Rule]:
    """Two rules with known floors for floor-violation tests."""
    return [
        Rule(
            id="anaphylaxis",
            triggers=RuleTriggers(any_of=["throat_tightness"], min_qualifier_matches=1),
            level="critical",
            action_key="emergency.action.epi_then_ems",
            minimum_sensitivity_floor="lenient",
            citations=["WAO 2020"],
        ),
        Rule(
            id="acs",
            triggers=RuleTriggers(
                primary="chest_pain", any_of=["diaphoresis"], min_qualifier_matches=1
            ),
            level="critical",
            action_key="emergency.action.call_ems_cardiac",
            minimum_sensitivity_floor="balanced",
            citations=["AHA/ACC 2021"],
        ),
    ]


def test_floor_enforcement_lenient_floor_blocks_any_threshold_raise():
    rules = _baseline_rules()
    profiles = {
        "lenient": SensitivityProfile(
            rule_overrides={"anaphylaxis": {"min_qualifier_matches": 2}},
            ambiguous_rules_enabled=False,
        ),
    }
    with pytest.raises(FloorViolationError) as exc:
        enforce_floors(rules, profiles)
    assert "anaphylaxis" in str(exc.value)
    assert "min_qualifier_matches" in str(exc.value)


def test_floor_enforcement_balanced_floor_allows_lenient_override():
    """Plan example: lenient may raise ACS (floor=balanced) to 2.

    The rule author signals "operator discretion" via
    ``floor=balanced``; lenient profile's threshold raise is therefore
    permitted. Only ``floor=lenient`` seals the rule.
    """
    rules = _baseline_rules()
    profiles = {
        "lenient": SensitivityProfile(
            rule_overrides={"acs": {"min_qualifier_matches": 2}},
            ambiguous_rules_enabled=False,
        ),
    }
    enforce_floors(rules, profiles)  # no raise


def test_floor_enforcement_lenient_floor_blocks_any_field_override():
    """A sealed rule rejects any override field, not just threshold."""
    rules = _baseline_rules()
    profiles = {
        "lenient": SensitivityProfile(
            rule_overrides={"anaphylaxis": {"level": "urgent"}},
        ),
    }
    with pytest.raises(FloorViolationError) as exc:
        enforce_floors(rules, profiles)
    assert "sealed" in str(exc.value).lower()


def test_floor_enforcement_strict_lowering_threshold_allowed():
    rules = _baseline_rules()
    # Strict profile lowering ACS threshold to 0 — allowed; that's
    # the whole point of strict mode.
    profiles = {
        "strict": SensitivityProfile(
            rule_overrides={"acs": {"min_qualifier_matches": 0}},
            ambiguous_rules_enabled=True,
        ),
    }
    # No raise — strict + balanced are both allowed lowerers.
    enforce_floors(rules, profiles)


def test_floor_enforcement_unknown_rule_id_rejected():
    rules = _baseline_rules()
    profiles = {
        "lenient": SensitivityProfile(
            rule_overrides={"nope": {"min_qualifier_matches": 5}},
        ),
    }
    with pytest.raises(FloorViolationError) as exc:
        enforce_floors(rules, profiles)
    assert "unknown" in str(exc.value).lower()


def test_load_validated_emergency_config_passes_for_repo_yaml():
    """The shipped repo YAML must pass floor enforcement."""
    cfg, rules = load_validated_emergency_config()
    assert isinstance(cfg, EmergencyConfig)
    assert len(rules) >= 5
    # ``lenient`` profile's ACS override is allowed (floor=balanced).
    # Anaphylaxis is left untouched (would violate its lenient floor).
    lenient = cfg.sensitivity_profiles["lenient"]
    assert "anaphylaxis" not in lenient.rule_overrides


def test_load_validated_emergency_config_raises_on_bad_floor(tmp_path, monkeypatch):
    """Inject a violating YAML and confirm load fails."""
    # Point CONFIGS_DIR to a tmp copy with a tampered profile.
    from claritymed.config import CONFIGS_DIR
    import claritymed.core.emergency.config as cfg_mod
    import claritymed.core.emergency.rules as rules_mod

    # Mirror the rule pack as-is.
    real_rules = (CONFIGS_DIR / EMERGENCY_RULES_FILENAME).read_text()
    (tmp_path / EMERGENCY_RULES_FILENAME).write_text(real_rules)

    # Tampered emergency.yaml: lenient raises anaphylaxis threshold.
    bad = {
        "default_sensitivity": "balanced",
        "sensitivity_profiles": {
            "balanced": {},
            "lenient": {
                "rule_overrides": {"anaphylaxis": {"min_qualifier_matches": 5}},
                "ambiguous_rules_enabled": False,
            },
        },
    }
    (tmp_path / "emergency.yaml").write_text(yaml.safe_dump(bad))

    monkeypatch.setattr(cfg_mod, "CONFIGS_DIR", tmp_path)
    monkeypatch.setattr(rules_mod, "CONFIGS_DIR", tmp_path)
    with pytest.raises(FloorViolationError):
        load_validated_emergency_config()


# --- apply_profile_to_rules -----------------------------------------


def test_apply_profile_drops_ambiguous_when_disabled():
    rules = _baseline_rules() + [
        Rule(
            id="chest_pain_ambig",
            triggers=RuleTriggers(primary="chest_pain", min_qualifier_matches=0),
            level="urgent",
            action_key="emergency.action.urgent_eval_chest_pain",
            ambiguous=True,
            missing_qualifiers_hint=["radiation"],
            citations=["ACEP 2018"],
        ),
    ]
    profile = SensitivityProfile(ambiguous_rules_enabled=False)
    effective = apply_profile_to_rules(rules, profile)
    assert all(not r.ambiguous for r in effective)


def test_apply_profile_applies_min_qualifier_override():
    rules = _baseline_rules()
    profile = SensitivityProfile(rule_overrides={"acs": {"min_qualifier_matches": 0}})
    effective = apply_profile_to_rules(rules, profile)
    acs = next(r for r in effective if r.id == "acs")
    assert acs.triggers.min_qualifier_matches == 0


def test_apply_profile_does_not_mutate_input():
    rules = _baseline_rules()
    profile = SensitivityProfile(rule_overrides={"acs": {"min_qualifier_matches": 0}})
    _ = apply_profile_to_rules(rules, profile)
    acs = next(r for r in rules if r.id == "acs")
    assert acs.triggers.min_qualifier_matches == 1  # original untouched


# --- rule_engine.match ----------------------------------------------


def _acs_rule() -> Rule:
    return Rule(
        id="acs",
        triggers=RuleTriggers(
            primary="chest_pain",
            any_of=["radiation_left_arm", "diaphoresis", "dyspnea"],
            min_qualifier_matches=1,
            age_min=35,
        ),
        level="critical",
        action_key="emergency.action.call_ems_cardiac",
        citations=["AHA/ACC 2021"],
    )


def _stroke_rule() -> Rule:
    return Rule(
        id="stroke",
        triggers=RuleTriggers(
            primary="neuro_deficit",
            any_of=["facial_droop", "arm_weakness", "speech_slurred", "sudden_onset"],
            min_qualifier_matches=2,
        ),
        level="critical",
        action_key="emergency.action.call_ems_stroke",
        citations=["AHA/ASA 2019"],
    )


def _anaphylaxis_rule() -> Rule:
    return Rule(
        id="anaphylaxis",
        triggers=RuleTriggers(
            any_of=["throat_tightness", "lip_tongue_swelling"],
            min_qualifier_matches=1,
        ),
        level="critical",
        action_key="emergency.action.epi_then_ems",
        minimum_sensitivity_floor="lenient",
        citations=["WAO 2020"],
    )


def _ambiguous_chest_pain() -> Rule:
    return Rule(
        id="chest_pain_ambig",
        triggers=RuleTriggers(primary="chest_pain", min_qualifier_matches=0),
        level="urgent",
        action_key="emergency.action.urgent_eval_chest_pain",
        ambiguous=True,
        missing_qualifiers_hint=["radiation"],
        citations=["ACEP 2018"],
    )


def test_match_acs_happy_path():
    rules = [_acs_rule()]
    symptoms = ExtractedSymptoms(
        primary_complaint="chest_pain",
        qualifiers=["radiation_left_arm", "diaphoresis"],
        age=55,
        sex="M",
    )
    out = match(symptoms, rules)
    assert len(out) == 1
    assert out[0].rule_id == "acs"
    assert out[0].level == "critical"
    assert set(out[0].matched_qualifiers) == {"radiation_left_arm", "diaphoresis"}


def test_match_acs_skipped_under_age_gate():
    rules = [_acs_rule()]
    symptoms = ExtractedSymptoms(
        primary_complaint="chest_pain",
        qualifiers=["radiation_left_arm"],
        age=20,
    )
    assert match(symptoms, rules) == []


def test_match_acs_misses_without_qualifier():
    rules = [_acs_rule()]
    symptoms = ExtractedSymptoms(primary_complaint="chest_pain", age=55)
    assert match(symptoms, rules) == []


def test_match_stroke_requires_two_qualifiers():
    rules = [_stroke_rule()]
    one = ExtractedSymptoms(
        primary_complaint="neuro_deficit", qualifiers=["facial_droop"]
    )
    assert match(one, rules) == []
    two = ExtractedSymptoms(
        primary_complaint="neuro_deficit",
        qualifiers=["facial_droop", "arm_weakness"],
    )
    out = match(two, rules)
    assert len(out) == 1
    assert out[0].rule_id == "stroke"


def test_match_anaphylaxis_no_primary_required():
    rules = [_anaphylaxis_rule()]
    symptoms = ExtractedSymptoms(
        primary_complaint=None, qualifiers=["throat_tightness"]
    )
    out = match(symptoms, rules)
    assert len(out) == 1
    assert out[0].rule_id == "anaphylaxis"


def test_match_ambiguous_catches_bare_chest_pain():
    rules = [_acs_rule(), _ambiguous_chest_pain()]
    # No qualifier, age too young for ACS — ambiguous still fires.
    symptoms = ExtractedSymptoms(primary_complaint="chest_pain", age=20)
    out = match(symptoms, rules)
    assert len(out) == 1
    assert out[0].rule_id == "chest_pain_ambig"
    assert out[0].level == "urgent"


def test_match_sorts_critical_before_urgent():
    rules = [_ambiguous_chest_pain(), _acs_rule()]
    symptoms = ExtractedSymptoms(
        primary_complaint="chest_pain",
        qualifiers=["radiation_left_arm"],
        age=55,
    )
    out = match(symptoms, rules)
    assert [m.level for m in out] == ["critical", "urgent"]


# --- build_assessment ------------------------------------------------


def test_build_assessment_unions_citations_and_picks_top_action():
    matched = [
        MatchedRule(
            rule_id="acs",
            level="critical",
            suggested_action_i18n_key="emergency.action.call_ems_cardiac",
            citations=["AHA/ACC 2021", "ESI v4 Level 1-2"],
            matched_qualifiers=["diaphoresis"],
        ),
        MatchedRule(
            rule_id="chest_pain_ambig",
            level="urgent",
            suggested_action_i18n_key="emergency.action.urgent_eval_chest_pain",
            citations=["ACEP 2018", "AHA/ACC 2021"],  # duplicate
            matched_qualifiers=[],
        ),
    ]
    symptoms = ExtractedSymptoms(
        primary_complaint="chest_pain", qualifiers=["diaphoresis"]
    )
    a = build_assessment(matched, symptoms, reasoning="severe chest pain pattern")
    assert a.level == "critical"
    assert a.suggested_action_i18n_key == "emergency.action.call_ems_cardiac"
    # Citations deduplicated, order preserved.
    assert a.citations == ["AHA/ACC 2021", "ESI v4 Level 1-2", "ACEP 2018"]


def test_build_assessment_empty_matched_is_routine():
    symptoms = ExtractedSymptoms()
    a = build_assessment([], symptoms, reasoning="")
    assert a.level == "routine"
    assert a.matched_rules == []


# --- EmergencyTriage end-to-end -------------------------------------


class _StubComposer:
    """Deterministic composer for tests — no LLM."""

    async def compose(
        self, matched_rules: list[MatchedRule], symptoms, *, language: str
    ) -> str:
        ids = ",".join(m.rule_id for m in matched_rules)
        return f"[{language}] matched: {ids}"


@pytest.mark.asyncio
async def test_triage_assess_from_symptoms_routes_through_rules():
    triage = EmergencyTriage(rules=[_acs_rule()], composer=_StubComposer())
    symptoms = ExtractedSymptoms(
        primary_complaint="chest_pain",
        qualifiers=["radiation_left_arm"],
        age=55,
    )
    result = await triage.assess_from_symptoms(
        symptoms, sensitivity="balanced", language="en"
    )
    assert result.level == "critical"
    assert result.suggested_action_i18n_key == "emergency.action.call_ems_cardiac"
    assert result.reasoning == "[en] matched: acs"


@pytest.mark.asyncio
async def test_triage_assess_from_symptoms_off_short_circuits(monkeypatch):
    from claritymed.core.emergency import service as svc_mod

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        svc_mod,
        "audit_event",
        lambda kind, payload=None: calls.append((kind, payload or {})),
    )
    triage = EmergencyTriage(rules=[_acs_rule()], composer=_StubComposer())
    symptoms = ExtractedSymptoms(
        primary_complaint="chest_pain",
        qualifiers=["radiation_left_arm"],
        age=55,
    )
    result = await triage.assess_from_symptoms(
        symptoms, sensitivity="off", language="en"
    )
    assert result.level == "routine"
    # ``redflag.gate_disabled`` is emitted by ``assess()``, not by
    # ``assess_from_symptoms()``. Direct callers (tests, evals) that call
    # ``assess_from_symptoms`` with sensitivity="off" are not real gate
    # runs, so no audit event is expected here.
    assert not any(c[0] == "redflag.gate_disabled" for c in calls)


@pytest.mark.asyncio
async def test_triage_assess_from_symptoms_no_match_returns_routine():
    triage = EmergencyTriage(rules=[_acs_rule()], composer=_StubComposer())
    symptoms = ExtractedSymptoms(
        primary_complaint="headache", qualifiers=["mild"], age=55
    )
    result = await triage.assess_from_symptoms(
        symptoms, sensitivity="balanced", language="en"
    )
    assert result.level == "routine"


@pytest.mark.asyncio
async def test_triage_assess_without_extractor_returns_routine():
    triage = EmergencyTriage(rules=[_acs_rule()])
    result = await triage.assess(
        "我胸口剧痛", history=None, sensitivity="balanced", language="zh"
    )
    assert result.level == "routine"


@pytest.mark.asyncio
async def test_triage_composer_failure_falls_open():
    class _BoomComposer:
        async def compose(self, *_a, **_kw) -> str:
            raise RuntimeError("model down")

    triage = EmergencyTriage(rules=[_acs_rule()], composer=_BoomComposer())
    symptoms = ExtractedSymptoms(
        primary_complaint="chest_pain",
        qualifiers=["radiation_left_arm"],
        age=55,
    )
    result = await triage.assess_from_symptoms(
        symptoms, sensitivity="balanced", language="en"
    )
    # Critical level + action key survive composer downtime — the
    # i18n action key carries the load-bearing safety advice.
    assert result.level == "critical"
    assert result.suggested_action_i18n_key == "emergency.action.call_ems_cardiac"
    assert result.reasoning == ""


# --- composer prompt registry ---------------------------------------


def test_emergency_composer_prompt_loads_both_languages():
    registry = PromptRegistry()
    en = registry.get("emergency_composer", language="en")
    zh = registry.get("emergency_composer", language="zh")
    assert "matched_rules" in en
    assert "matched_rules" in zh


# --- i18n action keys ------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "emergency.action.call_ems_cardiac",
        "emergency.action.call_ems_stroke",
        "emergency.action.epi_then_ems",
        "emergency.action.call_ems_sah",
        "emergency.action.crisis_hotline_988",
        "emergency.action.urgent_eval_chest_pain",
    ],
)
def test_action_keys_present_in_both_languages(key):
    from claritymed.core.i18n.loader import t

    assert t(key, lang="en") != key
    assert t(key, lang="zh") != key


# --- Composer Protocol shape check ----------------------------------


def test_stub_composer_satisfies_protocol():
    """Quick structural confirmation — fail at test-collection if the
    Protocol signature drifts."""
    stub: Composer = _StubComposer()  # type: ignore[assignment]
    assert hasattr(stub, "compose")


# --- _ComposerInput rendering --------------------------------------


def test_composer_input_text_renders_all_fields():
    from claritymed.core.emergency.composer import _ComposerInput

    matched = [
        MatchedRule(
            rule_id="acs",
            level="critical",
            suggested_action_i18n_key="emergency.action.call_ems_cardiac",
            citations=["x"],
            matched_qualifiers=["radiation_left_arm"],
        ),
    ]
    symptoms = ExtractedSymptoms(
        primary_complaint="chest_pain",
        qualifiers=["radiation_left_arm"],
        age=55,
        sex="M",
        key_history=["prior_mi"],
    )
    text = _ComposerInput(matched, symptoms).to_text()
    assert "id: acs" in text
    assert "level: critical" in text
    assert "primary_complaint: chest_pain" in text
    assert "age: 55" in text
    assert "sex: M" in text
    assert "prior_mi" in text


def test_composer_input_text_omits_none_fields():
    """``age``/``sex``/``key_history`` are skipped when not present."""
    from claritymed.core.emergency.composer import _ComposerInput

    matched: list[MatchedRule] = []
    symptoms = ExtractedSymptoms(primary_complaint="chest_pain")
    text = _ComposerInput(matched, symptoms).to_text()
    assert "age:" not in text
    assert "sex:" not in text
    assert "key_history:" not in text


# --- rule_engine demographic gates ---------------------------------


def test_match_age_max_gate_blocks():
    rule = Rule(
        id="pediatric",
        triggers=RuleTriggers(primary="fever", age_max=5, min_qualifier_matches=0),
        level="urgent",
        action_key="emergency.action.x",
        citations=["c"],
    )
    # Adult excluded by age_max.
    symptoms = ExtractedSymptoms(primary_complaint="fever", age=30)
    assert match(symptoms, [rule]) == []
    # Toddler passes.
    symptoms2 = ExtractedSymptoms(primary_complaint="fever", age=2)
    assert len(match(symptoms2, [rule])) == 1


def test_match_sex_gate_blocks():
    rule = Rule(
        id="ectopic",
        triggers=RuleTriggers(
            primary="abdominal_pain", sex="F", min_qualifier_matches=0
        ),
        level="critical",
        action_key="emergency.action.x",
        citations=["c"],
    )
    male = ExtractedSymptoms(primary_complaint="abdominal_pain", sex="M")
    assert match(male, [rule]) == []
    female = ExtractedSymptoms(primary_complaint="abdominal_pain", sex="F")
    assert len(match(female, [rule])) == 1


def test_match_key_history_gate_requires_any():
    rule = Rule(
        id="anticoag_head_trauma",
        triggers=RuleTriggers(
            primary="head_trauma",
            key_history_any_of=["anticoagulant", "warfarin"],
            min_qualifier_matches=0,
        ),
        level="critical",
        action_key="emergency.action.x",
        citations=["c"],
    )
    no_hx = ExtractedSymptoms(primary_complaint="head_trauma", key_history=[])
    assert match(no_hx, [rule]) == []
    on_warfarin = ExtractedSymptoms(
        primary_complaint="head_trauma", key_history=["warfarin"]
    )
    assert len(match(on_warfarin, [rule])) == 1


# --- service: config-driven effective_rules + extractor failure -----


@pytest.mark.asyncio
async def test_triage_uses_config_profiles_for_effective_rules():
    """When a config is wired, ``assess_from_symptoms`` should apply
    the profile's overrides (e.g. lenient drops ambiguous rules)."""
    from claritymed.core.emergency.config import (
        EmergencyConfig,
        SensitivityProfile,
    )

    rules = [_acs_rule(), _ambiguous_chest_pain()]
    config = EmergencyConfig(
        default_sensitivity="balanced",
        sensitivity_profiles={
            "balanced": SensitivityProfile(),
            "lenient": SensitivityProfile(ambiguous_rules_enabled=False),
        },
    )
    triage = EmergencyTriage(rules=rules, config=config, composer=_StubComposer())
    # Bare chest_pain in `balanced` → ambiguous fires (urgent).
    bare = ExtractedSymptoms(primary_complaint="chest_pain", age=20)
    out_balanced = await triage.assess_from_symptoms(
        bare, sensitivity="balanced", language="en"
    )
    assert out_balanced.level == "urgent"
    # Same input in `lenient` → ambiguous filtered out, no match → routine.
    out_lenient = await triage.assess_from_symptoms(
        bare, sensitivity="lenient", language="en"
    )
    assert out_lenient.level == "routine"


@pytest.mark.asyncio
async def test_triage_assess_extractor_failure_falls_open(caplog):
    import logging

    class _BoomExtractor:
        async def extract(self, *_a, **_kw):
            raise RuntimeError("extractor model down")

    triage = EmergencyTriage(
        rules=[_acs_rule()], composer=_StubComposer(), extractor=_BoomExtractor()
    )
    with caplog.at_level(logging.ERROR):
        result = await triage.assess(
            "我胸口剧痛",
            history=None,
            sensitivity="balanced",
            language="zh",
        )
    assert result.level == "routine"
    assert any("extractor failed" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_triage_assess_routes_through_extractor_on_success():
    """With a working extractor, ``assess`` delegates to
    ``assess_from_symptoms`` and yields the matched level."""

    class _FakeExtractor:
        async def extract(self, query, history, *, age=None, sex=None):
            del query, history, age, sex
            return ExtractedSymptoms(
                primary_complaint="chest_pain",
                qualifiers=["radiation_left_arm"],
                age=55,
            )

    triage = EmergencyTriage(
        rules=[_acs_rule()],
        composer=_StubComposer(),
        extractor=_FakeExtractor(),
    )
    result = await triage.assess(
        "I have severe chest pain radiating to my left arm",
        history=None,
        sensitivity="balanced",
        language="en",
    )
    assert result.level == "critical"
    assert result.suggested_action_i18n_key == "emergency.action.call_ems_cardiac"


@pytest.mark.asyncio
async def test_triage_assess_forwards_profile_demographics_to_extractor():
    """``assess`` threads ``profile_age`` / ``profile_sex`` into ``extract``.

    AskService reads these from ``ProfileStore`` so demographic-gated
    rules fire even when the user has not re-stated their basics this
    turn. Lock the wiring with a capturing fake extractor.
    """
    captured: dict[str, Any] = {}

    class _CapturingExtractor:
        async def extract(self, query, history, *, age=None, sex=None):
            captured["query"] = query
            captured["history"] = history
            captured["age"] = age
            captured["sex"] = sex
            return ExtractedSymptoms(primary_complaint=None)

    triage = EmergencyTriage(
        rules=[_acs_rule()],
        composer=_StubComposer(),
        extractor=_CapturingExtractor(),
    )
    await triage.assess(
        "abdominal pain",
        history=None,
        sensitivity="balanced",
        language="en",
        profile_age=32,
        profile_sex="F",
    )
    assert captured["age"] == 32
    assert captured["sex"] == "F"


# --- Fix #2: merged PE rule (multi-primary OR) + duplicate id guard ---


def _pe_rule() -> Rule:
    """Merged PE rule with list primary — covers both presentations."""
    return Rule(
        id="pulmonary_embolism",
        triggers=RuleTriggers(
            primary=["dyspnea", "chest_pain"],
            any_of=[
                "pleuritic",
                "unilateral_leg_swelling",
                "hemoptysis",
                "syncope",
                "tachycardia",
            ],
            min_qualifier_matches=1,
        ),
        level="critical",
        action_key="emergency.action.urgent_eval_pe",
        citations=[
            "Wells Criteria (Ann Intern Med 2001;135:98)",
            "ACEP Clinical Policy: Suspected PE 2018",
        ],
    )


def test_rule_pack_rejects_duplicate_ids():
    """RulePack.model_validate must raise when two rules share the same id."""
    rule_data = {
        "rules": [
            {
                "id": "dup_rule",
                "triggers": {
                    "any_of": ["throat_tightness"],
                    "min_qualifier_matches": 1,
                },
                "level": "critical",
                "action_key": "emergency.action.epi_then_ems",
                "citations": ["WAO 2020"],
            },
            {
                "id": "dup_rule",
                "triggers": {
                    "any_of": ["lip_tongue_swelling"],
                    "min_qualifier_matches": 1,
                },
                "level": "critical",
                "action_key": "emergency.action.epi_then_ems",
                "citations": ["WAO 2020"],
            },
        ]
    }
    with pytest.raises(ValidationError) as exc:
        RulePack.model_validate(rule_data)
    assert "dup_rule" in str(exc.value)


def test_merged_pe_rule_fires_on_dyspnea_with_pleuritic():
    """Dyspnea-primary presentation: merged PE rule fires."""
    rule = _pe_rule()
    symptoms = ExtractedSymptoms(
        primary_complaint="dyspnea",
        qualifiers=["pleuritic"],
    )
    results = match(symptoms, [rule])
    assert len(results) == 1
    assert results[0].rule_id == "pulmonary_embolism"
    assert results[0].level == "critical"


def test_merged_pe_rule_fires_on_chest_pain_with_pleuritic():
    """Chest-pain-primary presentation: merged PE rule also fires."""
    rule = _pe_rule()
    symptoms = ExtractedSymptoms(
        primary_complaint="chest_pain",
        qualifiers=["pleuritic"],
    )
    results = match(symptoms, [rule])
    assert len(results) == 1
    assert results[0].rule_id == "pulmonary_embolism"
    assert results[0].level == "critical"


def test_merged_pe_rule_does_not_fire_on_headache():
    """Non-PE primary (headache) must not match the PE rule."""
    rule = _pe_rule()
    symptoms = ExtractedSymptoms(
        primary_complaint="headache",
        qualifiers=["pleuritic"],
    )
    results = match(symptoms, [rule])
    assert results == []


def test_load_rules_no_duplicate_pe_ids():
    """The repo YAML must now have exactly one pulmonary_embolism rule."""
    rules = load_rules()
    pe_rules = [r for r in rules if r.id == "pulmonary_embolism"]
    assert len(pe_rules) == 1, (
        f"Expected exactly one pulmonary_embolism rule, got {len(pe_rules)}"
    )
    # Merged rule's primary must be a list covering both presentations.
    assert isinstance(pe_rules[0].triggers.primary, list)
    assert "dyspnea" in pe_rules[0].triggers.primary
    assert "chest_pain" in pe_rules[0].triggers.primary


# --- Fix #3: unknown age/sex blocks demographic-gated rules -----------


def test_age_none_blocks_age_gated_acs():
    """Unknown age blocks ACS (age_min=35); balanced profile fires chest_pain_ambiguous_high_risk at urgent.

    Rationale: ACS (age_min=35) firing on totally unknown demographics
    would generate too many false positives in non-clinical chitchat.
    The ambiguous_high_risk catch-all catches these cases at urgent instead.
    See _gate_demographics docstring for full rationale.
    """
    rules = load_rules()
    # Balanced profile: ambiguous rules are enabled (default).
    from claritymed.core.emergency.config import SensitivityProfile

    profile = SensitivityProfile(ambiguous_rules_enabled=True)
    effective_rules = apply_profile_to_rules(rules, profile)

    # Classic ACS symptom pattern but no age known.
    symptoms = ExtractedSymptoms(
        primary_complaint="chest_pain",
        qualifiers=["radiation_left_arm", "diaphoresis"],
        age=None,
    )
    matched = match(symptoms, effective_rules)
    rule_ids = [m.rule_id for m in matched]

    # ACS must NOT fire (age_min=35 gate blocks when age=None).
    assert "acs_acute_coronary_syndrome" not in rule_ids, (
        "ACS fired with age=None — demographic gate is broken"
    )
    # The ambiguous catch-all SHOULD fire for balanced profile.
    assert "chest_pain_ambiguous_high_risk" in rule_ids, (
        "Expected chest_pain_ambiguous_high_risk to fire as catch-all"
    )
    acs_level = next(
        (m.level for m in matched if m.rule_id == "chest_pain_ambiguous_high_risk"),
        None,
    )
    assert acs_level == "urgent"


def test_sex_none_blocks_ectopic():
    """Unknown sex blocks ectopic_pregnancy_bleed (sex=F gate).

    Rationale: ectopic_pregnancy_bleed (sex=F) firing on totally unknown
    demographics would generate false positives for male users. The
    ambiguous_high_risk catch-all is not wired for abdominal pain in v1,
    so the rule simply doesn't fire.
    """
    rules = load_rules()
    symptoms = ExtractedSymptoms(
        primary_complaint="abdominal_pain",
        qualifiers=["vaginal_bleeding", "missed_period"],
        age=28,
        sex=None,  # sex unknown
    )
    matched = match(symptoms, rules)
    rule_ids = [m.rule_id for m in matched]
    assert "ectopic_pregnancy_bleed" not in rule_ids, (
        "ectopic_pregnancy_bleed fired with sex=None — sex gate is broken"
    )
