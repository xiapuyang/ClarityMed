"""Deterministic matcher: extracted symptoms × rule list → matched rules.

Pure Python by design (KTD-E2). Every emergency decision the gate
emits traces to a citable rule in ``configs/emergency_rules.yaml``,
not to an LLM softmax — that is what makes the system defensible vs.
"we asked a model". The composer LLM still writes user-facing prose
on top of these matches, but the *decision* lives here.

Returned :class:`MatchedRule` objects are sorted by severity
(critical → routine), with rules of the same severity preserved in
input order. The triage facade reads the first element as the
authoritative level for short-circuit + suggested-action decisions.
"""

from __future__ import annotations

from claritymed.core.emergency.rules import Rule
from claritymed.core.emergency.schemas import (
    EmergencyLevel,
    ExtractedSymptoms,
    MatchedRule,
)

_LEVEL_ORDER: dict[EmergencyLevel, int] = {
    "critical": 0,
    "urgent": 1,
    "moderate": 2,
    "routine": 3,
}


def _gate_demographics(symptoms: ExtractedSymptoms, rule: Rule) -> bool:
    """Return False when an age/sex/key_history gate excludes the rule.

    Design intent (Fix #3): unknown age/sex blocks age/sex-gated rules.
    This is the safe default for ambiguous-data cases.

    Rationale: ACS (age_min=35) and ectopic_pregnancy_bleed (sex=F)
    firing on totally unknown demographics would generate too many false
    positives in non-clinical chitchat (e.g., a greeting before the
    user has described their age or sex). The ambiguous_high_risk
    catch-all (balanced+strict profiles) catches chest-pain-pattern
    cases at level=urgent instead, which still drives the "please
    seek evaluation" path without the demographic assumption.

    Concretely:
    - age gate: if age_min or age_max is set and symptoms.age is None,
      the rule does NOT fire (safer to miss than to false-positive).
    - sex gate: if the rule requires sex=F or sex=M and symptoms.sex
      is None, the rule does NOT fire.

    This behavior is intentional and tested by
    test_age_none_blocks_age_gated_acs and test_sex_none_blocks_ectopic.
    Do not remove the ``symptoms.age is None`` short-circuit without
    updating those tests first.
    """
    t = rule.triggers
    if t.age_min is not None and (symptoms.age is None or symptoms.age < t.age_min):
        return False
    if t.age_max is not None and (symptoms.age is None or symptoms.age > t.age_max):
        return False
    if t.sex is not None and symptoms.sex != t.sex:
        return False
    if t.key_history_any_of:
        if not any(h in symptoms.key_history for h in t.key_history_any_of):
            return False
    return True


def _matched_qualifiers(symptoms: ExtractedSymptoms, rule: Rule) -> list[str]:
    """Return the subset of the rule's ``any_of`` present in symptoms."""
    return [q for q in rule.triggers.any_of if q in symptoms.qualifiers]


def _gate_primary(symptoms: ExtractedSymptoms, rule: Rule) -> bool:
    """Return True when the rule's primary gate passes.

    Gate semantics:
    - ``None``   → no primary gate; always passes (e.g. anaphylaxis
                    fires on qualifier triad alone).
    - ``str``    → exact equality with ``symptoms.primary_complaint``.
    - ``list``   → membership check; any listed primary is sufficient
                    (OR semantics). Used by rules with multiple valid
                    presentations, e.g. pulmonary_embolism fires on
                    both ``dyspnea`` and ``chest_pain``.
    """
    t = rule.triggers
    if t.primary is None:
        return True
    if isinstance(t.primary, list):
        return symptoms.primary_complaint in t.primary
    return symptoms.primary_complaint == t.primary


def match(symptoms: ExtractedSymptoms, rules: list[Rule]) -> list[MatchedRule]:
    """Walk every rule in ``rules`` and return the firing matches.

    Match conditions (AND across the four):

    1. ``triggers.primary`` is None, equals ``symptoms.primary_complaint``,
       or (when a list) contains ``symptoms.primary_complaint``.
    2. The count of matched qualifiers ``>=`` ``triggers.min_qualifier_matches``
       (a zero-threshold ambiguous rule firing on any primary suffices).
    3. Demographic gates (age / sex / key_history) all pass.
    4. The rule has at least one qualifier match **or** carries an
       ambiguous catch-all that explicitly allows zero matches.

    Output is sorted by severity (most severe first) so callers can
    read element 0 as the authoritative level.
    """
    out: list[MatchedRule] = []
    for rule in rules:
        if not _gate_primary(symptoms, rule):
            continue
        if not _gate_demographics(symptoms, rule):
            continue
        hits = _matched_qualifiers(symptoms, rule)
        if len(hits) < rule.triggers.min_qualifier_matches:
            continue
        # ``min=0`` is intentional for ambiguous catch-alls (any
        # chest_pain → urgent). All other rules require at least one
        # qualifier match, which the >= check already enforces.
        out.append(
            MatchedRule(
                rule_id=rule.id,
                level=rule.level,
                suggested_action_i18n_key=rule.action_key,
                citations=list(rule.citations),
                matched_qualifiers=hits,
            )
        )
    out.sort(key=lambda m: _LEVEL_ORDER[m.level])
    return out
