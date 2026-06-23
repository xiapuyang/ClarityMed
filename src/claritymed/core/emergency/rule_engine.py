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
    """Return False when an age/sex/key_history gate excludes the rule."""
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


def match(symptoms: ExtractedSymptoms, rules: list[Rule]) -> list[MatchedRule]:
    """Walk every rule in ``rules`` and return the firing matches.

    Match conditions (AND across the four):

    1. ``triggers.primary`` is None **or** equals ``symptoms.primary_complaint``.
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
        t = rule.triggers
        if t.primary is not None and symptoms.primary_complaint != t.primary:
            continue
        if not _gate_demographics(symptoms, rule):
            continue
        hits = _matched_qualifiers(symptoms, rule)
        if len(hits) < t.min_qualifier_matches:
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
