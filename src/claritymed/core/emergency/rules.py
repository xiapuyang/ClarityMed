"""Emergency rule pack schema + loader + floor-enforcement validator.

The rule pack is the **deterministic** half of the gate (KTD-E2): the
extractor LLM produces :class:`ExtractedSymptoms`, the matcher then
walks this rule list with no model in the loop. Every emergency
decision the system makes therefore traces to a citable, version-
controlled rule rather than to "we asked an LLM".

Floor enforcement (plan §"Sensitivity Profiles" → "Floor enforcement"):
every rule carries ``minimum_sensitivity_floor``; the load-time
validator rejects any profile ``rule_overrides`` entry that would
violate the floor. The runtime matcher trusts the validated config
and applies overrides without re-checking, so a corrupt YAML cannot
silently drop a "false positives cost less than false negatives" rule
(anaphylaxis, active SI, ectopic) via the lenient profile.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from claritymed.config import CONFIGS_DIR
from claritymed.core.emergency.config import (
    EmergencyConfig,
    SensitivityProfile,
)
from claritymed.core.emergency.schemas import EmergencyLevel, SensitivityName

EMERGENCY_RULES_FILENAME = "emergency_rules.yaml"


class RuleTriggers(BaseModel):
    """One rule's match criteria.

    All fields except ``any_of`` / ``min_qualifier_matches`` are
    optional gates with AND semantics. A None field is "don't gate";
    a non-None field must match the extracted symptoms exactly.
    """

    model_config = ConfigDict(extra="forbid")

    # Required primary complaint. None means "match on qualifiers alone"
    # (e.g. anaphylaxis fires on the throat/lip/urticaria triad without
    # a single named primary). A list means "any of these primaries matches"
    # (OR semantics) — used when a rule has multiple valid presentations
    # (e.g. pulmonary_embolism: dyspnea OR chest_pain).
    primary: str | list[str] | None = None
    any_of: list[str] = Field(default_factory=list)
    # Default 1; the ambiguous-high-risk catch-all sets this to 0 so
    # any chest_pain (no qualifier required) still triggers urgent.
    min_qualifier_matches: int = Field(default=1, ge=0)
    age_min: int | None = Field(default=None, ge=0, le=130)
    age_max: int | None = Field(default=None, ge=0, le=130)
    sex: Literal["F", "M"] | None = None
    key_history_any_of: list[str] = Field(default_factory=list)


class Rule(BaseModel):
    """One emergency rule, as authored in ``emergency_rules.yaml``."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    triggers: RuleTriggers
    level: EmergencyLevel
    action_key: str = Field(min_length=1)
    # ``lenient`` is the strictest floor (no override allowed);
    # ``balanced`` is the default (lenient overrides clamped at rule
    # default); ``strict`` means any profile may override freely.
    minimum_sensitivity_floor: Literal["strict", "balanced", "lenient"] = "balanced"
    citations: list[str] = Field(min_length=1)
    # Catch-all flag for ambiguous-high-risk rules. The lenient profile's
    # ``ambiguous_rules_enabled=false`` filters these out entirely.
    ambiguous: bool = False
    # Drives the sparse-input recovery path: when only an ambiguous
    # rule fires, the agent is prompted to elicit these qualifiers
    # before answering the user's surface question.
    missing_qualifiers_hint: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ambiguous_carries_hints(self) -> "Rule":
        if self.ambiguous and not self.missing_qualifiers_hint:
            raise ValueError(
                f"rule {self.id!r}: ``ambiguous: true`` requires at "
                "least one ``missing_qualifiers_hint`` token so the "
                "agent has something to elicit from the user."
            )
        return self


class RulePack(BaseModel):
    """Top-level shape of ``configs/emergency_rules.yaml``."""

    model_config = ConfigDict(extra="forbid")

    rules: list[Rule] = Field(min_length=1)

    @model_validator(mode="after")
    def _no_duplicate_ids(self) -> "RulePack":
        seen: set[str] = set()
        duplicates: list[str] = []
        for rule in self.rules:
            if rule.id in seen and rule.id not in duplicates:
                duplicates.append(rule.id)
            seen.add(rule.id)
        if duplicates:
            raise ValueError(
                f"emergency_rules.yaml contains duplicate rule id(s): "
                f"{duplicates}. Each rule id must be unique — merge "
                "multi-presentation rules using a list for "
                "triggers.primary instead."
            )
        return self


# --- floor enforcement ----------------------------------------------


class FloorViolationError(ValueError):
    """A profile's ``rule_overrides`` violates a rule's floor.

    Raised at config-load. Carries the offending profile / rule / field
    so an operator editing YAML sees exactly which override to remove.
    """


def _override_raises_threshold(
    override: dict[str, Any], rule: Rule
) -> tuple[bool, str | None]:
    """Return (raises, field_name) — does the override loosen the rule?

    "Raising threshold" means making the rule LESS sensitive (fewer
    matches): increasing ``min_qualifier_matches`` or shrinking
    ``any_of`` are the canonical examples.
    """
    if "min_qualifier_matches" in override:
        new = int(override["min_qualifier_matches"])
        if new > rule.triggers.min_qualifier_matches:
            return True, "min_qualifier_matches"
    return False, None


def enforce_floors(
    rules: list[Rule], profiles: dict[SensitivityName, SensitivityProfile]
) -> None:
    """Validate every (profile × rule) override against the rule's floor.

    Raises :class:`FloorViolationError` on the first violation. Designed
    as a load-time check — the runtime matcher trusts validated config.

    Floor rules (plan example aligned — see KTD-E10):

    * ``floor=lenient`` → the rule is fully sealed against overrides
      (anaphylaxis, active SI, ectopic). Any presence in
      ``rule_overrides`` fails, regardless of which profile and what
      direction the override moves the threshold. The evidence base
      for these rules is "false positives cost less than false
      negatives" — no profile gets to relax them.
    * ``floor=balanced`` (default) → any profile may override. Strict
      typically lowers thresholds; lenient may raise them; both are
      permitted because the rule author marked the floor as
      "operator discretion".
    * ``floor=strict`` → no restrictions; equivalent to ``balanced``
      for Phase 2 (kept for the rule schema's future use).

    Unknown rule_ids in any profile's ``rule_overrides`` also fail —
    surfaces dangling references when a rule is removed without
    cleaning up the profile config.
    """
    by_id = {r.id: r for r in rules}
    for profile_name, profile in profiles.items():
        for rule_id, override in profile.rule_overrides.items():
            rule = by_id.get(rule_id)
            if rule is None:
                raise FloorViolationError(
                    f"profile {profile_name!r}: rule_override references "
                    f"unknown rule_id {rule_id!r}"
                )
            if rule.minimum_sensitivity_floor == "lenient":
                # Sealed rule — any override is a violation. Use the
                # field name that triggered the failure when the
                # override is a threshold change so the operator sees
                # what to remove.
                raises, field = _override_raises_threshold(override, rule)
                changed_field = field if raises else next(iter(override))
                raise FloorViolationError(
                    f"profile {profile_name!r}: rule_override touches "
                    f"{changed_field!r} on rule {rule_id!r}, but the "
                    "rule's minimum_sensitivity_floor is 'lenient' — "
                    "this rule is sealed against profile overrides. "
                    "Either remove the override or downgrade the floor "
                    "with an explicit comment in the rule YAML."
                )


# --- runtime apply ---------------------------------------------------


def apply_profile_to_rules(
    rules: list[Rule], profile: SensitivityProfile
) -> list[Rule]:
    """Return the per-turn effective rule list for ``profile``.

    Pre-condition: :func:`enforce_floors` has been called on the
    (rules, all-profiles) pair at load time. This function therefore
    applies overrides without re-validating — the validator is the
    authoritative gate.

    Behavior:

    * ``ambiguous_rules_enabled=False`` drops every ``ambiguous`` rule.
    * Per-rule overrides update ``min_qualifier_matches`` in-place
      (other override fields land in Phase 5+ as the rule pack grows).
    """
    out: list[Rule] = []
    for rule in rules:
        if rule.ambiguous and not profile.ambiguous_rules_enabled:
            continue
        override = profile.rule_overrides.get(rule.id)
        if override is None:
            out.append(rule)
            continue
        # Re-validate via model_validate so future override fields
        # (level escalation, action_key swap) go through pydantic's
        # coercion layer rather than ad-hoc dict updates.
        data = rule.model_dump(mode="python")
        if "min_qualifier_matches" in override:
            data["triggers"]["min_qualifier_matches"] = override[
                "min_qualifier_matches"
            ]
        if "level" in override:
            data["level"] = override["level"]
        out.append(Rule.model_validate(data))
    return out


# --- loader -----------------------------------------------------------


def load_rules(path: Path | None = None) -> list[Rule]:
    """Read ``configs/emergency_rules.yaml`` and return validated rules."""
    target = path or (CONFIGS_DIR / EMERGENCY_RULES_FILENAME)
    if not target.exists():
        return []
    with target.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    pack = RulePack.model_validate(raw)
    return list(pack.rules)


def _check_provider_in_catalog(cfg: EmergencyConfig) -> None:
    """Fail-loud if ``emergency.yaml::provider_id`` is set but unusable.

    Two failure modes:
    1. The id does not exist in ``models.yaml`` at all — likely a typo.
    2. The id resolves to ``kind != "local"`` — the gate requires local
       because its extractor sees raw patient text (PHI; KTD-E1).

    A missing ``provider_id`` (``None``) is valid — the gate falls back
    to "first kind=local in models.yaml" at runtime.

    Raises ``ValueError`` so the app fails at startup rather than
    silently disabling the gate (which would be worse — the audit trail
    would show routine_noop on every turn with no explanation).
    """
    if cfg.provider_id is None:
        return
    from claritymed.stores.models import load_models

    models = load_models()
    matching = [p for p in models.providers if p.id == cfg.provider_id]
    if not matching:
        available_ids = [p.id for p in models.providers]
        raise ValueError(
            f"emergency.yaml::provider_id={cfg.provider_id!r} does not match "
            f"any provider in models.yaml. "
            f"Available ids: {available_ids}"
        )
    if matching[0].kind != "local":
        raise ValueError(
            f"emergency.yaml::provider_id={cfg.provider_id!r} resolves to "
            f"kind={matching[0].kind!r}; the gate requires kind='local' "
            "because its extractor sees raw patient text (PHI; KTD-E1). "
            "Either change provider_id to a local entry or remove it to "
            "fall back to the first kind=local provider in models.yaml."
        )


def load_validated_emergency_config() -> tuple[EmergencyConfig, list[Rule]]:
    """One-call helper: load both config + rules, then enforce floors.

    Returns ``(config, rules)`` ready for the triage service to use.

    Raises:
    * :class:`FloorViolationError` if any profile override violates a
      rule's ``minimum_sensitivity_floor`` — config-time failure so an
      operator can fix before the gate starts denying user turns.
    * ``ValueError`` if ``emergency.yaml::provider_id`` is set but
      does not resolve to a ``kind=local`` entry in ``models.yaml``.
      The docstring on :attr:`EmergencyConfig.provider_id` promised this
      check; it is now backed by code rather than a comment.
    """
    from claritymed.core.emergency.config import load_emergency_config

    cfg = load_emergency_config()
    rules = load_rules()
    enforce_floors(rules, cfg.sensitivity_profiles)
    _check_provider_in_catalog(cfg)
    return cfg, rules
