"""Loader for ``configs/emergency.yaml``.

Phase 1 ships the minimum surface: ``default_sensitivity`` + the four
sensitivity-profile slots.

Historical note: earlier revisions rejected ``default_sensitivity ==
"off"`` at load time as a hard safety invariant (see CLAUDE.md — the
constraint was one of six ``off`` safeguards). That validator was
removed on operator request: ``off`` is now a valid app-wide default.
Operators shipping ``default_sensitivity: off`` should be aware that
``CLARITYMED_FORCE_EMERGENCY_GATE`` still defaults to ``on`` and will
upgrade a resolved ``off`` back to ``lenient`` unless explicitly
overridden; the deterministic disclaimer footer, per-turn
``redflag.gate_disabled`` audit event, and per-user
``off_acknowledged_at`` semantics remain unchanged.

Phase 2 lands per-rule overrides, ambiguous-rule toggles, qualifier
elicitation budgets, and the floor-enforcement validator that rejects
profile overrides which violate a rule's ``minimum_sensitivity_floor``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from claritymed.config import CONFIGS_DIR

# Canonical definition lives in claritymed.core.emergency.schemas.SensitivityName.
# Redefined here because ``from __future__ import annotations`` defers all
# annotation evaluation, which breaks Pydantic's model-build step when a
# cross-module Literal type alias is resolved lazily. Both definitions must
# stay in sync; add new values in schemas.py first, then mirror here.
SensitivityName = Literal["strict", "balanced", "lenient", "off"]

EMERGENCY_CONFIG_FILENAME = "emergency.yaml"


class SensitivityProfile(BaseModel):
    """One entry under ``sensitivity_profiles`` in ``emergency.yaml``.

    Phase 1 only reads ``reply_footer_i18n_key`` (drives the off /
    strict deterministic-suffix path in :class:`AskService`). The other
    fields are accepted now so the YAML schema stabilises before Phase 2
    starts populating them.
    """

    model_config = ConfigDict(extra="forbid")

    rule_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    ambiguous_rules_enabled: bool = True
    qualifier_elicitation_max_rounds: int = Field(default=2, ge=0, le=10)
    reply_footer_i18n_key: str | None = None
    # Only meaningful on the ``off`` profile — names the synthetic level
    # returned without invoking the gate. Always ``routine`` in v1.
    short_circuit_to: Literal["routine"] | None = None


class EmergencyConfig(BaseModel):
    """Top-level shape of ``configs/emergency.yaml``."""

    model_config = ConfigDict(extra="forbid")

    default_sensitivity: SensitivityName = "balanced"
    # Pin the gate's extractor / composer / critical_reply LLMs to one
    # specific ``ProviderConfig.id`` from ``configs/models.yaml``. When
    # unset, ``_provider.build_local_gate_model`` falls back to "first
    # kind=local entry in models.yaml" — adequate for a fresh install
    # but order-dependent, so any deploy with more than one local
    # provider should pin this explicitly. KTD-E1 still applies: the
    # pinned id must be ``kind: local``; the cross-catalog check in
    # :func:`claritymed.core.emergency.rules._check_provider_in_catalog`
    # (called from :func:`load_validated_emergency_config`) enforces this
    # at startup — a typo or a cloud id fails loud rather than silently
    # routing PHI to the cloud or disabling the gate.
    provider_id: str | None = None
    sensitivity_profiles: dict[SensitivityName, SensitivityProfile] = Field(
        default_factory=dict
    )

    @field_validator("default_sensitivity", mode="before")
    @classmethod
    def _coerce_yaml_off_bool(cls, v: Any) -> Any:
        """Rescue the classic YAML 1.1 footgun.

        Unquoted ``off`` / ``on`` / ``yes`` / ``no`` in YAML 1.1 parse
        as booleans. An operator who writes ``default_sensitivity: off``
        without quotes ships a bareword that PyYAML hands us as
        :class:`bool` ``False`` — the Literal validator then rejects it
        with a confusing "Input should be 'strict'…" message. Coerce
        the two YAML-bool spellings back to their string equivalents
        here so the operator's clear intent survives. ``True`` maps to
        ``"on"``, which is not a valid ``SensitivityName`` and will
        still surface a validation error further downstream — that is
        deliberate, since ``on`` never carried a sensible meaning here.
        """
        if v is False:
            return "off"
        if v is True:
            return "on"
        return v


def load_emergency_config(path: Path | None = None) -> EmergencyConfig:
    """Read and validate ``configs/emergency.yaml``.

    Returns a default-constructed :class:`EmergencyConfig` when the
    file is absent so a fresh install with no operator edits still
    boots — the defaults (``balanced``, empty profiles) match the plan.
    """
    target = path or (CONFIGS_DIR / EMERGENCY_CONFIG_FILENAME)
    if not target.exists():
        return EmergencyConfig()
    with target.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return EmergencyConfig.model_validate(raw)
