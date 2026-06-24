"""Loader for ``configs/emergency.yaml``.

Phase 1 ships the minimum surface: ``default_sensitivity`` + the four
sensitivity-profile slots. The loader runs a single hard validator at
load time — ``default_sensitivity == "off"`` is rejected because that
would let an operator quietly disable the safety net app-wide. ``off``
is reachable only as a per-user choice (and gated by
``CLARITYMED_FORCE_EMERGENCY_GATE`` on top).

Phase 2 lands per-rule overrides, ambiguous-rule toggles, qualifier
elicitation budgets, and the floor-enforcement validator that rejects
profile overrides which violate a rule's ``minimum_sensitivity_floor``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

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

    @model_validator(mode="after")
    def _reject_default_off(self) -> "EmergencyConfig":
        if self.default_sensitivity == "off":
            raise ValueError(
                "emergency.yaml: default_sensitivity cannot be 'off'. "
                "Disabling the emergency triage gate app-wide would "
                "leave every user unprotected without their consent. "
                "'off' is a per-user choice (and is further gated by "
                "the CLARITYMED_FORCE_EMERGENCY_GATE env override)."
            )
        return self


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
