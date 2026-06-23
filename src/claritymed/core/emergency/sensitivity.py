"""Sensitivity resolver.

The triage gate needs to know — for every clinical turn — which
sensitivity profile to use. Plan §"Sensitivity Profiles" → "Resolution
order" fixes the priority:

    1. CLI override        (per-request, e.g. --emergency-sensitivity)
    2. Per-user setting    (Account.emergency.sensitivity)
    3. App default         (configs/emergency.yaml::default_sensitivity)
    4. Hard fallback       ("balanced")

On top of that there is the deploy-time master switch
``CLARITYMED_FORCE_EMERGENCY_GATE`` (see :func:`claritymed.config.emergency_gate_force_on`).
When that is on (default), the resolver refuses to honor a resolved
``"off"`` and downgrades it to ``"lenient"``. The intent is encoded:
the user said "I don't want noise" → we keep precision high, but we
do not leave them unprotected.

This function returns a :class:`ResolvedSensitivity` carrying both the
**requested** value (what the resolver picked before the env override)
and the **effective** value (after the override). The two-field shape
lets the audit / disclaimer layers report the downgrade transparently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from claritymed.config import emergency_gate_force_on
from claritymed.core.emergency.schemas import SensitivityName

Reason = Literal[
    "cli_override",
    "user_preference",
    "app_default",
    "hard_fallback",
]

# How the env-override downgrade behaves. "lenient" mirrors the plan
# (KTD-E10 spirit): respect the user's "less noise" intent but refuse
# to fully disable the gate.
_OFF_DOWNGRADE: SensitivityName = "lenient"


@dataclass(frozen=True)
class ResolvedSensitivity:
    """Result of the resolver — both the picked + the effective value.

    ``requested`` is what the priority chain produced; ``effective`` is
    what the runtime uses after the env-override downgrade. They are
    equal except when ``requested == "off"`` and the env override is on,
    in which case ``effective`` is the downgraded value.
    """

    requested: SensitivityName
    effective: SensitivityName
    reason: Reason
    env_override_applied: bool


def resolve_sensitivity(
    *,
    cli_override: SensitivityName | None,
    user_pref: SensitivityName | None,
    app_default: SensitivityName,
) -> ResolvedSensitivity:
    """Resolve the effective sensitivity for one turn.

    Callers pass three layers explicitly so the resolver stays a pure
    function — no env reads beyond ``emergency_gate_force_on``. That
    keeps the priority order auditable in one place and the function
    unit-testable without monkeypatching globals.
    """
    if cli_override is not None:
        picked: SensitivityName = cli_override
        reason: Reason = "cli_override"
    elif user_pref is not None:
        picked = user_pref
        reason = "user_preference"
    elif app_default is not None:
        picked = app_default
        reason = "app_default"
    else:
        # Defensive: app_default is typed as required, but a config
        # mishap could surface ``None`` here.
        picked = "balanced"
        reason = "hard_fallback"

    if picked == "off" and emergency_gate_force_on():
        return ResolvedSensitivity(
            requested=picked,
            effective=_OFF_DOWNGRADE,
            reason=reason,
            env_override_applied=True,
        )
    return ResolvedSensitivity(
        requested=picked,
        effective=picked,
        reason=reason,
        env_override_applied=False,
    )
