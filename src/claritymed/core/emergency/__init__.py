"""Emergency triage gate — deterministic pre-step before the agent loop.

Public surface:

* :class:`EmergencyAssessment` — the pre-step's structured output;
  consumed by ``AskService`` to decide critical short-circuit vs.
  context injection, and to populate ``GroundedAnswer.red_flags[]``.
* :class:`EmergencyTriage` — service facade. Always invoked once per
  clinical ``AskService.handle`` turn before the agent runs.
* :func:`resolve_sensitivity` — CLI → user-setting → app-default →
  ``balanced`` fallback resolver, honoring the
  ``CLARITYMED_FORCE_EMERGENCY_GATE`` env override.

Plan: ``docs/plans/2026-06-23-001-feat-emergency-triage-gate-plan.md``.

Strict invariant (KTD-E5): downstream tools never import from this
module. The gate is one-way; ``symptoms_plugin`` and any future tool
retain their own independent safety mechanisms.
"""

from claritymed.core.emergency.composer import (
    Composer,
    LLMComposer,
    build_default_composer,
)
from claritymed.core.emergency.critical_reply import (
    CriticalReplyComposer,
    CriticalReplyResult,
    build_default_critical_reply,
)
from claritymed.core.emergency.extractor import (
    Extractor,
    LLMExtractor,
    build_default_extractor,
)
from claritymed.core.emergency.schemas import (
    EmergencyAssessment,
    ExtractedSymptoms,
    MatchedRule,
)
from claritymed.core.emergency.sensitivity import (
    ResolvedSensitivity,
    resolve_sensitivity,
)
from claritymed.core.emergency.service import EmergencyTriage

__all__ = [
    "Composer",
    "CriticalReplyComposer",
    "CriticalReplyResult",
    "EmergencyAssessment",
    "EmergencyTriage",
    "Extractor",
    "ExtractedSymptoms",
    "LLMComposer",
    "LLMExtractor",
    "MatchedRule",
    "ResolvedSensitivity",
    "build_default_composer",
    "build_default_critical_reply",
    "build_default_extractor",
    "resolve_sensitivity",
]
