"""Pydantic contracts shared across orchestrator, tools, stores, and evals.

These types are the system's *spine*. Renaming or retyping a field here breaks
every downstream module, so the rule is: in v1, fields freeze. Optional fields
may be added later; renames and type changes are not permitted.

All result objects are ``frozen=True`` and ``extra="forbid"`` — an LLM that
hallucinates an extra field gets a ValidationError, not silent acceptance.
"""

from claritymed.core.schemas.account import Account, Role
from claritymed.core.schemas.answer import (
    Citation,
    Disclaimer,
    GroundedAnswer,
    RedFlag,
)
from claritymed.core.schemas.evals import EvalsConfig
from claritymed.core.schemas.lab import LabFlag, LabPanel, LabValue, ReferenceRange
from claritymed.core.schemas.models import (
    ModelsConfig,
    ProviderConfig,
    ProviderKind,
    ThinkingLevel,
)
from claritymed.core.schemas.patient import (
    PASSIVE_PROFILE_FIELDS,
    PROACTIVE_PROFILE_FIELDS,
    Allergy,
    Condition,
    LongitudinalRecord,
    MaritalStatus,
    Medication,
    Patient,
    Profile,
    Solicitation,
    solicitation_for,
)
from claritymed.core.schemas.request import RequestContext
from claritymed.core.schemas.uncertainty import (
    UncertaintyResult,
    UncertaintySource,
    UncertaintyType,
)
from claritymed.core.schemas.vision import (
    DiseaseVisionModel,
    ModelMetadata,
    PredictionSet,
    QualityReport,
)

__all__ = [
    "Account",
    "Allergy",
    "Citation",
    "Condition",
    "Disclaimer",
    "DiseaseVisionModel",
    "EvalsConfig",
    "GroundedAnswer",
    "LabFlag",
    "LabPanel",
    "LabValue",
    "LongitudinalRecord",
    "MaritalStatus",
    "Medication",
    "ModelMetadata",
    "ModelsConfig",
    "PASSIVE_PROFILE_FIELDS",
    "PROACTIVE_PROFILE_FIELDS",
    "Patient",
    "PredictionSet",
    "Profile",
    "ProviderConfig",
    "ProviderKind",
    "Solicitation",
    "QualityReport",
    "RedFlag",
    "ReferenceRange",
    "RequestContext",
    "Role",
    "ThinkingLevel",
    "UncertaintyResult",
    "UncertaintySource",
    "UncertaintyType",
    "solicitation_for",
]
