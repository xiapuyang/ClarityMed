"""Tool-args pydantic schemas.

The seven LLM-callable ingest tools each have a frozen args contract here.
``ToolDispatcher`` validates against the right schema before any approval
check; ``ApprovalModal`` reads ``model_json_schema()`` to render the
per-field ``modify_args`` form; the headless CLI parses JSON straight into
these models.  One source of truth for each tool's argument shape — no
drift between LLM, modal, and CLI.

Why ``extra="forbid"``: an LLM hallucinating a field gets a ValidationError
at the boundary, not a silent dropped value inside the tool body.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from claritymed.core.schemas.records import ExtractedLab
from claritymed.core.schemas.patient import AllergySeverity, AllergySource


class AttachmentRef(BaseModel):
    """Reference to a CAS blob from inside a tool's args.

    Distinct from ``records.Attachment``: that one is what lands in the
    manifest (full metadata + ocr_status). This one is what the LLM emits
    in a tool call — sha + filename are mandatory, mime/size are populated
    by the tool body from the blob itself (so the LLM can't lie about the
    file type to dodge an approval rule).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    filename: str = Field(min_length=1, max_length=255)


class SaveRecordArgs(BaseModel):
    """``save_record`` — one PHI event written to records/<category>/<slug>/."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    category: str = Field(
        min_length=1,
        max_length=64,
        examples=["checkups", "labs", "imaging", "vaccinations"],
    )
    kind: str = Field(
        min_length=1,
        max_length=64,
        examples=["checkup", "lab_report", "imaging_study", "vaccination"],
    )
    # See ``records.Manifest.event_date`` — same rename to avoid shadowing
    # ``datetime.date`` with a same-named field.
    event_date: date | None = Field(
        default=None,
        alias="date",
        examples=["2026-03-12"],
    )
    title: str = Field(
        min_length=1,
        max_length=256,
        examples=["Annual checkup", "Lipid panel — Dr. Chen"],
    )
    provider: str | None = Field(default=None, max_length=128)
    attachments: list[AttachmentRef] = Field(default_factory=list)
    extracted_labs: list[ExtractedLab] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list, examples=[["routine", "fasting"]])
    notes: str | None = None


class SaveMedicationArgs(BaseModel):
    """``save_medication`` — one medication upserted to profile.db.

    ``onset_date`` is when the patient started; ``end_date`` is when it was
    discontinued. Null end_date is the canonical "currently taking" encoding —
    null is *not* "unknown", so leave both null when the source is silent
    rather than inferring "they probably still take it".
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=1,
        max_length=128,
        examples=["metformin", "lisinopril"],
    )
    code: str | None = Field(default=None, max_length=64)
    dose: str | None = Field(
        default=None, max_length=64, examples=["500 mg", "10 mg", "1 puff"]
    )
    frequency: str | None = Field(
        default=None,
        max_length=64,
        examples=["twice daily", "every 8 hours", "as needed"],
    )
    onset_date: date | None = Field(default=None, examples=["2024-01-15"])
    end_date: date | None = None


class SaveAllergyArgs(BaseModel):
    """``save_allergy`` — one allergy upserted to profile.db.

    Allergy severity and source reuse the existing patient-schema literals
    so the tool can't smuggle in a value that ``ProfileStore.add_allergy``
    refuses; one validation path instead of two.

    ``onset_date`` is when the allergy was first noticed; ``end_date`` (rare)
    means the allergy has been resolved — most allergies stay null here.
    """

    model_config = ConfigDict(extra="forbid")

    substance: str = Field(
        min_length=1, max_length=128, examples=["penicillin", "peanut"]
    )
    severity: AllergySeverity
    source: AllergySource
    onset_date: date | None = None
    end_date: date | None = None


class SaveConditionArgs(BaseModel):
    """``save_condition`` — one condition upserted to profile.db.

    ``end_date`` is the resolved date; null means still ongoing. Duration is
    derived in app code from ``onset_date`` + ``end_date``; we deliberately do
    not let the LLM persist a separate free-text duration string, since a
    structured date pair is easier to reason over.
    """

    model_config = ConfigDict(extra="forbid")

    display: str = Field(
        min_length=1,
        max_length=128,
        examples=["type 2 diabetes", "asthma", "hypertension"],
    )
    code: str | None = Field(default=None, max_length=64)
    onset_date: date | None = None
    end_date: date | None = None


# Fields that ``update_profile_field`` is allowed to touch. Hardcoded rather
# than read from ``Profile.model_fields`` because the LLM should never be
# trusted to update structural metadata (``created_at``, ``user_id``).
# The two tiers (proactive vs passive) are intentionally listed together —
# the gate that "do not solicit passive fields" lives in the prompt, not in
# the schema; both tiers are storable when the user volunteers a value.
ProfileField = Literal[
    "sex",
    "weight_kg",
    "height_cm",
    "birth_date",
    "residence",
    "birthplace",
    "marital_status",
    "has_children",
    "current_occupation",
    "past_occupations",
]


class UpdateProfileFieldArgs(BaseModel):
    """``update_profile_field`` — one Profile column write.

    ``value`` is intentionally permissive (str/float/bool/None) at the
    boundary; the tool body coerces to the target column type via the
    Patient schema so the LLM can pass ``"60"`` instead of ``60.0`` without
    rejection.
    """

    model_config = ConfigDict(extra="forbid")

    field: ProfileField
    value: str | float | bool | None = Field(
        default=None,
        examples=[72.5, "Berlin", True, "1990-04-22"],
    )


class SaveToLibraryArgs(BaseModel):
    """``save_to_library`` — one library entry under library/<category>/<slug>/.

    ``public=True`` is the only path that flips the underlying chunk's
    ``can_cloud`` flag. Default ``False`` keeps the user's curated material
    local-only unless they explicitly opt in.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        min_length=1,
        max_length=256,
        examples=[
            "2024 Hypertension Guideline",
            "Harrison's — Chapter 271",
        ],
    )
    attachments: list[AttachmentRef] = Field(default_factory=list)
    authors: list[str] = Field(default_factory=list, examples=[["Jameson", "Loscalzo"]])
    year: int | None = Field(default=None, ge=1800, le=2200, examples=[2024])
    tags: list[str] = Field(default_factory=list, examples=[["textbook"]])
    public: bool = False


class DeleteRecordArgs(BaseModel):
    """``delete_record`` — remove records/<category>/<slug>/ + Qdrant chunks.

    ``confirm_kind`` is the second-channel anti-mistake check: the LLM must
    pass the same ``kind`` field the manifest already declares, so a
    misrouted ``delete_record`` for the wrong record_path raises rather
    than silently destroying data.
    """

    model_config = ConfigDict(extra="forbid")

    record_path: str = Field(min_length=1, examples=["checkups/2026-03-12-annual"])
    confirm_kind: str = Field(
        min_length=1, max_length=64, examples=["checkup", "lab_report"]
    )


# Registry mapping tool name → args model. Used by ToolDispatcher to look up
# the right validator without N if-branches. Keeping this dict in sync with
# the seven tool implementations is the single drift point — one new tool =
# one new model + one new dict entry.
TOOL_ARG_SCHEMAS: dict[str, type[BaseModel]] = {
    "save_record": SaveRecordArgs,
    "save_medication": SaveMedicationArgs,
    "save_allergy": SaveAllergyArgs,
    "save_condition": SaveConditionArgs,
    "update_profile_field": UpdateProfileFieldArgs,
    "save_to_library": SaveToLibraryArgs,
    "delete_record": DeleteRecordArgs,
}
