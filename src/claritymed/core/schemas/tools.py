"""Tool-args pydantic schemas.

The seven LLM-callable ingest tools each have a frozen args contract here.
``ToolDispatcher`` validates against the right schema before any approval
check; ``ApprovalModal`` reads ``model_json_schema()`` to render the
per-field ``modify_args`` form; the headless CLI parses JSON straight into
these models.  One source of truth for each tool's argument shape — no
drift between LLM, modal, and CLI.

Why ``extra="forbid"``: an LLM hallucinating a field gets a ValidationError
at the boundary, not a silent dropped value inside the tool body.

LLM output tolerance
--------------------
``ToolArgsBase`` (shared base for all seven schemas) strips extra keys whose
value is ``None`` before Pydantic validates.  This handles a common small-model
behaviour: passing explicit ``null`` for every optional arg the model doesn't
intend to set, which would otherwise trip ``extra="forbid"``.

For tools that use a ``field``/``value`` wrapper (currently only
``update_profile_field``), call ``_normalize_field_value(data, valid_keys)``
inside a ``model_validator(mode="before")`` to reshape the shorthand form
``{"birth_date": "1989-01-01"}`` → ``{"field": "birth_date", "value": "1989-01-01"}``.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from claritymed.core.schemas.records import ExtractedLab
from claritymed.core.schemas.patient import AllergySeverity, AllergySource


def _coerce_partial_date(v: Any) -> Any:
    """Expand year-only / year-month strings into full ISO dates.

    LLMs frequently emit ``"2020"`` or ``"2020-05"`` when the user said
    "since 2020" / "2020 年起" — they know the year but not the day. The
    persisted contract is still ``date``; we fill the missing components
    with ``-01-01`` / ``-01`` so the row gets stored, then downstream code
    treats those as approximations like any other day-of-year default.
    Anything outside the two recognised partial shapes (including ``None``
    and full ISO strings) passes through unchanged for the stock ``date``
    parser to handle or reject.
    """
    if not isinstance(v, str):
        return v
    s = v.strip()
    if len(s) == 4 and s.isdigit():
        return f"{s}-01-01"
    if len(s) == 7 and s[4] == "-" and s[:4].isdigit() and s[5:].isdigit():
        return f"{s}-01"
    return v


PartialDate = Annotated[date | None, BeforeValidator(_coerce_partial_date)]


def _normalize_field_value(
    data: object,
    valid_keys: frozenset[str],
) -> object:
    """Reshape ``{field_name: value}`` → ``{"field": field_name, "value": value}``.

    Detects a lone key from ``valid_keys`` and promotes it to the canonical
    ``field``/``value`` structure.  No-op when ``"field"`` is already present
    or when the input is not a plain dict.  Reusable by any tool that wraps a
    domain-specific field name in a generic ``field``/``value`` pair.
    """
    if not isinstance(data, dict) or "field" in data:
        return data
    matches = [k for k in data if k in valid_keys]
    if len(matches) == 1:
        return {"field": matches[0], "value": data[matches[0]]}
    return data


class ToolArgsBase(BaseModel):
    """Shared base for all tool arg schemas.

    Strips extra keys with ``None`` values before Pydantic validates the
    schema.  Small LLMs often emit every optional argument explicitly as
    ``null``; without this, ``extra="forbid"`` would reject calls that are
    semantically correct.
    """

    @model_validator(mode="before")
    @classmethod
    def _strip_null_extras(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        known: set[str] = set(cls.model_fields)
        for f in cls.model_fields.values():
            alias: Any = getattr(f, "alias", None)
            if alias:
                known.add(alias)
        return {k: v for k, v in data.items() if k in known or v is not None}


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


class SaveRecordArgs(ToolArgsBase):
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
    event_date: PartialDate = Field(
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
    attachments: list[AttachmentRef] = Field(
        default_factory=list,
        description=(
            "Array of {sha256, filename} objects for files referenced by this "
            "event. Pass a JSON array literal — `[]` for none, NOT the string "
            "`'[]'`."
        ),
    )
    extracted_labs: list[ExtractedLab] = Field(
        default_factory=list,
        description=(
            "Array of structured lab values parsed from the source. Pass `[]` "
            "(JSON array, not the string `'[]'`) when nothing was extracted."
        ),
    )
    tags: list[str] = Field(
        default_factory=list,
        description=(
            "Array of short topical labels. Pass a JSON array — `[]` for none, "
            'NOT the string `\'[]\'`; `["a", "b"]`, NOT `\'["a", "b"]\'`.'
        ),
        examples=[["routine", "fasting"]],
    )
    notes: str | None = None


class SaveMedicationArgs(ToolArgsBase):
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
    onset_date: PartialDate = Field(default=None, examples=["2024-01-15"])
    end_date: PartialDate = None


class SaveAllergyArgs(ToolArgsBase):
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
    onset_date: PartialDate = None
    end_date: PartialDate = None


class SaveConditionArgs(ToolArgsBase):
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
    onset_date: PartialDate = None
    end_date: PartialDate = None


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


_PROFILE_FIELD_KEYS: frozenset[str] = frozenset(
    {
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
    }
)


class UpdateProfileFieldArgs(ToolArgsBase):
    """``update_profile_field`` — one Profile column write.

    ``value`` is intentionally permissive (str/float/bool/None) at the
    boundary; the tool body coerces to the target column type via the
    Patient schema so the LLM can pass ``"60"`` instead of ``60.0`` without
    rejection.

    Accepts the shorthand form ``{"birth_date": "1989-01-01"}`` in addition
    to the canonical ``{"field": "birth_date", "value": "1989-01-01"}`` via
    ``_normalize_field_value``.
    """

    model_config = ConfigDict(extra="forbid")

    field: ProfileField
    value: str | float | bool | None = Field(
        default=None,
        examples=[72.5, "Berlin", True, "1990-04-22"],
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: object) -> object:
        return _normalize_field_value(data, _PROFILE_FIELD_KEYS)


class SaveToLibraryArgs(ToolArgsBase):
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
    attachments: list[AttachmentRef] = Field(
        default_factory=list,
        description=(
            "Array of {sha256, filename} objects for the uploaded document(s). "
            "Pass a JSON array literal — `[]` for none, NOT the string `'[]'`."
        ),
    )
    authors: list[str] = Field(
        default_factory=list,
        description=(
            "Array of author name strings. Pass a JSON array — `[]` for none, "
            "NOT the string `'[]'`."
        ),
        examples=[["Jameson", "Loscalzo"]],
    )
    year: int | None = Field(default=None, ge=1800, le=2200, examples=[2024])
    tags: list[str] = Field(
        default_factory=list,
        description=(
            "Array of short topic labels. Pass a JSON array — `[]` for none, "
            "NOT the string `'[]'`."
        ),
        examples=[["textbook"]],
    )
    public: bool = False


class DeleteRecordArgs(ToolArgsBase):
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
