"""Lab panel value contracts.

Every numeric field carries an explicit ``unit`` because LLMs must never
infer one. ``ReferenceRange.source`` tracks whether the bounds came from the
patient's own report (preferred), a LOINC default, or nowhere — interpretive
code refuses to flag anything as out-of-range when ``source == "unknown"``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

LabFlag = Literal[
    "normal",
    "low",
    "high",
    "critical_low",
    "critical_high",
    "unknown",
]
RangeSource = Literal["report_provided", "loinc_default", "unknown"]


class ReferenceRange(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    low: float | None = None
    high: float | None = None
    text: str | None = None
    source: RangeSource

    @model_validator(mode="after")
    def _low_le_high(self) -> "ReferenceRange":
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError(f"low {self.low} > high {self.high}")
        return self


class LabValue(BaseModel):
    """One lab measurement. ``value`` may be numeric or qualitative."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    value: float | str
    unit: str = Field(min_length=1)
    loinc: str | None = None
    reference_range: ReferenceRange | None = None
    flag: LabFlag | None = None
    ocr_confidence: float = Field(ge=0.0, le=1.0)


class LabPanel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    user_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,32}$")
    collected_at: datetime
    values: list[LabValue] = Field(min_length=1)
    source_doc: str | None = None
