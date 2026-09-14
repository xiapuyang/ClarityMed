"""Apply one user's ``FactsBundle`` to ``profile.db``.

Per R12, equivalence checks read the **live** ``profile.db`` (not
``rows.jsonl``) — that's the rule that lets the importer skip per-batch
transactions: a crash mid-batch leaves the DB partially populated, and
the next resume re-derives equivalence from the live rows. *This
guarantee depends on R12 staying deterministic and stable.* If a future
change moves equivalence to embedding similarity or fuzzy matching,
per-batch transactions become necessary.

Profile merge (mirrors plan §"Profile-merge dance" Key Decision):

* Read existing ``Profile`` from store → ``model_dump`` to dict.
* Overlay only the template-provided keys (skipping ``None`` values so
  they don't overwrite live data).
* Short-circuit: if the merged dict equals the current dict, skip
  ``upsert_profile`` — avoids bumping ``update_time`` on a replay.
* Validate via ``model_validate`` (not ``model_copy``) so date-shaped
  strings from YAML coerce correctly. See CLAUDE.md "Pydantic
  model_copy vs model_validate."

Per-fact ``row_id`` formula (mirrors plan §"Fact ID for state tracking"
+ "Fact row_id canonical-key uses JSON encoding"):

    row_id = f"fact:{user_id}:{fact_kind}:{sha8}"
    canonical_key = json.dumps([fact_kind, *key_components], ...)
    sha8 = sha256(canonical_key)[:8]

JSON encoding (not ``|`` join) — robust against free-text fields that
contain ``|`` (e.g. ``substance="shellfish | crab"``).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from pydantic import ValidationError

from claritymed.core.schemas.patient import (
    Allergy,
    Condition,
    Medication,
    Profile,
)
from claritymed.ingest.records.template_schema import FactsBundle
from claritymed.stores.profile import ProfileStore

logger = logging.getLogger(__name__)

FactKind = Literal["profile", "allergy", "condition", "medication"]
FactStatus = Literal["done", "skipped", "error"]
PROFILE_ROW_ID_SENTINEL = "0"  # profile is a singleton; one row id per user.


@dataclass(frozen=True)
class FactApplyResult:
    """Outcome of one fact-write attempt.

    The orchestrator translates ``status="skipped"`` into the
    ``skipped(already_exists)`` row state and turns errors into
    per-row ``error`` rows that get retried on resume.
    """

    fact_kind: FactKind
    row_id: str
    status: FactStatus
    error_detail: str | None = None


def apply_facts(user_id: str, facts: FactsBundle) -> list[FactApplyResult]:
    """Apply every fact in the bundle. Returns one result per attempted fact.

    Order: profile (singleton, merge-on-write) → allergies → conditions
    → medications. The orchestrator can interleave row_id append in
    ``rows.jsonl`` so resume sees per-fact progress.

    No per-batch transaction; see module docstring.
    """
    results: list[FactApplyResult] = []
    store = ProfileStore(user_id)

    if facts.profile is not None:
        results.append(_apply_profile(store, user_id, facts.profile))

    for allergy in facts.allergies:
        results.append(_apply_allergy(store, user_id, allergy))

    for condition in facts.conditions:
        results.append(_apply_condition(store, user_id, condition))

    for medication in facts.medications:
        results.append(_apply_medication(store, user_id, medication))

    return results


# --- profile ----------------------------------------------------------


def _apply_profile(
    store: ProfileStore,
    user_id: str,
    template_profile: dict[str, Any],
) -> FactApplyResult:
    row_id = compute_fact_row_id(user_id, "profile", [PROFILE_ROW_ID_SENTINEL])
    current = store.get_profile()
    current_dict = current.model_dump(mode="python") if current else {}

    # Only overlay explicitly-provided keys with non-None values. ``None``
    # values in the template are skill noise (it dumped a default rather
    # than omitting the key) and must not overwrite live data.
    merged = dict(current_dict)
    for k, v in template_profile.items():
        if v is not None:
            merged[k] = v

    # Short-circuit no-op: replay-after-crash with identical content must
    # not bump update_time.
    if current is not None and merged == current_dict:
        return FactApplyResult("profile", row_id, "done", None)

    try:
        updated = Profile.model_validate(merged)
    except ValidationError as exc:
        return FactApplyResult("profile", row_id, "error", f"profile_validation:{exc}")

    try:
        store.upsert_profile(updated, owner_user_id=user_id)
    except Exception as exc:
        return FactApplyResult("profile", row_id, "error", repr(exc))

    return FactApplyResult("profile", row_id, "done", None)


# --- allergy ----------------------------------------------------------


def _apply_allergy(
    store: ProfileStore,
    user_id: str,
    allergy: Allergy,
) -> FactApplyResult:
    key = [allergy.substance.casefold()]
    row_id = compute_fact_row_id(user_id, "allergy", key)
    existing = store.list_allergies()
    if any(a.substance.casefold() == key[0] for a in existing):
        return FactApplyResult("allergy", row_id, "skipped", "already_exists")
    try:
        store.add_allergy(allergy, owner_user_id=user_id)
    except Exception as exc:
        return FactApplyResult("allergy", row_id, "error", repr(exc))
    return FactApplyResult("allergy", row_id, "done", None)


# --- condition --------------------------------------------------------


def _apply_condition(
    store: ProfileStore,
    user_id: str,
    condition: Condition,
) -> FactApplyResult:
    key = [condition.display.casefold(), _iso_or_none(condition.onset_date)]
    row_id = compute_fact_row_id(user_id, "condition", key)
    existing = store.list_conditions()
    if any(
        c.display.casefold() == key[0] and c.onset_date == condition.onset_date
        for c in existing
    ):
        return FactApplyResult("condition", row_id, "skipped", "already_exists")
    try:
        store.add_condition(condition, owner_user_id=user_id)
    except Exception as exc:
        return FactApplyResult("condition", row_id, "error", repr(exc))
    return FactApplyResult("condition", row_id, "done", None)


# --- medication -------------------------------------------------------


def _apply_medication(
    store: ProfileStore,
    user_id: str,
    medication: Medication,
) -> FactApplyResult:
    key = [medication.display.casefold(), _iso_or_none(medication.onset_date)]
    row_id = compute_fact_row_id(user_id, "medication", key)
    existing = store.list_medications()
    if any(
        m.display.casefold() == key[0] and m.onset_date == medication.onset_date
        for m in existing
    ):
        return FactApplyResult("medication", row_id, "skipped", "already_exists")
    try:
        store.add_medication(medication, owner_user_id=user_id)
    except Exception as exc:
        return FactApplyResult("medication", row_id, "error", repr(exc))
    return FactApplyResult("medication", row_id, "done", None)


# --- row_id helper ----------------------------------------------------


def compute_fact_row_id(
    user_id: str,
    fact_kind: FactKind,
    key_components: list[Any],
) -> str:
    """Stable ``fact:<uid>:<kind>:<sha8>`` row id.

    JSON encoding (not ``|`` join) is robust against free-text fields
    containing ``|`` — e.g. ``substance="shellfish | crab"`` and
    ``substance="shellfish"`` produce distinct sha8s under JSON but
    would collide under ``"shellfish | crab" | "" → "shellfish|crab|"``.
    """
    canonical_key = json.dumps(
        [fact_kind, *key_components],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    sha8 = hashlib.sha256(canonical_key.encode("utf-8")).hexdigest()[:8]
    return f"fact:{user_id}:{fact_kind}:{sha8}"


def _iso_or_none(d: date | None) -> str | None:
    return d.isoformat() if d is not None else None
