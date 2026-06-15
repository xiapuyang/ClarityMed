"""Wire ↔ TypedEnv translation for question + answer payloads (i18n-aware).

Operates entirely on :class:`CanonicalDataset` — no dataset-specific
shape leaks. Two directions:

* :func:`build_question` — render the evidence at ``ev_idx`` as a
  localized :class:`Question`. Text + option labels resolve through
  ``t(key, lang=...)`` using the dataset's ``question_key`` /
  ``value_key`` conventions. Each option's ``value`` field carries the
  raw value id so the plugin can pass it back as ``answer_value``
  without a label re-match round-trip.
* :func:`synth_patient` — translate the user's answer back into the
  ``{bin_pos, cat_val, multi_val}`` dict :meth:`TypedEnv._write` reads.
  Prefers raw value ids when supplied by the client; falls back to
  matching the answer string against every value's localized label.

Categorical option ordering follows the corpus's ``possible-values``
order (i.e. :meth:`CanonicalEvidence.raw_values` insertion order), not
lexicographic sort. Options are capped at ``DatasetSpec.max_options``
(default 12) to keep the TUI picker manageable.
Numeric-categorical evidences (every value parses as a number) are
emitted as :class:`NumericSpec` so the modal renders an Input widget.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from claritymed.core.i18n import t
from claritymed.core.interaction.schemas import NumericSpec, Question, QuestionOption
from claritymed.core.symptoms.datasets import CanonicalDataset, CanonicalEvidence
from claritymed.core.symptoms.schemas import DatasetSpec


class QuestionPayloadError(Exception):
    """Raised when a question or answer cannot be translated.

    Carries an HTTP-style ``status_code`` so :mod:`servers.symptoms.app`
    can map the exception to ``HTTPException`` without pulling fastapi
    into this module. ``500`` for schema-shape bugs (operator's fault);
    ``422`` for malformed user input.
    """

    def __init__(self, detail: str, status_code: int = 422) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def _is_numeric_values(values: list[str]) -> bool:
    if not values:
        return False
    for v in values:
        try:
            float(v)
        except (TypeError, ValueError):
            return False
    return True


def _header(name: str) -> str:
    """Compress an evidence id into a ≤20-char chip label."""
    return name[:20]


def _question_text(
    ev: CanonicalEvidence,
    spec: DatasetSpec,
    language: str,
) -> str:
    """Pick a user-facing question string for an evidence.

    Lookup order:

    1. ``t(spec.question_key(ev.id), lang=language)`` — the per-dataset
       i18n bundle.
    2. ``ev.native_question_text[language]`` then ``[en]`` — the corpus's
       own native text.
    3. ``ev.id`` — last-ditch so we never render an empty question.
    """
    key = spec.question_key(ev.id)
    text = t(key, lang=language)
    if not text or text == key:
        text = (
            ev.native_question_text.get(language)
            or ev.native_question_text.get("en")
            or ev.id
        )
    text = str(text).strip()
    if len(text) > 295:
        text = text[:295]
    # Accept both ASCII "?" and CJK "？" as terminators.
    if not text.endswith(("?", "？")):
        text += "?"
    if len(text) < 4:
        text = f"Q: {text}"
    return text


def _value_label(
    ev: CanonicalEvidence,
    spec: DatasetSpec,
    raw_value: str,
    language: str,
) -> str:
    """Resolve a categorical value to its localized label.

    Falls through to ``ev.native_value_labels[raw][language]`` (the
    corpus's native text) then to the raw value itself.
    """
    key = spec.value_key(ev.id, raw_value)
    label = t(key, lang=language)
    if not label or label == key:
        meaning = ev.native_value_labels.get(raw_value) or {}
        label = meaning.get(language) or meaning.get("en") or raw_value
    label = str(label).strip()
    if not label:
        label = raw_value
    if label.upper() in ("NA", "N/A"):
        label = t("symptoms.ui.not_applicable", lang=language)
    return label[:40]


def _condition_name(
    spec: DatasetSpec,
    canonical: CanonicalDataset,
    condition_slug: str,
    language: str,
) -> str:
    """Resolve a condition slug to its localized display name."""
    key = spec.condition_name_key(condition_slug)
    name = t(key, lang=language)
    if name and name != key:
        return str(name).strip()
    cond = canonical.condition_by_id(condition_slug)
    fallback = (
        cond.native_name.get(language) or cond.native_name.get("en") or condition_slug
    )
    return str(fallback).strip()


def _binary_question(
    ev: CanonicalEvidence, spec: DatasetSpec, language: str
) -> Question:
    yes_label = t(spec.binary_yes_key, lang=language)
    if not yes_label or yes_label == spec.binary_yes_key:
        yes_label = "Yes"
    no_label = t(spec.binary_no_key, lang=language)
    if not no_label or no_label == spec.binary_no_key:
        no_label = "No"
    return Question(
        question=_question_text(ev, spec, language),
        header=_header(ev.id),
        options=[
            QuestionOption(
                label=yes_label[:40], description=yes_label[:40], value="yes"
            ),
            QuestionOption(label=no_label[:40], description=no_label[:40], value="no"),
        ],
    )


def _numeric_question(
    ev: CanonicalEvidence, spec: DatasetSpec, raw_values: list[str], language: str
) -> Question:
    nums = [float(v) for v in raw_values]
    all_int = all(n.is_integer() for n in nums)
    low = int(min(nums)) if all_int else min(nums)
    high = int(max(nums)) if all_int else max(nums)
    try:
        numeric = NumericSpec(min=low, max=high, step=1)
    except Exception as exc:
        raise QuestionPayloadError(
            f"evidence {ev.id!r} numeric range invalid (min={low}, max={high}): {exc}",
            status_code=500,
        ) from exc
    return Question(
        question=_question_text(ev, spec, language),
        header=_header(ev.id),
        options=[],
        numeric=numeric,
    )


def _options_from_values(
    ev: CanonicalEvidence,
    spec: DatasetSpec,
    raw_values: list[str],
    language: str,
) -> list[QuestionOption]:
    """Localize up to spec.max_options values, preserving corpus order."""
    truncated = raw_values[: spec.max_options]
    out: list[QuestionOption] = []
    for raw in truncated:
        label = _value_label(ev, spec, raw, language)
        out.append(QuestionOption(label=label, description=label, value=raw))
    return out


def _categorical_question(
    ev: CanonicalEvidence,
    spec: DatasetSpec,
    *,
    multi: bool,
    language: str,
) -> Question:
    raw_values = ev.raw_values()
    if not multi and _is_numeric_values(raw_values) and raw_values:
        return _numeric_question(ev, spec, raw_values, language)
    options = _options_from_values(ev, spec, raw_values, language)
    if len(options) < 2:
        # The constraint is structural; an evidence with <2 distinct
        # values is a data bug. Fail loud so the operator fixes the
        # schema rather than the model emitting a malformed payload.
        raise QuestionPayloadError(
            f"evidence {ev.id!r} has fewer than 2 values "
            f"({raw_values!r}); cannot render as a categorical Question.",
            status_code=500,
        )
    return Question(
        question=_question_text(ev, spec, language),
        header=_header(ev.id),
        options=options,
        multi_select=multi,
    )


def build_question(
    canonical: CanonicalDataset,
    spec: DatasetSpec,
    ev_idx: int,
    *,
    language: str = "en",
) -> Question:
    """Render the evidence at algorithm-internal ``ev_idx`` as a localized Question."""
    ev = canonical.evidence_by_idx(ev_idx)
    if ev.dtype == "B":
        return _binary_question(ev, spec, language)
    if ev.dtype == "C":
        return _categorical_question(ev, spec, multi=False, language=language)
    if ev.dtype == "M":
        return _categorical_question(ev, spec, multi=True, language=language)
    raise QuestionPayloadError(
        f"evidence {ev.id!r} has unknown dtype {ev.dtype!r}",
        status_code=500,
    )


# --- answer → env patient dict ---------------------------------------------

_BINARY_YES_RAW = {"yes", "true", "y", "1"}
_BINARY_NO_RAW = {"no", "false", "n", "0"}


def _coerce_binary(
    answer: Any,
    spec: DatasetSpec,
    language: str,
    answer_value: str | None,
) -> bool:
    if answer_value is not None:
        token = answer_value.strip().lower()
        if token in _BINARY_YES_RAW:
            return True
        if token in _BINARY_NO_RAW:
            return False
    if isinstance(answer, bool):
        return answer
    if isinstance(answer, (int, float)):
        return bool(answer)
    if isinstance(answer, str):
        token = answer.strip().lower()
        if token in _BINARY_YES_RAW:
            return True
        if token in _BINARY_NO_RAW:
            return False
        yes_label = t(spec.binary_yes_key, lang=language).strip().lower()
        no_label = t(spec.binary_no_key, lang=language).strip().lower()
        if token == yes_label:
            return True
        if token == no_label:
            return False
    raise QuestionPayloadError(
        f"binary evidence requires Yes/No, got {answer!r}",
        status_code=422,
    )


def _match_categorical_value(
    answer: Any,
    answer_value: str | None,
    ev: CanonicalEvidence,
    spec: DatasetSpec,
    language: str,
) -> int:
    """Return the local index for a categorical answer."""
    raw_values = ev.raw_values()
    if answer_value is not None:
        cv = ev.value_by_raw(answer_value)
        if cv is not None:
            return cv.local_idx
    if isinstance(answer, (int, float)) and _is_numeric_values(raw_values):
        arr = np.array([float(v) for v in raw_values])
        return int(np.argmin(np.abs(arr - float(answer))))
    token = str(answer).strip()
    cv = ev.value_by_raw(token)
    if cv is not None:
        return cv.local_idx
    token_lower = token.lower()
    for raw in raw_values:
        label = _value_label(ev, spec, raw, language)
        if label.strip().lower() == token_lower:
            return ev.value_by_raw(raw).local_idx
    raise QuestionPayloadError(
        (
            f"answer {answer!r} did not match any of "
            f"{sorted(raw_values)[:8]}... ({len(raw_values)} total values)"
        ),
        status_code=422,
    )


def synth_patient(
    canonical: CanonicalDataset,
    spec: DatasetSpec,
    ev_idx: int,
    answer: Any,
    *,
    answer_value: str | list[str] | None = None,
    language: str = "en",
) -> dict:
    """Build the dict :meth:`TypedEnv._write` consumes."""
    ev = canonical.evidence_by_idx(ev_idx)
    if ev.dtype == "B":
        present = _coerce_binary(
            answer,
            spec,
            language,
            answer_value if isinstance(answer_value, str) else None,
        )
        return {
            "bin_pos": {ev_idx} if present else set(),
            "cat_val": {},
            "multi_val": {},
        }
    if ev.dtype == "C":
        local = _match_categorical_value(
            answer,
            answer_value if isinstance(answer_value, str) else None,
            ev,
            spec,
            language,
        )
        return {"bin_pos": set(), "cat_val": {ev_idx: local}, "multi_val": {}}
    if ev.dtype == "M":
        labels = answer if isinstance(answer, list) else [answer]
        raw_values = (
            answer_value if isinstance(answer_value, list) else [None] * len(labels)
        )
        if len(raw_values) < len(labels):
            raw_values = list(raw_values) + [None] * (len(labels) - len(raw_values))
        locals_: list[int] = []
        for lbl, raw in zip(labels, raw_values):
            locals_.append(_match_categorical_value(lbl, raw, ev, spec, language))
        return {
            "bin_pos": set(),
            "cat_val": {},
            "multi_val": {ev_idx: locals_},
        }
    raise QuestionPayloadError(
        f"evidence at idx {ev_idx} has unknown dtype {ev.dtype!r}",
        status_code=500,
    )


def localize_condition(
    canonical: CanonicalDataset,
    spec: DatasetSpec,
    condition_slug: str,
    language: str,
) -> str:
    """Public helper for the differential formatter to render a condition name."""
    return _condition_name(spec, canonical, condition_slug, language)
