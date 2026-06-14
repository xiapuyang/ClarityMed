"""Canonical (normalized) in-memory representation of a symptoms dataset.

Every dataset adapter produces a :class:`CanonicalDataset`. The server,
the questions translator, and the differential formatter all operate on
this type — they never touch the dataset's native files. Adding a new
dataset means writing a new adapter; the rest of the codebase is
unchanged.

ID conventions:

* **Evidence ids** keep the dataset's native form (DDXPlus ``E_91``) —
  short and stable enough to be useful as i18n keys.
* **Condition ids** are slugified (``"Spontaneous pneumothorax"`` →
  ``"spontaneous_pneumothorax"``) so they're safe as YAML keys and
  filesystem-portable. The native display name lives in
  :attr:`CanonicalCondition.native_name` for fallback rendering.
* **Algorithm indices** (``CanonicalEvidence.idx`` /
  ``CanonicalCondition.idx``) are the algorithm-internal positions
  carried by ``TypedEnv`` + ``Agent`` (see ``ingest/symptoms/typed_basd.py``).
  Stored alongside the stable id so audit log + payload code can swap
  either way without rebuilding lookups.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from claritymed.core.symptoms.schemas import DatasetSpec, ModelSpec

_SLUG_BAD = re.compile(r"[^a-z0-9]+")
_SLUG_EDGE = re.compile(r"^_+|_+$")


def slugify_condition(name: str) -> str:
    """Slugify a condition display name into a safe id.

    DDXPlus condition keys are display strings like ``"Spontaneous
    pneumothorax"``; this turns them into ``"spontaneous_pneumothorax"``
    so they're safe as YAML keys and stable across re-exports.
    """
    lower = name.lower().strip()
    slug = _SLUG_BAD.sub("_", lower)
    slug = _SLUG_EDGE.sub("", slug)
    return slug or "unknown"


@dataclass(frozen=True)
class CanonicalValue:
    """One value of a categorical or multi-choice evidence."""

    raw: str
    local_idx: int


@dataclass(frozen=True)
class CanonicalEvidence:
    """One evidence (question) the model can ask, in normalized form.

    ``native_question_text`` and ``native_value_labels`` are the corpus's
    own text — used by the question renderer as the fallback after i18n
    keys miss. Empty when the corpus only ships one language.
    """

    id: str
    idx: int
    dtype: str  # "B" | "C" | "M"
    values: list[CanonicalValue] = field(default_factory=list)
    native_question_text: dict[str, str] = field(default_factory=dict)
    native_value_labels: dict[str, dict[str, str]] = field(default_factory=dict)
    is_high_specificity: bool = False

    def value_by_raw(self, raw: str) -> CanonicalValue | None:
        for v in self.values:
            if v.raw == raw:
                return v
        return None

    def raw_values(self) -> list[str]:
        return [v.raw for v in self.values]


@dataclass(frozen=True)
class CanonicalCondition:
    """One pathology the model can diagnose, in normalized form."""

    id: str  # slug
    idx: int
    severity: int
    icd10: str | None = None
    native_name: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalDataset:
    """In-memory normalized dataset, suitable for the server's runtime path.

    ``layout`` is the output of :func:`claritymed.ingest.symptoms.typed_basd.build_layout`
    — we keep the dict shape because that's what ``TypedEnv`` consumes,
    and re-wrapping it would just double the surface area.

    ``severity_vector`` is the dense ``[n_dis]`` float array that
    :func:`interactive_eval` reads at inference time.
    """

    id: str
    evidences: list[CanonicalEvidence]
    conditions: list[CanonicalCondition]
    layout: dict
    severity_vector: np.ndarray

    # Built once at construction so handlers don't pay an O(n) cost per request.
    _evidence_by_id: dict[str, CanonicalEvidence] = field(repr=False)
    _evidence_by_idx: dict[int, CanonicalEvidence] = field(repr=False)
    _condition_by_id: dict[str, CanonicalCondition] = field(repr=False)
    _condition_by_idx: dict[int, CanonicalCondition] = field(repr=False)

    @classmethod
    def build(
        cls,
        *,
        id: str,
        evidences: list[CanonicalEvidence],
        conditions: list[CanonicalCondition],
        layout: dict,
        severity_vector: np.ndarray,
    ) -> "CanonicalDataset":
        """Construct a :class:`CanonicalDataset` and pre-populate lookups."""
        ev_by_id = {ev.id: ev for ev in evidences}
        ev_by_idx = {ev.idx: ev for ev in evidences}
        cond_by_id = {c.id: c for c in conditions}
        cond_by_idx = {c.idx: c for c in conditions}
        if len(ev_by_id) != len(evidences):
            raise ValueError(f"duplicate evidence ids in dataset {id!r}")
        if len(cond_by_id) != len(conditions):
            raise ValueError(f"duplicate condition slugs in dataset {id!r}")
        if len(severity_vector) != len(conditions):
            raise ValueError(
                f"severity_vector length {len(severity_vector)} does not "
                f"match conditions count {len(conditions)} for {id!r}"
            )
        return cls(
            id=id,
            evidences=evidences,
            conditions=conditions,
            layout=layout,
            severity_vector=severity_vector,
            _evidence_by_id=ev_by_id,
            _evidence_by_idx=ev_by_idx,
            _condition_by_id=cond_by_id,
            _condition_by_idx=cond_by_idx,
        )

    def evidence_by_id(self, evidence_id: str) -> CanonicalEvidence:
        try:
            return self._evidence_by_id[evidence_id]
        except KeyError as exc:
            raise KeyError(
                f"evidence {evidence_id!r} not in dataset {self.id!r}"
            ) from exc

    def evidence_by_idx(self, idx: int) -> CanonicalEvidence:
        try:
            return self._evidence_by_idx[idx]
        except KeyError as exc:
            raise KeyError(f"evidence idx {idx} not in dataset {self.id!r}") from exc

    def condition_by_id(self, condition_id: str) -> CanonicalCondition:
        try:
            return self._condition_by_id[condition_id]
        except KeyError as exc:
            raise KeyError(
                f"condition {condition_id!r} not in dataset {self.id!r}"
            ) from exc

    def condition_by_idx(self, idx: int) -> CanonicalCondition:
        try:
            return self._condition_by_idx[idx]
        except KeyError as exc:
            raise KeyError(f"condition idx {idx} not in dataset {self.id!r}") from exc

    @property
    def n_evidences(self) -> int:
        return len(self.evidences)

    @property
    def n_conditions(self) -> int:
        return len(self.conditions)


@dataclass(frozen=True)
class LoadedModel:
    """One materialized model (weights + agent) tied to a :class:`ModelSpec`."""

    spec: ModelSpec
    agent: Any  # torch model with .next_action, .should_stop, .diagnose, .train_step
    manifest: dict


@dataclass(frozen=True)
class LoadedDataset:
    """Server-side bundle: canonical data + per-model_id loaded models.

    Multi-model: one dataset can have multiple checkpoints loaded
    simultaneously (e.g. A/B comparison, shadow inference).
    :meth:`select_model` picks one per the spec's ``model_selection``
    strategy. ``round_robin`` keeps a tiny mutable counter in
    ``_rr_counter``; the dataclass is frozen otherwise so request-time
    code can't accidentally mutate the canonical or model bundles.
    """

    spec: DatasetSpec
    canonical: CanonicalDataset
    models: dict[str, LoadedModel]
    _rr_counter: list[int] = field(default_factory=lambda: [0], repr=False)

    def model_ids(self) -> list[str]:
        return list(self.models.keys())

    def model(self, model_id: str) -> LoadedModel:
        try:
            return self.models[model_id]
        except KeyError as exc:
            raise KeyError(
                f"model {model_id!r} not loaded for dataset {self.spec.id!r}; "
                f"available: {sorted(self.models)}"
            ) from exc

    def select_model(self) -> LoadedModel:
        """Pick a model per ``spec.model_selection``.

        * ``"first"`` — always the primary model (first in the spec list).
        * ``"round_robin"`` — cycles through ``model_ids`` in spec order.
          The counter is in-process; multiple workers each round-robin
          independently (no shared state across uvicorn workers — that's
          fine for v1 load patterns).
        """
        ordered = self.spec.model_ids
        if self.spec.model_selection == "first":
            return self.model(ordered[0])
        # round_robin
        idx = self._rr_counter[0] % len(ordered)
        self._rr_counter[0] = (self._rr_counter[0] + 1) % len(ordered)
        return self.model(ordered[idx])
