"""Differential ranking + cancel/cap-hit outcome formatting.

Implements the §HLD outcome decision table from the disease-prediction
plan. Server-side only — no safety prose, no tier label; the LLM
composes those downstream from raw severity ints.

Two flavours of output:

* :func:`format_differential` — full top-K differential at done / cap-hit.
* :func:`format_cancel_outcome` — for cancellations and cap-hits, also
  computes ``partial_confidence`` (top-3 mass) and the
  ``severity_override`` flag for the "low-confidence but a severity ≤2
  disease shows up" case.

Operates on :class:`CanonicalDataset` — no dataset-specific shape leaks.
"""

from __future__ import annotations

import numpy as np

from claritymed.core.symptoms.datasets import LoadedDataset
from claritymed.servers.symptoms.questions import localize_condition
from claritymed.servers.symptoms.state import SubSessionState
from claritymed.servers.symptoms.wire import DifferentialRow, EvidenceCollectedRow

# Display threshold for the differential — the demo's interactive_eval
# uses 0.01 as the "include in pred" cutoff; mirror it so DDF1 the
# benchmark reports matches what we surface.
DIFFERENTIAL_PROB_THRESHOLD = 0.01

# Top-K rendered to the LLM. Five gives the model enough alternatives
# without burying the leading diagnosis.
TOP_K = 5

# Severity ≤ this is "concerning" — used by the severity_override flag
# during partial-result rendering. Matches the post_process tier check
# in the plugin (Critical=1, Urgent=2).
SEVERITY_OVERRIDE_THRESHOLD = 2

# Minimum probability for a low-severity disease to trigger the
# severity_override flag. Below this the model has effectively ruled
# it out — surfacing it would be alarmist.
SEVERITY_OVERRIDE_PROB = 0.1


def _evidence_rows(sub: SubSessionState) -> list[EvidenceCollectedRow]:
    return [EvidenceCollectedRow(**row) for row in sub.evidence_collected]


def _topk_rows(
    ds: LoadedDataset, probs: np.ndarray, language: str
) -> list[DifferentialRow]:
    """Return the top-K conditions above the display threshold."""
    if probs.ndim == 2:
        probs = probs[0]
    eligible = np.where(probs >= DIFFERENTIAL_PROB_THRESHOLD)[0]
    ordered = sorted(eligible.tolist(), key=lambda i: -probs[i])
    rows: list[DifferentialRow] = []
    for idx in ordered[:TOP_K]:
        cond = ds.canonical.condition_by_idx(idx)
        rows.append(
            DifferentialRow(
                condition_id=cond.id,
                condition_idx=cond.idx,
                condition_name=localize_condition(
                    ds.canonical, ds.spec, cond.id, language
                ),
                probability=float(probs[idx]),
                severity=cond.severity,
                icd10=cond.icd10,
            )
        )
    return rows


def format_differential(
    ds: LoadedDataset,
    sub: SubSessionState,
    probs: np.ndarray,
    language: str | None = None,
) -> tuple[list[DifferentialRow], list[EvidenceCollectedRow]]:
    """Format a completed differential — top-K + collected evidence."""
    lang = language or sub.language
    return _topk_rows(ds, probs, lang), _evidence_rows(sub)


def partial_confidence(probs: np.ndarray) -> float:
    """Compute the top-3 probability mass (sum of top-3 probs)."""
    if probs.ndim == 2:
        probs = probs[0]
    top3 = np.sort(probs)[-3:]
    return float(top3.sum())


def severity_override(ds: LoadedDataset, probs: np.ndarray) -> tuple[bool, int | None]:
    """Detect the low-confidence-but-concerning-pattern branch."""
    if probs.ndim == 2:
        probs = probs[0]
    sev = ds.canonical.severity_vector
    mask = (sev <= SEVERITY_OVERRIDE_THRESHOLD) & (probs > SEVERITY_OVERRIDE_PROB)
    if not mask.any():
        return False, None
    seen_severities = sev[mask].astype(int)
    return True, int(seen_severities.min())


def format_cancel_outcome(
    ds: LoadedDataset,
    sub: SubSessionState,
    probs: np.ndarray,
    language: str | None = None,
) -> dict:
    """Compute partial-result fields for ``DELETE /sessions`` and cap-hit."""
    lang = language or sub.language
    confidence = partial_confidence(probs)
    meets_threshold = confidence >= ds.spec.partial_min_confidence
    override_fired, max_low_sev = severity_override(ds, probs)
    show_partial = meets_threshold or override_fired
    partial_rows = _topk_rows(ds, probs, lang) if show_partial else []
    return {
        "partial_differential": partial_rows,
        "evidence_collected": _evidence_rows(sub),
        "turn_count": sub.turn_count,
        "partial_confidence": confidence,
        "meets_confidence_threshold": meets_threshold,
        "severity_override": override_fired,
        "max_low_severity_seen": max_low_sev,
    }
