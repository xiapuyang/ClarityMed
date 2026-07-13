"""Information-gain question-selection policy for the XGBoost agent.

Given a partial XGBoost feature vector, an evidence → columns mapping, a
classifier that yields ``predict_proba``, and marginals estimating
``P(col=1 | state)`` for each column, pick the next evidence to ask by
maximising expected classification-entropy reduction.

The per-column IG for an unasked column ``f`` is::

    IG(f | s) = H(class | s) - p1 · H(class | s ∪ {f=1}) - p0 · H(class | s ∪ {f=0})

where ``p1 = clamp(marginals[f] + smoothing, 0, 1)`` and ``p0 = 1 - p1``.
Additive ``smoothing`` guards against the classifier committing on rare
positives whose marginals are near 0 in the training population.

The evidence-level score is the **mean** of its columns' IGs. Summing
per-column IG systematically inflates wide evidences (E_55 pain-location
has ~160 columns; E_133 same) over narrow binary evidences (E_77 sputum,
E_88 fatigue) — even when the wide evidence's average per-option signal
is smaller. For binary evidences with one column, mean == the single
column IG. For categorical / multi it's an independent-column
approximation of the per-option expected entropy reduction, which is
cheap and defensible without paying the ``2^K`` cost of exact joint IG
on wide blocks.

This module is dataset-agnostic: it takes a ``classifier_predict``
callable rather than any specific XGBoost API, so tests can pass a
synthetic 2-class predictor without loading a real model.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

EPS = 1e-9


ClassifierPredict = Callable[[np.ndarray], np.ndarray]
"""``(batch, n_features) → (batch, n_classes)`` posterior probabilities."""


ClassProjection = list[list[int]]
"""Buckets of class indices to sum into a lower-dim distribution.

``projection[k]`` is the list of raw class indices whose probability mass
gets summed into projected class ``k``. Example on a 49-class DDXPlus
classifier where we care only about ``{Pneumonia, Influenza, Other}``::

    projection = [[pne_idx], [inf_idx], [other 47 indices]]

Applied inside :func:`_project` before entropy — makes the IG policy
optimize the entropy of the projected distribution rather than the raw
one, so it stops wasting queries on distinctions we don't care about.
"""


def _project(probs: np.ndarray, projection: ClassProjection) -> np.ndarray:
    """Sum ``probs`` across each bucket in ``projection``.

    ``probs`` may be ``(n_classes,)`` or ``(batch, n_classes)``. Returns the
    same leading shape with the last axis collapsed to ``len(projection)``.
    Each bucket's mass is guaranteed to lie in ``[0, 1]`` when input probs
    sum to 1 and bucket indices are disjoint (both invariants held by the
    caller; not re-checked at hot-path scoring time).
    """
    if probs.ndim == 1:
        out = np.empty(len(projection), dtype=probs.dtype)
        for k, bucket in enumerate(projection):
            out[k] = probs[bucket].sum()
        return out
    out2 = np.empty((probs.shape[0], len(projection)), dtype=probs.dtype)
    for k, bucket in enumerate(projection):
        out2[:, k] = probs[:, bucket].sum(axis=1)
    return out2


def entropy(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return Shannon entropy (base 2) along ``axis`` of a probability array.

    ``probs`` is clamped to ``[EPS, 1]`` before the log so an all-mass-on-one
    distribution returns 0 instead of NaN.
    """
    p = np.clip(probs, EPS, 1.0)
    return -np.sum(p * np.log2(p), axis=axis)


def information_gain_per_column(
    x_state: np.ndarray,
    unasked_columns: np.ndarray,
    classifier_predict: ClassifierPredict,
    marginals: np.ndarray,
    smoothing: float = 0.0,
    class_projection: ClassProjection | None = None,
) -> np.ndarray:
    """Batched per-column IG for a single partial state.

    ``x_state`` shape ``(n_features,)``. ``unasked_columns`` is an int
    ndarray of the column indices to score. Returns an array of the same
    length as ``unasked_columns``.

    The two hypothetical states — ``state ∪ {col=1}`` and
    ``state ∪ {col=0}`` — are stacked into one big ``(2·U, n_features)``
    matrix and scored in a single classifier call so cost is dominated
    by one XGBoost inference rather than ``2·U`` micro-calls.

    When ``class_projection`` is set, entropy is computed over the
    projected distribution instead of the raw classifier output — the
    IG policy optimizes distinguishing the projected buckets rather than
    every raw class. See :data:`ClassProjection` for the format. Under a
    ``{target1, target2, Other}`` projection on a 49-class classifier,
    features that only distinguish two 'Other' diseases from each other
    score IG ≈ 0 (the projected posterior doesn't change), and features
    that separate targets from Other score high. Directly cuts the query
    budget on tasks like "surface P(pneumonia) + P(influenza) honestly
    but stop asking the moment the target vs non-target decision is
    settled".
    """
    u = len(unasked_columns)
    if u == 0:
        return np.zeros(0, dtype=np.float32)

    # Baseline entropy at current state.
    probs_now = classifier_predict(x_state[np.newaxis, :])[0]
    if class_projection is not None:
        probs_now = _project(probs_now, class_projection)
    h_now = float(entropy(probs_now))

    # Build (2·U, n_features): first U rows are state_yes[col=1],
    # last U rows are state_no[col=0]. Restoring the original slot value
    # in state_no is unnecessary because it's already whatever x_state
    # carries — the col value is only overwritten in state_yes.
    batch = np.tile(x_state, (2 * u, 1))
    batch[np.arange(u), unasked_columns] = 1.0
    batch[np.arange(u, 2 * u), unasked_columns] = 0.0
    probs = classifier_predict(batch)
    if class_projection is not None:
        probs = _project(probs, class_projection)
    ent = entropy(probs, axis=-1)
    h_yes = ent[:u]
    h_no = ent[u:]

    p1 = np.clip(marginals[unasked_columns] + smoothing, 0.0, 1.0)
    p0 = 1.0 - p1
    return (h_now - p1 * h_yes - p0 * h_no).astype(np.float32)


def information_gain_per_evidence(
    x_state: np.ndarray,
    asked_mask: np.ndarray,
    ev_col_index: list[list[int]],
    classifier_predict: ClassifierPredict,
    marginals: np.ndarray,
    smoothing: float = 0.0,
    class_projection: ClassProjection | None = None,
) -> np.ndarray:
    """Aggregate column-level IG into per-evidence scores.

    Returns a float array of shape ``(n_ev,)``. Asked evidences are set
    to ``-inf`` so the caller's ``argmax`` never re-picks them without
    special-casing.
    """
    n_ev = len(ev_col_index)
    scores = np.full(n_ev, -np.inf, dtype=np.float32)

    # Gather every unasked column across every unasked evidence into one
    # flat array so we only run classifier_predict once per turn.
    flat_cols: list[int] = []
    per_ev_slices: list[tuple[int, int]] = [(0, 0)] * n_ev
    for ev_i, cols in enumerate(ev_col_index):
        if asked_mask[ev_i]:
            continue
        start = len(flat_cols)
        flat_cols.extend(cols)
        per_ev_slices[ev_i] = (start, len(flat_cols))
    if not flat_cols:
        return scores

    per_col_ig = information_gain_per_column(
        x_state,
        np.asarray(flat_cols, dtype=np.int64),
        classifier_predict,
        marginals,
        smoothing=smoothing,
        class_projection=class_projection,
    )
    for ev_i, (start, end) in enumerate(per_ev_slices):
        if start == end:
            continue
        # Mean of per-column IG across the evidence's columns. For binary
        # (K=1) this reduces to the exact 2-term IG. For cat/multi it's
        # the per-option average — the independent-column approximation
        # described in the module docstring, chosen over sum so a wide
        # evidence must earn its rank on per-option signal rather than
        # accumulating tiny signals across K columns.
        scores[ev_i] = float(per_col_ig[start:end].mean())
    return scores


def pick_next_evidence(
    x_state: np.ndarray,
    asked_mask: np.ndarray,
    ev_col_index: list[list[int]],
    classifier_predict: ClassifierPredict,
    marginals: np.ndarray,
    smoothing: float = 0.0,
    class_projection: ClassProjection | None = None,
) -> int:
    """Return the evidence index that maximises IG. -1 when all asked.

    Ties are broken by evidence index (numpy ``argmax`` on ``-inf`` for
    asked positions). If every evidence is already asked the caller must
    fall back to another policy — that's the ``-1`` sentinel.
    """
    scores = information_gain_per_evidence(
        x_state,
        asked_mask,
        ev_col_index,
        classifier_predict,
        marginals,
        smoothing=smoothing,
        class_projection=class_projection,
    )
    if np.all(np.isneginf(scores)):
        return -1
    return int(np.argmax(scores))
