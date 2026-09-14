"""Binary clinical metrics for cross-dataset drift evaluation.

Multi-class classifiers trained on different datasets cannot be compared
directly when their label sets differ (BUSI 3-class ``{benign, malignant,
normal}`` vs breast_us_kaggle 2-class ``{benign, malignant}``; future
Kermany 2-class vs RSNA 2-class+bbox). Cross-eval here collapses every
prediction + ground-truth label into a binary clinical task — positive
vs not — driven by a caller-supplied ``positive_labels`` frozenset. The
clinically critical metric (recall for cancer / disease) becomes
apples-to-apples across heterogeneous label spaces.

Why this lives in ``core/vision/`` rather than
``ingest/vision/forge/``: the cross-dataset bench loads *runtime*
artifacts (``manifest.json`` + ``weights.pt`` from
``~/.claritymed/models/vision/``), not forge phase artifacts, and the
metric primitive is read-only against numpy arrays. The forge
``Task.evaluate()`` path computes an in-distribution breakdown that the
framework's feasibility scoring depends on; this sibling primitive
answers the orthogonal cross-distribution question without disturbing
that contract.

See ``docs/plans/2026-06-17-001-feat-cross-dataset-drift-bench-plan.md``
for the broader bench design.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# ``sklearn.metrics.roc_auc_score`` is available transitively via
# qdrant-client / llama-index-core. If a future dep cleanup drops the
# transitive, the ImportError here is preferable to silently swapping
# in a hand-rolled AUC that could mishandle ties or degenerate label
# sets — promote sklearn to a direct dep at that point.
from sklearn.metrics import roc_auc_score


class BinaryMetrics(BaseModel):
    """One bench cell — the ``(model, eval_dataset, threshold)`` outcome.

    ``specificity`` and ``auc`` are ``None`` for degenerate single-class
    eval sets (no true negatives possible / sklearn cannot score AUC on
    one class). The Markdown renderer turns ``None`` into ``"N/A"``; the
    JSON output keeps the null so downstream consumers can detect the
    case.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sensitivity: float = Field(ge=0.0, le=1.0)
    specificity: float | None = Field(default=None, ge=0.0, le=1.0)
    accuracy: float = Field(ge=0.0, le=1.0)
    auc: float | None = Field(default=None, ge=0.0, le=1.0)
    n_total: int = Field(ge=0)
    n_positive: int = Field(ge=0)


def binary_clinical_metrics(
    *,
    gt_labels: list[int],
    probs: np.ndarray,
    label_tuple: tuple[str, ...],
    positive_labels: frozenset[str],
    threshold: float | None = None,
) -> BinaryMetrics:
    """Compute binary-collapsed clinical metrics for one cross-dataset cell.

    Args:
        gt_labels: Ground-truth class indices, one per sample. Indices
            are positional into ``label_tuple``.
        probs: Predicted probabilities, shape ``(N, len(label_tuple))``.
            Caller is responsible for whatever softmax / temperature
            scaling the model's tuned inference declares — this function
            never touches raw logits.
        label_tuple: The eval dataset's label tuple (positional order
            matches ``probs`` columns and ``gt_labels`` indices).
        positive_labels: Subset of ``label_tuple`` collapsed to the
            binary positive class. Empty intersection raises.
        threshold: If ``None``, predicts via strict argmax over the full
            multi-class softmax then collapses (the model's natural
            decision — what a vanilla classifier outputs at inference).
            If given, predicts ``sum(probs[positive_cols]) >= threshold``
            — what a deployed model with a tuned operating point does.

    Returns:
        A frozen :class:`BinaryMetrics` carrying sensitivity, specificity,
        accuracy, AUC, and sample counts.

    Raises:
        ValueError: When ``positive_labels`` contains a label absent
            from ``label_tuple`` — fail-loud so a typo'd registry entry
            never silently scores against the wrong class.
    """
    missing = positive_labels - set(label_tuple)
    if missing:
        raise ValueError(
            f"positive_labels {sorted(missing)!r} not in label_tuple "
            f"{label_tuple!r}; check the registry entry's positive_labels "
            f"matches the eval dataset's labels"
        )

    positive_idx = np.array(
        [i for i, lbl in enumerate(label_tuple) if lbl in positive_labels],
        dtype=np.int64,
    )

    gt = np.asarray(gt_labels, dtype=np.int64)
    gt_positive = np.isin(gt, positive_idx)
    prob_positive = probs[:, positive_idx].sum(axis=1)

    if threshold is None:
        # Natural-threshold path: argmax over the full label space, then
        # collapse to binary. Matches what a model without a tuned
        # operating point outputs at deployment.
        preds = probs.argmax(axis=1)
        pred_positive = np.isin(preds, positive_idx)
    else:
        pred_positive = prob_positive >= threshold

    tp = int((gt_positive & pred_positive).sum())
    fn = int((gt_positive & ~pred_positive).sum())
    tn = int((~gt_positive & ~pred_positive).sum())
    fp = int((~gt_positive & pred_positive).sum())

    n_total = int(gt.shape[0])
    n_positive = int(gt_positive.sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # Specificity needs at least one true negative to be defined; if
    # the eval set is all-positive, surface ``None`` rather than 0 (the
    # latter would be indistinguishable from "model misses every negative").
    specificity: float | None = tn / (tn + fp) if (tn + fp) > 0 else None
    accuracy = (tp + tn) / max(n_total, 1)

    # AUC is undefined when the eval set is single-class (sklearn would
    # raise). Surface as ``None`` so the bench Markdown can render
    # ``"N/A"`` — never silently substitute 0 or 1, which would be a
    # misleading floor / ceiling on the matrix.
    if 0 < n_positive < n_total:
        auc: float | None = float(roc_auc_score(gt_positive, prob_positive))
    else:
        auc = None

    return BinaryMetrics(
        sensitivity=float(sensitivity),
        specificity=float(specificity) if specificity is not None else None,
        accuracy=float(accuracy),
        auc=auc,
        n_total=n_total,
        n_positive=n_positive,
    )


__all__ = ["BinaryMetrics", "binary_clinical_metrics"]
