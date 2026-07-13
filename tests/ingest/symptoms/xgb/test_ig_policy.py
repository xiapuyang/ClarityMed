"""Tests for the IG question-selection policy.

Uses a synthetic ``classifier_predict`` callable so we don't need a real
XGBoost model to verify IG picks the right feature. Test cases target
the two canonical failure modes typed-BASD's greedy policy hits:

* Discriminative vs noise: IG must pick the feature that separates the
  classes, not the one with the highest marginal.
* Sequential dependence: after the first discriminative feature is
  answered, a previously-neutral feature may become discriminative.
"""

from __future__ import annotations

import numpy as np

from claritymed.ingest.symptoms.xgb.ig_policy import (
    entropy,
    information_gain_per_column,
    information_gain_per_evidence,
    pick_next_evidence,
)


def test_entropy_zero_on_certain_distribution():
    # EPS-clamped log leaves a residual ~3e-8; treat as effectively zero.
    assert entropy(np.array([1.0, 0.0])) < 1e-6
    assert entropy(np.array([0.0, 1.0])) < 1e-6


def test_entropy_maximal_on_uniform_binary():
    # Uniform binary → 1 bit.
    assert abs(entropy(np.array([0.5, 0.5])) - 1.0) < 1e-6


def test_entropy_batched_along_last_axis():
    probs = np.array([[1.0, 0.0], [0.5, 0.5]])
    ent = entropy(probs, axis=-1)
    assert ent.shape == (2,)
    assert ent[0] < 1e-6
    assert abs(ent[1] - 1.0) < 1e-6


def test_information_gain_picks_discriminative_over_noise():
    """Two features, two classes: feature 0 is discriminative, feature 1 is noise.

    Initial state is 50/50 (max entropy). Knowing feature 0 flips to
    high-confidence class 1 when on. Feature 1 leaves the posterior
    unchanged. IG must pick feature 0.
    """

    def clf(x: np.ndarray) -> np.ndarray:
        out = np.zeros((len(x), 2), dtype=np.float32)
        for i, row in enumerate(x):
            if row[0] > 0.5:
                out[i] = [0.02, 0.98]  # feature 0 on → class 1
            else:
                out[i] = [0.5, 0.5]  # feature 0 off → uncertain
        return out

    state = np.zeros(2, dtype=np.float32)
    marginals = np.array([0.5, 0.5], dtype=np.float32)
    ig = information_gain_per_column(
        state, np.array([0, 1]), clf, marginals, smoothing=0.0
    )
    assert ig[0] > ig[1]
    assert ig[1] < 1e-5  # feature 1 is pure noise → IG≈0


def test_information_gain_zero_when_classifier_ignores_feature():
    """If the classifier posterior doesn't change with the feature, IG=0."""

    def clf(x: np.ndarray) -> np.ndarray:
        # Always 50/50 regardless of state.
        return np.tile([0.5, 0.5], (len(x), 1)).astype(np.float32)

    state = np.zeros(3, dtype=np.float32)
    marginals = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    ig = information_gain_per_column(
        state, np.array([0, 1, 2]), clf, marginals, smoothing=0.0
    )
    assert np.allclose(ig, 0.0, atol=1e-5)


def test_pick_next_evidence_returns_argmax():
    """Simple end-to-end: two binary evidences, one discriminative."""

    def clf(x: np.ndarray) -> np.ndarray:
        out = np.zeros((len(x), 2), dtype=np.float32)
        for i, row in enumerate(x):
            out[i] = [0.02, 0.98] if row[0] > 0.5 else [0.98, 0.02]
        return out

    state = np.zeros(2, dtype=np.float32)
    ev_col_index = [[0], [1]]  # 2 binary evidences
    asked = np.array([False, False])
    marginals = np.array([0.5, 0.5], dtype=np.float32)
    picked = pick_next_evidence(state, asked, ev_col_index, clf, marginals)
    assert picked == 0


def test_pick_next_evidence_skips_asked():
    """Asked evidences must be excluded from the argmax."""

    def clf(x: np.ndarray) -> np.ndarray:
        out = np.zeros((len(x), 2), dtype=np.float32)
        for i, row in enumerate(x):
            out[i] = [0.02, 0.98] if row[0] > 0.5 else [0.98, 0.02]
        return out

    state = np.zeros(2, dtype=np.float32)
    ev_col_index = [[0], [1]]
    asked = np.array([True, False])  # feature 0 already asked
    marginals = np.array([0.5, 0.5], dtype=np.float32)
    picked = pick_next_evidence(state, asked, ev_col_index, clf, marginals)
    assert picked == 1


def test_pick_next_evidence_all_asked_returns_sentinel():
    def clf(x: np.ndarray) -> np.ndarray:
        return np.tile([0.5, 0.5], (len(x), 1)).astype(np.float32)

    state = np.zeros(2, dtype=np.float32)
    ev_col_index = [[0], [1]]
    asked = np.array([True, True])
    marginals = np.array([0.5, 0.5], dtype=np.float32)
    assert pick_next_evidence(state, asked, ev_col_index, clf, marginals) == -1


def test_sequential_dependence_second_pick_flips_after_first_answer():
    """3 features: feature 0 is the gate; features 1 and 2 are only
    discriminative once feature 0 is known.

    Classifier: if state[0]=0 (gate not opened), posterior is 50/50
    regardless of features 1 and 2. Once state[0]=1, feature 1 becomes
    strongly discriminative (class 0 when off, class 1 when on).

    Verify: at initial state (all zeros), IG should pick feature 0 (the
    only feature that changes the posterior). After feature 0 = 1 is
    revealed, IG picks feature 1 next.
    """

    def clf(x: np.ndarray) -> np.ndarray:
        out = np.zeros((len(x), 2), dtype=np.float32)
        for i, row in enumerate(x):
            if row[0] < 0.5:
                out[i] = [0.5, 0.5]  # gate closed → uninformative
            elif row[1] > 0.5:
                out[i] = [0.02, 0.98]  # gate open, feature 1 = 1 → class 1
            else:
                out[i] = [0.98, 0.02]  # gate open, feature 1 = 0 → class 0
        return out

    ev_col_index = [[0], [1], [2]]
    marginals = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    # Initial pick: the only feature that reduces entropy AT ALL from
    # 50/50 is feature 0 (the gate). Features 1 and 2 don't change the
    # posterior when state[0]=0.
    state = np.zeros(3, dtype=np.float32)
    asked = np.zeros(3, dtype=bool)
    first = pick_next_evidence(state, asked, ev_col_index, clf, marginals)
    assert first == 0

    # After feature 0 = 1 revealed, feature 1 becomes the discriminator.
    state[0] = 1.0
    asked[0] = True
    second = pick_next_evidence(state, asked, ev_col_index, clf, marginals)
    assert second == 1


def test_evidence_level_ig_averages_column_ig_for_multi_value():
    """For a 2-column evidence, the evidence-level IG is the *mean* of its
    column IGs — sum would over-reward wide evidences (see module docstring)."""

    def clf(x: np.ndarray) -> np.ndarray:
        out = np.zeros((len(x), 2), dtype=np.float32)
        for i, row in enumerate(x):
            score = 0.98 if row[0] > 0.5 else 0.02
            score = max(0.02, min(0.98, score + (0.5 if row[1] > 0.5 else 0)))
            out[i] = [1 - score, score]
        return out

    state = np.zeros(3, dtype=np.float32)
    # Evidence 0 has two columns (cols 0 and 1); evidence 1 is a single-column noise feature.
    ev_col_index = [[0, 1], [2]]
    asked = np.zeros(2, dtype=bool)
    marginals = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    scores = information_gain_per_evidence(state, asked, ev_col_index, clf, marginals)
    per_col = information_gain_per_column(
        state, np.array([0, 1]), clf, marginals, smoothing=0.0
    )
    assert abs(scores[0] - (per_col[0] + per_col[1]) / 2) < 1e-5


def test_projected_ig_zero_when_feature_only_moves_within_other_bucket():
    """With a projection ``{[0], [1, 2]}`` on a 3-class classifier where the
    feature swings the mass ONLY within the ``[1, 2]`` bucket (class 1 vs
    class 2), the raw-distribution IG is positive but the projected IG is
    ≈ 0 — the projected posterior doesn't change with the feature.

    This is exactly the property that lets a 49-class model produce a low
    interaction length when we only care about ``{Pne, Inf, Other}``: it
    stops paying for questions that distinguish diseases inside ``Other``.
    """

    def clf(x: np.ndarray) -> np.ndarray:
        out = np.zeros((len(x), 3), dtype=np.float32)
        for i, row in enumerate(x):
            # Class 0's share stays fixed at 0.4 regardless of feature.
            # Feature moves mass around inside the (1, 2) bucket by
            # different amounts, so raw entropy DIFFERS between yes / no
            # branches (previously I used a symmetric swap where raw
            # entropies were equal by permutation → raw IG=0, invalidating
            # the test). Now: yes → concentrated (low entropy), no → flat
            # (high entropy). Projected mass on Other stays 0.6 → proj IG=0.
            if row[0] > 0.5:
                out[i] = [0.4, 0.55, 0.05]  # concentrated within Other
            else:
                out[i] = [0.4, 0.35, 0.25]  # flatter within Other
        return out

    from claritymed.ingest.symptoms.xgb.ig_policy import (
        ClassProjection,
        information_gain_per_column,
    )

    state = np.zeros(1, dtype=np.float32)
    marginals = np.array([0.5], dtype=np.float32)
    raw_ig = information_gain_per_column(
        state, np.array([0]), clf, marginals, smoothing=0.0
    )
    projection: ClassProjection = [[0], [1, 2]]
    proj_ig = information_gain_per_column(
        state,
        np.array([0]),
        clf,
        marginals,
        smoothing=0.0,
        class_projection=projection,
    )
    assert raw_ig[0] > 1e-3, "raw IG must be positive — feature moves 3-class posterior"
    assert abs(proj_ig[0]) < 1e-5, (
        "projected IG must be ≈ 0 — feature only redistributes within Other bucket"
    )


def test_projected_ig_high_when_feature_separates_target_from_other():
    """Complement to the previous test: a feature that pushes the target
    posterior asymmetrically (uncertain baseline → confident yes-branch)
    must score high under projection.

    Uses an asymmetric mapping (not just a permutation) so raw entropies
    genuinely differ between yes and no branches — otherwise the two
    entropy values cancel to IG=0 regardless of the posterior shift.
    """

    def clf(x: np.ndarray) -> np.ndarray:
        out = np.zeros((len(x), 3), dtype=np.float32)
        for i, row in enumerate(x):
            if row[0] > 0.5:
                out[i] = [0.98, 0.01, 0.01]  # confident target
            else:
                out[i] = [0.50, 0.25, 0.25]  # uncertain (baseline)
        return out

    from claritymed.ingest.symptoms.xgb.ig_policy import (
        ClassProjection,
        information_gain_per_column,
    )

    state = np.zeros(1, dtype=np.float32)
    marginals = np.array([0.5], dtype=np.float32)
    projection: ClassProjection = [[0], [1, 2]]
    proj_ig = information_gain_per_column(
        state,
        np.array([0]),
        clf,
        marginals,
        smoothing=0.0,
        class_projection=projection,
    )
    # Projected: yes-branch H([0.98, 0.02]) ≈ 0.14; no-branch
    # H([0.50, 0.50]) = 1.0; baseline (state=0) is the no branch → 1.0.
    # IG = 1.0 − 0.5·0.14 − 0.5·1.0 ≈ 0.43 — nontrivial, well above 0.
    assert proj_ig[0] > 0.3


def test_evidence_level_ig_wide_evidence_does_not_dominate_narrow_binary():
    """Mean aggregation prevents a wide evidence with tiny per-column IG
    from outranking a narrow binary evidence with one strong column.

    Under the old sum aggregation this test would fail: the wide 10-column
    evidence's small per-column IGs would add up above the single strong
    binary column. Under mean, the strong single-column evidence wins.
    """

    def clf(x: np.ndarray) -> np.ndarray:
        out = np.zeros((len(x), 2), dtype=np.float32)
        for i, row in enumerate(x):
            if row[0] > 0.5:
                out[i] = [0.02, 0.98]  # binary col 0 → strong signal
            else:
                # wide-evidence columns 1..10 each nudge the posterior a tiny bit
                nudge = 0.05 * row[1:11].sum()
                p = min(0.5 + nudge, 0.7)
                out[i] = [1 - p, p]
        return out

    state = np.zeros(11, dtype=np.float32)
    ev_col_index = [[0], list(range(1, 11))]  # narrow binary + wide 10-col
    asked = np.zeros(2, dtype=bool)
    marginals = np.full(11, 0.5, dtype=np.float32)
    picked = pick_next_evidence(state, asked, ev_col_index, clf, marginals)
    assert picked == 0, "narrow binary with strong signal must beat wide evidence"


def test_smoothing_shifts_marginal():
    """A rare-positive column (marginal ≈ 0) gets its IG effectively
    zeroed under no smoothing; with smoothing > 0 the p=0 term is
    revived so its IG can rank above a purely-non-discriminative one."""

    def clf(x: np.ndarray) -> np.ndarray:
        out = np.zeros((len(x), 2), dtype=np.float32)
        for i, row in enumerate(x):
            if row[0] > 0.5:
                out[i] = [0.02, 0.98]
            else:
                out[i] = [0.5, 0.5]  # uncertain when feature off
        return out

    state = np.zeros(1, dtype=np.float32)
    marginals = np.array([0.0], dtype=np.float32)  # never seen in training
    ig_no_smooth = information_gain_per_column(
        state, np.array([0]), clf, marginals, smoothing=0.0
    )
    ig_smoothed = information_gain_per_column(
        state, np.array([0]), clf, marginals, smoothing=0.1
    )
    # With p1=0 the "yes" branch weight is 0, IG only reflects the p0
    # branch's entropy. Adding smoothing shifts weight to the yes branch,
    # revealing the class-1-collapsing signal.
    assert ig_smoothed[0] > ig_no_smooth[0]
