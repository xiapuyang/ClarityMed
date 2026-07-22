"""XGBoost agent for the symptom-prediction pipeline.

Bundles the classifier (patho head), the ev-marginals regressor (sym
head equivalent), and the IG question-selection policy behind the same
Agent surface :mod:`typed_basd` exposes so the FastAPI server can
dispatch between algorithms via ``ModelSpec.algorithm_module`` without
knowing which underlying implementation it holds.

Save/load via joblib. The checkpoint is a single pickle carrying:

* ``classifier`` — trained :class:`xgboost.XGBClassifier` (or a
  :class:`sklearn.calibration.CalibratedClassifierCV` wrapper when
  ``calibration != "none"``).
* ``ev_marginals`` — fitted marginals estimator (or ``None`` for the
  global-mean fallback path).
* ``global_marginals`` — training-set column means, used when
  ``ev_marginals`` is ``None`` or when a partial state has no signal
  the marginals model can condition on.
* ``feature_columns`` — the ordered column list the classifier was
  trained on; the adapter cross-checks this against a live-schema
  derivation and fails loud on mismatch (KTD-6 feature-order chain).
* ``thres`` / ``mode`` / ``temp`` — server overrides parity-block with
  the typed-BASD agent so :func:`apply_model_overrides` works
  algorithm-agnostically. ``temp`` is currently ignored by XGBoost (its
  posterior is calibrated at training time via
  :class:`CalibratedClassifierCV`).

Stop semantics differ from typed-BASD (KTD-D3 overload):

* typed-BASD: ``max symptom prob < thres`` → stop asking (heuristic mode).
* XGBoost: ``max class prob > thres`` → stop asking (confidence-based).

Both use the same YAML field for operator brevity; the docstring on
:class:`ModelSpec.stop_thres` calls the overload out.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from claritymed.ingest.symptoms.xgb.encoding import (
    encode_typed_state_batch,
    evidence_column_index,
)
from claritymed.ingest.symptoms.xgb.ig_policy import (
    ClassProjection,
    information_gain_per_evidence,
    pick_next_evidence,
)

# Server + interactive_eval read these attributes on the agent — keep
# names in lockstep with typed_basd.Agent for the ``apply_model_overrides``
# override site (loader.py) to work uniformly.
_MODE_CONFIDENCE = "confidence"

# Stop-policy variants when a class projection is active. See
# ``docs/solutions/…-xgb-49class-projected-ig.md`` for the derivation.
#
# * ``proj_max``: mirrors the raw ``max``-based stop, but on the projected
#   distribution — ``max(P_proj) > thres`` fires. Symmetric across buckets.
# * ``proj_variant_a``: asymmetric —
#     stop when any target-bucket probability exceeds ``target_thres``
#     OR the aggregate "Other" bucket exceeds ``other_thres``.
#   The other-bucket threshold is deliberately higher: confirming "not a
#   target" is a broader claim and premature stop hides late-rising
#   targets. Both thresholds tune independently.
_STOP_PROJ_MAX = "proj_max"
_STOP_PROJ_VARIANT_A = "proj_variant_a"
# * ``proj_target_sum``: stop when the SUM of target-bucket probabilities
#   exceeds ``target_thres`` OR the Other bucket exceeds ``other_thres``.
#   Complements ``proj_variant_a`` (which uses per-target max): the sum
#   variant fires when the model is confident it's a target class but
#   can't yet resolve Pne vs Flu individually — the "minimize questions
#   × maximize target recall" utility. Recommended default for
#   Pne+Flu two-target datasets where the frontend groups both under
#   "respiratory infection likely".
_STOP_PROJ_TARGET_SUM = "proj_target_sum"
_VARIANT_A_TARGET_THRES = 0.60
_VARIANT_A_OTHER_THRES = 0.85
_TARGET_SUM_TARGET_THRES = 0.70
_TARGET_SUM_OTHER_THRES = 0.85


@dataclass
class XgbAgent:
    """XGBoost agent implementing the same surface as :class:`typed_basd.Agent`.

    Public methods consumed by the server + :func:`interactive_eval`:

    * :meth:`next_action` — evidence indices maximising IG for each row.
    * :meth:`should_stop` — confidence-based stop (max class prob > thres).
    * :meth:`diagnose` — ``(argmax_class, prob_matrix)`` posterior.

    Attributes ``thres`` / ``mode`` / ``temp`` are overridable at load
    time by :func:`servers.symptoms.loader.apply_model_overrides` from
    the ``ModelSpec`` YAML. ``ig_smoothing`` and ``global_marginals`` are
    XGBoost-specific and stay agent-local.
    """

    classifier: Any
    ev_marginals: Any | None
    schema: dict
    columns: list[str]
    ev_col_index: list[list[int]]
    n_features: int
    thres: float
    mode: str = _MODE_CONFIDENCE
    temp: float = 1.0
    ig_smoothing: float = 0.05
    ig_recall_weight: float = 0.0
    # Direction the recall bonus counts: "asymmetric" rewards only
    # target-increasing shifts (correct for "confirm Pne/Flu recall"),
    # "symmetric" is the legacy |ΔP_target| behavior. Both no-ops when
    # ig_recall_weight == 0. See ig_policy.information_gain_per_column.
    ig_recall_mode: str = "asymmetric"
    global_marginals: np.ndarray | None = None
    # PoC additions for the 49-class + projected-IG design (see docs):
    # target_class_idxs pins the classes we want the IG policy + stop gate
    # to optimize for. Every other class is bucketed into ``Other`` at
    # projection time. When None, next_action / should_stop fall back to
    # the raw distribution (current v2 behavior). Set at construction
    # time from the dataset's ``target_condition_ids`` list — not baked
    # into weights, so a checkpoint can serve multiple downstream tasks.
    target_class_idxs: list[int] | None = None
    stop_policy: str = _STOP_PROJ_MAX
    variant_a_target_thres: float = _VARIANT_A_TARGET_THRES
    variant_a_other_thres: float = _VARIANT_A_OTHER_THRES
    target_sum_target_thres: float = _TARGET_SUM_TARGET_THRES
    target_sum_other_thres: float = _TARGET_SUM_OTHER_THRES
    # Serve-time penalty multiplier applied to antecedent-evidence IG scores
    # before argmax. 1.0 = no penalty (default); 0.1–0.3 pushes antecedents
    # to the back of the queue so current-symptom evidences dominate early
    # turns. Set via ModelSpec YAML (antecedent_penalty key) — no retraining.
    antecedent_penalty: float = 1.0

    def _antecedent_ev_mask(self) -> np.ndarray:
        """Bool array ``(n_ev,)`` — True where evidence is an antecedent.

        Built once from the schema and cached. Antecedents are risk factors /
        comorbidities (e.g. crowded living, obesity) rather than current
        symptoms. When ``antecedent_penalty < 1.0`` the IG policy multiplies
        antecedent scores by this factor so symptom-type evidences dominate
        early turns.
        """
        cached = getattr(self, "_cached_antecedent_mask", None)
        if cached is not None:
            return cached
        mask = np.array(
            [bool(ev.get("is_antecedent", False)) for ev in self.schema["evs"]],
            dtype=bool,
        )
        object.__setattr__(self, "_cached_antecedent_mask", mask)
        return mask

    def _predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Wrap ``classifier.predict_proba`` — always ``(batch, n_classes)``.

        XGBoost's binary path in xgboost>=2.0 returns ``(batch, 2)``
        naturally; kept as a wrapper for a) DataFrame-vs-array shape
        stability and b) so callers don't import xgboost directly.
        """
        return self.classifier.predict_proba(x)

    def _class_projection(self) -> ClassProjection | None:
        """Build the projection buckets from ``target_class_idxs``.

        Returns ``None`` when no target set — IG + stop fall back to the
        raw-distribution behavior (v2 semantics). When a target set is
        given, each target class gets its own bucket, and all remaining
        classes lump into a single trailing ``Other`` bucket. The result
        is cached on the instance in a lazy field to avoid rebuilding the
        big ``other`` list every call — DDXPlus has 49 classes, target
        list is usually 2-5, so ``other`` is 44-47 int entries and
        constructing it 200× per eval game shows up in the profiler.
        """
        if self.target_class_idxs is None:
            return None
        cached = getattr(self, "_cached_projection", None)
        if cached is not None:
            return cached
        n_classes = int(self.classifier.classes_.shape[0])
        target_set = set(self.target_class_idxs)
        other = [i for i in range(n_classes) if i not in target_set]
        projection: ClassProjection = [[i] for i in self.target_class_idxs]
        projection.append(other)
        # Instance-level cache — mypy doesn't know about the extra slot on
        # a dataclass, so store via object.__setattr__ to bypass frozen
        # dataclass semantics if we ever tighten that.
        object.__setattr__(self, "_cached_projection", projection)
        return projection

    def _marginals_batch(self, x_xgb: np.ndarray) -> np.ndarray:
        """Estimate ``P(col=1 | state)`` for a whole batch of rows.

        Prefers the fitted ``ev_marginals`` regressor; falls back to the
        global column means broadcast across the batch. Batching is
        essential at eval time: :class:`MultiOutputRegressor.predict`
        fans out to one call per output feature, so a per-row loop
        multiplies fixed per-call overhead by the batch size (at
        F=1000 features, ~1000 tiny XGB predicts per row) and can take
        hours on a full DDXPlus test split. Calling once on the whole
        batch collapses that into F predicts of ``(B,)`` — seconds.
        """
        if self.ev_marginals is not None:
            import joblib

            with joblib.parallel_config(backend="threading"):
                raw = self.ev_marginals.predict(x_xgb)
            return np.clip(np.asarray(raw, dtype=np.float32), 0.0, 1.0)
        if self.global_marginals is None:
            base = np.full(self.n_features, 0.05, dtype=np.float32)
        else:
            base = self.global_marginals.astype(np.float32)
        return np.broadcast_to(base, x_xgb.shape).copy()

    def next_action(self, state: np.ndarray) -> np.ndarray:
        """Return an ``(B,)`` array of evidence indices to ask next.

        Marginals are batched across the whole input (see
        :meth:`_marginals_batch`); IG scoring stays per-row because
        each row's ``asked`` mask differs. When every evidence is
        already asked (``pick_next_evidence`` returns -1) we fall back
        to evidence 0 — the caller's stop gate terminates the game on
        the next turn.
        """
        if state.ndim == 1:
            state = state[np.newaxis, :]
        batch = state.shape[0]
        x_xgb, asked = encode_typed_state_batch(
            state, self.schema, self.ev_col_index, self.n_features
        )
        marginals_batch = self._marginals_batch(x_xgb)
        projection = self._class_projection()
        apply_penalty = self.antecedent_penalty < 1.0
        ant_mask = self._antecedent_ev_mask() if apply_penalty else None
        out = np.zeros(batch, dtype=np.int64)
        for i in range(batch):
            if apply_penalty:
                scores = information_gain_per_evidence(
                    x_xgb[i],
                    asked[i],
                    self.ev_col_index,
                    self._predict_proba,
                    marginals_batch[i],
                    smoothing=self.ig_smoothing,
                    class_projection=projection,
                    recall_weight=self.ig_recall_weight,
                    recall_mode=self.ig_recall_mode,
                )
                scores[ant_mask] *= self.antecedent_penalty  # type: ignore[index]
                picked = (
                    int(np.argmax(scores)) if not np.all(np.isneginf(scores)) else -1
                )
            else:
                picked = pick_next_evidence(
                    x_xgb[i],
                    asked[i],
                    self.ev_col_index,
                    self._predict_proba,
                    marginals_batch[i],
                    smoothing=self.ig_smoothing,
                    class_projection=projection,
                    recall_weight=self.ig_recall_weight,
                    recall_mode=self.ig_recall_mode,
                )
            out[i] = picked if picked >= 0 else 0
        return out

    def should_stop(self, state: np.ndarray) -> np.ndarray:
        """Return ``(B,)`` bool: True where the classifier is confident enough.

        Without a class projection the check is ``max(raw probs) > thres``
        — the v2 semantics. With a projection, the check runs on the
        projected distribution and one of two policies fires:

        * ``proj_max`` (default): ``max(P_proj) > thres`` — symmetric.
        * ``proj_variant_a``: any target-bucket probability > 0.60 OR the
          Other bucket > 0.85. The asymmetric floor for "Other" prevents
          premature stops from hiding a rising target class.
        """
        if state.ndim == 1:
            state = state[np.newaxis, :]
        x_xgb, _asked = encode_typed_state_batch(
            state, self.schema, self.ev_col_index, self.n_features
        )
        probs = self._predict_proba(x_xgb)
        projection = self._class_projection()
        if projection is None:
            return probs.max(axis=1) > self.thres
        n_target = len(projection) - 1
        # Sum-over-bucket — same math as ig_policy._project but the extra
        # import path costs nothing at this stage and we can inline for
        # clarity, keeping the hot next_action path un-slowed.
        proj_probs = np.empty((probs.shape[0], len(projection)), dtype=probs.dtype)
        for k, bucket in enumerate(projection):
            proj_probs[:, k] = probs[:, bucket].sum(axis=1)
        if self.stop_policy == _STOP_PROJ_VARIANT_A:
            target_max = proj_probs[:, :n_target].max(axis=1)
            other = proj_probs[:, n_target]
            return (target_max > self.variant_a_target_thres) | (
                other > self.variant_a_other_thres
            )
        if self.stop_policy == _STOP_PROJ_TARGET_SUM:
            # Sum-of-targets stop: "definitely respiratory (Pne or Flu)"
            # counts as done even without resolving Pne vs Flu. Matches the
            # "minimize questions × maximize target recall" utility.
            target_sum = proj_probs[:, :n_target].sum(axis=1)
            other = proj_probs[:, n_target]
            return (target_sum > self.target_sum_target_thres) | (
                other > self.target_sum_other_thres
            )
        return proj_probs.max(axis=1) > self.thres

    def diagnose(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(argmax_class[B], prob_matrix[B, n_classes])``."""
        if state.ndim == 1:
            state = state[np.newaxis, :]
        x_xgb, _asked = encode_typed_state_batch(
            state, self.schema, self.ev_col_index, self.n_features
        )
        probs = self._predict_proba(x_xgb).astype(np.float32)
        return probs.argmax(axis=1), probs

    def save(self, path: Path) -> None:
        """Joblib-dump the whole agent to ``path``.

        Schema-derived fields (``schema``, ``ev_col_index``) are omitted
        from the persisted blob — they're rebuilt at load time from the
        current schema so a schema change fails loud on the manifest's
        ``feature_columns`` check rather than silently loading stale
        column mappings.
        """
        import joblib

        payload = {
            "classifier": self.classifier,
            "ev_marginals": self.ev_marginals,
            "columns": self.columns,
            "n_features": self.n_features,
            "thres": self.thres,
            "mode": self.mode,
            "temp": self.temp,
            "ig_smoothing": self.ig_smoothing,
            "ig_recall_weight": self.ig_recall_weight,
            "ig_recall_mode": self.ig_recall_mode,
            "global_marginals": self.global_marginals,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(
        cls,
        path: Path,
        schema: dict,
        columns_idx: dict[str, int] | None = None,
    ) -> "XgbAgent":
        """Rebuild an :class:`XgbAgent` from a joblib checkpoint + schema.

        ``columns_idx`` is optional — when ``None`` it's derived from the
        persisted ``columns`` list (which is authoritative). The adapter
        also runs a separate ``_verify_feature_columns`` check against
        the live schema so a train/serve column-order drift fails loud
        at startup rather than silently mis-indexing features.
        """
        import joblib

        payload = joblib.load(path)
        columns: list[str] = payload["columns"]
        if columns_idx is None:
            columns_idx = {c: i for i, c in enumerate(columns)}
        ev_col_index = evidence_column_index(schema, columns_idx)
        return cls(
            classifier=payload["classifier"],
            ev_marginals=payload["ev_marginals"],
            schema=schema,
            columns=columns,
            ev_col_index=ev_col_index,
            n_features=payload["n_features"],
            thres=payload["thres"],
            mode=payload.get("mode", _MODE_CONFIDENCE),
            temp=payload.get("temp", 1.0),
            ig_smoothing=payload.get("ig_smoothing", 0.05),
            ig_recall_weight=payload.get("ig_recall_weight", 0.0),
            # Default asymmetric — old v5 checkpoints trained under symmetric
            # abs() semantics but the module docstring for
            # ``information_gain_per_column`` treats asymmetric as the
            # recommended default now. Operators who want to reproduce the
            # legacy v5 numbers set ``ig_recall_mode: symmetric`` in the
            # ModelSpec override.
            ig_recall_mode=payload.get("ig_recall_mode", "asymmetric"),
            global_marginals=payload.get("global_marginals"),
        )


def build_xgb_agent(
    classifier: Any,
    ev_marginals: Any | None,
    schema: dict,
    columns: list[str],
    columns_idx: dict[str, int],
    *,
    thres: float = 0.90,
    ig_smoothing: float = 0.05,
    ig_recall_weight: float = 0.0,
    global_marginals: np.ndarray | None = None,
) -> XgbAgent:
    """Assemble an :class:`XgbAgent` from freshly-trained components.

    Mirrors :func:`typed_basd.build_basd` for symmetric operator tooling:
    training scripts call this to construct the agent before calling
    :meth:`XgbAgent.save`. Serve-time construction goes through
    :meth:`XgbAgent.load` instead.
    """
    ev_col_index = evidence_column_index(schema, columns_idx)
    return XgbAgent(
        classifier=classifier,
        ev_marginals=ev_marginals,
        schema=schema,
        columns=columns,
        ev_col_index=ev_col_index,
        n_features=len(columns),
        thres=thres,
        ig_smoothing=ig_smoothing,
        ig_recall_weight=ig_recall_weight,
        global_marginals=global_marginals,
    )
