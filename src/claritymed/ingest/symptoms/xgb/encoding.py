"""State ↔ feature-vector encoding for the XGBoost algorithm module.

XGBoost consumes a single dense one-hot feature vector, unlike typed-BASD
which uses a trinary/one-hot/multi-hot layout with an explicit asked flag
per block. This module owns the mapping in both directions:

* Training time: patient dicts → dense matrix (:func:`encode_patient_batch`).
* Inference time: typed-BASD state (from :class:`TypedEnv`) → dense matrix
  + per-evidence asked mask (:func:`encode_typed_state_batch`). The
  ``asked mask`` is essential — the raw XGBoost vector can't distinguish
  "not asked" from "asked, negative" for binary evidences and the IG
  policy needs to know which evidences are still available to ask.

Column layout (baked into the manifest via ``feature_columns``):

* Binary evidence  → 1 column named ``E_X``.
* Categorical evidence → K columns named ``E_X__value``, one per raw
  value in ``schema['evs'][i]['values']``. Only one is set per patient.
* Multi-value evidence → K columns named ``E_X__value``. Multi-hot when
  the patient's value list includes that value.

Column order is deterministic and derived from the DDXPlus schema
(sorted-by-name evidences × schema-order value lists). The adapter
fails loud on column-order mismatch at load time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DDXPLUS_EVIDENCES_JSON = "release_evidences.json"

# Evidences excluded from the feature space. E_131 (皮损脱皮 "lesions peel off?")
# and E_135 (皮损>1cm "lesion larger than 1cm?") are dermatology categoricals
# whose class-conditional prevalence in the DDXPlus Pneumonia+Influenza subset
# is a near-deterministic label proxy (P(=Y|Pne) ≈ 0.993 vs P(=Y|Inf) ≈ 0.0
# for E_131; symmetric for E_135). The XGB classifier learned to route the
# 2-class decision through these dermatology signals, and the IG policy asked
# them within 3 turns while never touching clinically-meaningful respiratory
# features (E_77 sputum, E_88 fatigue, E_66 dyspnea). Blacklisting drops the
# columns at feature-enumeration time so the classifier can't fit them and
# the IG policy can't ask them. Applies to all datasets that share this
# encoder — DDXPlus derm conditions live in a different subset that would
# need its own model anyway.
_LABEL_LEAKAGE_BLACKLIST: frozenset[str] = frozenset({"E_131", "E_135"})


def load_evidence_meta(data_dir: str | Path) -> dict[str, dict]:
    """Return ``{evidence_id: raw JSON entry}`` from ``release_evidences.json``.

    Kept separate from :func:`load_evidence_schema` so callers that only
    need question / value labels (feature-importance viewer, tree dumps)
    don't have to rebuild the typed layout.
    """
    path = Path(data_dir) / DDXPLUS_EVIDENCES_JSON
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = list(raw.values()) if isinstance(raw, dict) else raw
    return {e["name"]: e for e in items}


def feature_columns_from_schema(
    schema: dict, meta: dict[str, dict] | None = None
) -> tuple[list[str], list[str], dict[str, int]]:
    """Enumerate feature columns from a DDXPlus-shaped schema.

    Returns ``(columns, labels, index)`` where ``columns`` is the ordered
    list of column names, ``labels`` is a parallel list of human-readable
    ``E_X: question`` / ``E_X=value_meaning: question`` strings (falls
    back to raw ids when ``meta`` is ``None`` or missing entries), and
    ``index`` maps column name → ordinal position.

    Column order is: for each evidence (already sorted by name in
    ``schema['evs']``), emit binary as 1 column, cat/multi as K columns
    in the schema's value order. Any change to this layout requires a
    manifest version bump because the adapter cross-checks
    ``manifest.feature_columns`` on load.
    """
    columns: list[str] = []
    labels: list[str] = []
    meta = meta or {}
    for ev in schema["evs"]:
        ev_id = ev["name"]
        if ev_id in _LABEL_LEAKAGE_BLACKLIST:
            continue
        dtype = ev["dtype"]
        entry = meta.get(ev_id, {})
        q_text = entry.get("question_en", ev_id)
        if dtype == "B":
            columns.append(ev_id)
            labels.append(f"{ev_id}: {q_text}")
        else:
            for value in ev["values"]:
                columns.append(f"{ev_id}__{value}")
                meaning = entry.get("value_meaning", {}).get(str(value), {}).get(
                    "en"
                ) or str(value)
                labels.append(f"{ev_id}={meaning}: {q_text}")
    index = {c: i for i, c in enumerate(columns)}
    return columns, labels, index


def evidence_column_index(schema: dict, columns_idx: dict[str, int]) -> list[list[int]]:
    """Return ``[ev_idx] → list of XGBoost column indices``.

    For a binary evidence the inner list has length 1. For cat/multi it
    has length ``len(schema['evs'][ev_idx]['values'])``. Used by the IG
    policy to group column-level marginals under their parent evidence,
    and by :func:`encode_typed_state_batch` to route typed-BASD block
    slots to the right XGBoost columns.
    """
    per_ev: list[list[int]] = []
    for ev in schema["evs"]:
        ev_id = ev["name"]
        # Blacklisted evidences get an empty column list so the IG policy
        # scores them at -inf (see ig_policy.information_gain_per_evidence)
        # and the encoder silently drops any live answers routed to them.
        if ev_id in _LABEL_LEAKAGE_BLACKLIST:
            per_ev.append([])
            continue
        dtype = ev["dtype"]
        if dtype == "B":
            per_ev.append([columns_idx[ev_id]])
        else:
            per_ev.append([columns_idx[f"{ev_id}__{v}"] for v in ev["values"]])
    return per_ev


def encode_patient_batch(
    patients: list[dict], schema: dict, columns_idx: dict[str, int]
) -> np.ndarray:
    """One-hot encode DDXPlus patients into a dense ``float32`` matrix.

    Positive binary evidences → 1.0. Categorical answered value → 1.0 at
    that ``E_X__value`` slot. Multi-choice values → 1.0 at each answered
    ``E_X__value`` slot. Unspecified slots stay 0.0. Shape is
    ``(len(patients), len(columns_idx))``.
    """
    n_features = len(columns_idx)
    x = np.zeros((len(patients), n_features), dtype=np.float32)
    ev_names = [ev["name"] for ev in schema["evs"]]
    for i, p in enumerate(patients):
        for ev_i in p["bin_pos"]:
            col = columns_idx.get(ev_names[ev_i])
            if col is not None:
                x[i, col] = 1.0
        for ev_i, lv in p["cat_val"].items():
            raw = schema["evs"][ev_i]["values"][lv]
            col = columns_idx.get(f"{ev_names[ev_i]}__{raw}")
            if col is not None:
                x[i, col] = 1.0
        for ev_i, lvs in p["multi_val"].items():
            for lv in lvs:
                raw = schema["evs"][ev_i]["values"][lv]
                col = columns_idx.get(f"{ev_names[ev_i]}__{raw}")
                if col is not None:
                    x[i, col] = 1.0
    return x


def encode_typed_state_batch(
    typed_state: np.ndarray,
    schema: dict,
    ev_col_index: list[list[int]],
    n_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate a typed-BASD state batch to XGBoost dense + asked mask.

    typed-BASD blocks per evidence:

    * ``B`` (binary): 1 slot at ``off``. ``+1`` = positive, ``-1`` =
      negative, ``0`` = not asked.
    * ``C`` (categorical): ``[asked | one-hot(K)]``. The XGBoost column
      for the answered value is set to 1.0.
    * ``M`` (multi): ``[asked | multi-hot(K)]``. Each answered slot
      lights up its XGBoost column.

    Returns ``(x_xgb, asked_mask)``:

    * ``x_xgb``: ``(B, n_features)`` float32 dense matrix. Negative
      binaries and unanswered slots are both 0.0 — the caller uses
      ``asked_mask`` to distinguish "no" from "not yet asked".
    * ``asked_mask``: ``(B, n_ev)`` bool matrix. True where the evidence
      block's asked slot fires (``typed[off] != 0`` for binary and
      categorical/multi alike).
    """
    if typed_state.ndim == 1:
        typed_state = typed_state[np.newaxis, :]
    batch = typed_state.shape[0]
    n_ev = schema["n_ev"]
    off = schema["off"]
    typ = schema["typ"]
    x_xgb = np.zeros((batch, n_features), dtype=np.float32)
    asked = np.zeros((batch, n_ev), dtype=bool)
    for i in range(batch):
        row = typed_state[i]
        for ev_i in range(n_ev):
            block_start = int(off[ev_i])
            head = row[block_start]
            if head == 0.0:
                continue
            asked[i, ev_i] = True
            cols = ev_col_index[ev_i]
            if typ[ev_i] == "B":
                if head > 0:
                    x_xgb[i, cols[0]] = 1.0
                continue
            # Cat / Multi: read the K value slots after the asked flag.
            for k, col in enumerate(cols):
                if row[block_start + 1 + k] > 0:
                    x_xgb[i, col] = 1.0
    return x_xgb, asked
