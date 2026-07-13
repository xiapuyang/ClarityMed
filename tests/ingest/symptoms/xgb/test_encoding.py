"""Tests for the XGBoost encoding layer.

Uses a small synthetic schema (three evidences: one binary, one
categorical, one multi-choice) so nothing here depends on DDXPlus being
downloaded. The synthetic schema is passed through
:func:`build_layout` so we exercise the same shape XgbAgent will see at
serve time.
"""

from __future__ import annotations

import numpy as np
import pytest

from claritymed.ingest.symptoms.typed_basd import TypedEnv, build_layout
from claritymed.ingest.symptoms.xgb.encoding import (
    encode_patient_batch,
    encode_typed_state_batch,
    evidence_column_index,
    feature_columns_from_schema,
)


def _synthetic_schema() -> dict:
    """Three evidences: E_B (binary), E_C (categorical, 3 values), E_M (multi, 2 values)."""
    evs = [
        {"name": "E_B", "dtype": "B", "values": []},
        {"name": "E_C", "dtype": "C", "values": ["mild", "moderate", "severe"]},
        {"name": "E_M", "dtype": "M", "values": ["red", "blue"]},
    ]
    return build_layout(evs)


def _synthetic_patient() -> dict:
    """Patient with E_B positive, E_C=moderate, E_M=[red, blue]."""
    return dict(
        bin_pos={0},
        cat_val={1: 1},  # E_C → local idx 1 → "moderate"
        multi_val={2: [0, 1]},  # E_M → both values
        pos={0, 1, 2},
        init=0,
        d=0,
        age=3,
        sex=0,
        diff=np.array([1.0, 0.0]),
    )


def test_feature_columns_from_schema_binary_gets_one_column():
    schema = _synthetic_schema()
    columns, _labels, index = feature_columns_from_schema(schema)
    assert "E_B" in index
    # Binary evidence emits exactly one column (name matches evidence id).
    assert sum(1 for c in columns if c.startswith("E_B")) == 1


def test_feature_columns_from_schema_categorical_and_multi_get_one_per_value():
    schema = _synthetic_schema()
    columns, _labels, index = feature_columns_from_schema(schema)
    # Cat: 3 values → 3 columns.
    cat_cols = [c for c in columns if c.startswith("E_C__")]
    assert cat_cols == ["E_C__mild", "E_C__moderate", "E_C__severe"]
    # Multi: 2 values → 2 columns, in schema order.
    multi_cols = [c for c in columns if c.startswith("E_M__")]
    assert multi_cols == ["E_M__red", "E_M__blue"]
    assert len(columns) == 1 + 3 + 2
    # Index maps names to positions bijectively.
    assert {i: c for c, i in index.items()} == dict(enumerate(columns))


def test_feature_columns_ordering_stable_across_reloads():
    a = feature_columns_from_schema(_synthetic_schema())[0]
    b = feature_columns_from_schema(_synthetic_schema())[0]
    assert a == b


def test_feature_columns_uses_meta_labels_when_available():
    schema = _synthetic_schema()
    meta = {
        "E_C": {
            "question_en": "How severe?",
            "value_meaning": {"moderate": {"en": "Moderate pain"}},
        },
    }
    _columns, labels, index = feature_columns_from_schema(schema, meta)
    # The E_C__moderate column's label incorporates the meta translation.
    col_idx = index["E_C__moderate"]
    assert labels[col_idx] == "E_C=Moderate pain: How severe?"


def test_encode_patient_batch_produces_expected_one_hot_pattern():
    schema = _synthetic_schema()
    _columns, _labels, index = feature_columns_from_schema(schema)
    x = encode_patient_batch([_synthetic_patient()], schema, index)
    assert x.shape == (1, len(index))
    assert x.dtype == np.float32
    # Positive binary → 1.0 at its column.
    assert x[0, index["E_B"]] == 1.0
    # Cat: only the answered value's column is set.
    assert x[0, index["E_C__moderate"]] == 1.0
    assert x[0, index["E_C__mild"]] == 0.0
    assert x[0, index["E_C__severe"]] == 0.0
    # Multi-hot: both values light up.
    assert x[0, index["E_M__red"]] == 1.0
    assert x[0, index["E_M__blue"]] == 1.0


def test_encode_patient_batch_empty_patient_all_zeros():
    schema = _synthetic_schema()
    _columns, _labels, index = feature_columns_from_schema(schema)
    empty = dict(
        bin_pos=set(),
        cat_val={},
        multi_val={},
        pos=set(),
        init=0,
        d=0,
        age=0,
        sex=0,
        diff=np.zeros(2),
    )
    x = encode_patient_batch([empty], schema, index)
    assert np.all(x == 0.0)


def test_evidence_column_index_maps_evidence_to_its_columns():
    schema = _synthetic_schema()
    _columns, _labels, index = feature_columns_from_schema(schema)
    per_ev = evidence_column_index(schema, index)
    # Binary → one column (E_B).
    assert per_ev[0] == [index["E_B"]]
    # Categorical → K columns in schema order.
    assert per_ev[1] == [
        index["E_C__mild"],
        index["E_C__moderate"],
        index["E_C__severe"],
    ]
    # Multi → K columns in schema order.
    assert per_ev[2] == [index["E_M__red"], index["E_M__blue"]]


def test_encode_typed_state_batch_binary_positive_and_negative():
    schema = _synthetic_schema()
    _columns, _labels, index = feature_columns_from_schema(schema)
    per_ev = evidence_column_index(schema, index)
    env = TypedEnv([_synthetic_patient()], schema, n_dis=2)
    s, _ = env.initialize_state(1)
    # E_B is the patient's init evidence and is positive → typed state
    # should have +1 at off[0].
    assert s[0, schema["off"][0]] == 1.0
    x_xgb, asked = encode_typed_state_batch(s, schema, per_ev, len(index))
    assert x_xgb[0, index["E_B"]] == 1.0
    assert asked[0, 0] is np.True_ or asked[0, 0] == True  # noqa: E712
    # E_C / E_M weren't revealed by init → not asked, XGBoost zeros.
    assert not asked[0, 1]
    assert not asked[0, 2]
    assert x_xgb[0, index["E_C__moderate"]] == 0.0
    assert x_xgb[0, index["E_M__red"]] == 0.0


def test_encode_typed_state_batch_after_reveal_of_categorical():
    schema = _synthetic_schema()
    _columns, _labels, index = feature_columns_from_schema(schema)
    per_ev = evidence_column_index(schema, index)
    patient = _synthetic_patient()
    env = TypedEnv([patient], schema, n_dis=2)
    s, _ = env.initialize_state(1)
    # Reveal E_C (evidence idx 1) — patient answered "moderate".
    done = np.array([False])
    s = env.reveal(s, np.array([1]), done)
    x_xgb, asked = encode_typed_state_batch(s, schema, per_ev, len(index))
    assert asked[0, 1]
    assert x_xgb[0, index["E_C__moderate"]] == 1.0
    assert x_xgb[0, index["E_C__mild"]] == 0.0
    assert x_xgb[0, index["E_C__severe"]] == 0.0


def test_encode_typed_state_batch_negative_binary_marked_asked_zero_xgb():
    schema = _synthetic_schema()
    _columns, _labels, index = feature_columns_from_schema(schema)
    per_ev = evidence_column_index(schema, index)
    # Patient with E_B NOT positive — TypedEnv writes -1 in the binary slot
    # when the agent asks. Build a fake typed state directly to make this
    # deterministic.
    typed = np.zeros((1, schema["sym_size"]))
    typed[0, schema["off"][0]] = -1.0  # asked, negative
    x_xgb, asked = encode_typed_state_batch(typed, schema, per_ev, len(index))
    # asked_mask still True (we did ask; the answer was 'no').
    assert asked[0, 0]
    # XGBoost column is 0 — negative and unasked look identical to XGB;
    # the caller uses asked_mask to disambiguate.
    assert x_xgb[0, index["E_B"]] == 0.0


def test_encode_typed_state_batch_handles_1d_input():
    """Single-row 1-D input should broadcast to a 1-row output."""
    schema = _synthetic_schema()
    _columns, _labels, index = feature_columns_from_schema(schema)
    per_ev = evidence_column_index(schema, index)
    typed = np.zeros(schema["sym_size"])
    typed[schema["off"][0]] = 1.0
    x_xgb, asked = encode_typed_state_batch(typed, schema, per_ev, len(index))
    assert x_xgb.shape == (1, len(index))
    assert asked.shape == (1, schema["n_ev"])


@pytest.mark.parametrize("dtype", ["B", "C", "M"])
def test_column_count_per_evidence(dtype: str):
    """Column count per evidence: binary=1, cat=K, multi=K."""
    if dtype == "B":
        evs = [{"name": "E_X", "dtype": "B", "values": []}]
        expected = 1
    else:
        evs = [{"name": "E_X", "dtype": dtype, "values": ["a", "b", "c", "d"]}]
        expected = 4
    schema = build_layout(evs)
    columns, _labels, _index = feature_columns_from_schema(schema)
    assert len(columns) == expected


def test_label_leakage_blacklist_drops_columns_and_empties_ev_col_index():
    """E_131 / E_135 are dropped from feature columns and get empty
    ev_col_index entries so the IG policy can't pick them and the encoder
    silently ignores any answers routed to them."""
    evs = [
        {"name": "E_1", "dtype": "B", "values": []},
        {"name": "E_131", "dtype": "C", "values": ["V_10", "V_12"]},
        {"name": "E_135", "dtype": "C", "values": ["V_10", "V_12"]},
        {"name": "E_77", "dtype": "B", "values": []},
    ]
    schema = build_layout(evs)
    columns, _labels, index = feature_columns_from_schema(schema)
    assert columns == ["E_1", "E_77"]
    assert "E_131__V_12" not in index
    assert "E_135__V_10" not in index
    per_ev = evidence_column_index(schema, index)
    # Same ev_col_index length as schema['evs'] so downstream indexing by
    # ev_idx stays intact; blacklisted rows are empty lists.
    assert len(per_ev) == 4
    assert per_ev[0] == [index["E_1"]]
    assert per_ev[1] == []  # E_131 blacklisted
    assert per_ev[2] == []  # E_135 blacklisted
    assert per_ev[3] == [index["E_77"]]
