"""Unit tests for the ``xgb_v3_demo_and_verify`` operator helper.

Covers the pure helpers (bucketing, patient relabelling, evidence-label
rendering, top-K binary picker) so a refactor that breaks the numeric
core is caught by unit tests. The heavy end-to-end path (loading a v3
weights.pkl, running the interactive loop) is left for manual invocation
against real DDXPlus data — bootstrapping a fake xgboost checkpoint
here would double the test surface without meaningfully increasing
coverage of the pure helpers we care about.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import xgb_v3_demo_and_verify as mod


def test_pick_typical_indices_bands_and_ordering() -> None:
    """High → highest first; low → lowest first; mid → closest to center."""
    p_pne = np.array([0.99, 0.80, 0.66, 0.55, 0.50, 0.42, 0.10, 0.05])
    picks = mod._pick_typical_indices(p_pne, high=0.65, low=0.40, per_bucket=2)
    # High band (>=0.65): 0.99, 0.80 — sorted descending
    assert picks["high"] == [0, 1]
    # Low band (<0.40): 0.10, 0.05 — sorted ascending
    assert picks["low"] == [7, 6]
    # Mid band [0.40, 0.65): 0.55, 0.50, 0.42 — closest to (0.65+0.40)/2 = 0.525
    # sorted by |p - 0.525|: 0.50 (0.025), 0.55 (0.025), 0.42 (0.105)
    # both 0.50 and 0.55 tie; argsort picks in original order → [4, 3]
    assert set(picks["mid"][:2]) == {3, 4}


def test_pick_typical_indices_handles_empty_bands() -> None:
    """A band with no members returns an empty list, not a crash."""
    p_pne = np.array([0.99, 0.98, 0.97])  # all high
    picks = mod._pick_typical_indices(p_pne, high=0.65, low=0.40, per_bucket=3)
    assert picks["high"] == [0, 1, 2]
    assert picks["mid"] == []
    assert picks["low"] == []


def test_relabel_patients_to_subset_maps_targets_and_lumps_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pne → 0, Inf → 1, everything else → 2."""
    # Fake pidx: Pneumonia=17, Influenza=22, Bronchitis=5, TB=41
    full_pidx = {"Pneumonia": 17, "Influenza": 22, "Bronchitis": 5, "TB": 41}
    patients = [
        {"d": 17},  # Pneumonia
        {"d": 22},  # Influenza
        {"d": 5},  # Bronchitis
        {"d": 41},  # TB
    ]
    mod._relabel_patients_to_subset(patients, full_pidx)
    assert [p["d"] for p in patients] == [0, 1, 2, 2]


def test_evidence_labels_for_patient_covers_all_dtypes() -> None:
    """Binary + categorical + multi-value evidences all render."""
    schema = {
        "evs": [
            {"name": "E_1", "dtype": "B", "values": []},
            {"name": "E_2", "dtype": "C", "values": ["V_a", "V_b"]},
            {"name": "E_3", "dtype": "M", "values": ["V_x", "V_y"]},
        ]
    }
    meta = {
        "E_1": {"question_en": "Do you cough?"},
        "E_2": {
            "question_en": "What color?",
            "value_meaning": {
                "V_a": {"en": "red"},
                "V_b": {"en": "blue"},
            },
        },
        "E_3": {
            "question_en": "Where is pain?",
            "value_meaning": {
                "V_x": {"en": "chest"},
                "V_y": {"en": "back"},
            },
        },
    }
    patient = {
        "bin_pos": [0],
        "cat_val": {1: 0},
        "multi_val": {2: [0, 1]},
    }
    labels = mod._evidence_labels_for_patient(patient, schema, meta)
    assert labels == [
        "E_1: Do you cough?",
        "E_2=red: What color?",
        "E_3=chest: Where is pain?",
        "E_3=back: Where is pain?",
    ]


def test_top_k_binary_evidences_prefers_manifest_gain_over_schema_order(
    tmp_path: Path,
) -> None:
    """Manifest importance ranks binary evidences; categorical slots skipped."""
    schema = {
        "evs": [
            {"name": "E_1", "dtype": "B", "values": []},
            {"name": "E_2", "dtype": "C", "values": ["V_a"]},  # cat — must skip
            {"name": "E_3", "dtype": "B", "values": []},
            {"name": "E_4", "dtype": "B", "values": []},
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "feature_importance_top_k": [
                    {"name": "E_2__V_a", "gain": 999.0},  # cat slot → skip
                    {"name": "E_3", "gain": 500.0},
                    {"name": "E_1", "gain": 100.0},
                    {"name": "E_4", "gain": 50.0},
                ]
            }
        )
    )
    picked = mod._top_k_binary_evidences(manifest_path, schema, k=2)
    assert picked == ["E_3", "E_1"]


def test_top_k_binary_evidences_falls_back_to_schema_when_manifest_short(
    tmp_path: Path,
) -> None:
    """Under-supply from manifest → fill from schema-order binary evidences."""
    schema = {
        "evs": [
            {"name": "E_1", "dtype": "B", "values": []},
            {"name": "E_2", "dtype": "B", "values": []},
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"feature_importance_top_k": [{"name": "E_1", "gain": 100.0}]})
    )
    picked = mod._top_k_binary_evidences(manifest_path, schema, k=2)
    assert picked == ["E_1", "E_2"]


def test_answer_for_patient_renders_binary_yes_no() -> None:
    schema = {"evs": [{"name": "E_1", "dtype": "B", "values": []}]}
    meta = {"E_1": {"question_en": "Q"}}
    patient_pos = {"bin_pos": [0], "cat_val": {}, "multi_val": {}}
    patient_neg = {"bin_pos": [], "cat_val": {}, "multi_val": {}}
    assert mod._answer_for_patient(patient_pos, 0, schema, meta) == "yes"
    assert mod._answer_for_patient(patient_neg, 0, schema, meta) == "no"


def test_answer_for_patient_renders_categorical_value_meaning() -> None:
    schema = {"evs": [{"name": "E_1", "dtype": "C", "values": ["V_a", "V_b"]}]}
    meta = {
        "E_1": {
            "question_en": "Q",
            "value_meaning": {"V_a": {"en": "red"}, "V_b": {"en": "blue"}},
        }
    }
    patient_a = {"bin_pos": [], "cat_val": {0: 0}, "multi_val": {}}
    patient_b = {"bin_pos": [], "cat_val": {0: 1}, "multi_val": {}}
    patient_none = {"bin_pos": [], "cat_val": {}, "multi_val": {}}
    assert mod._answer_for_patient(patient_a, 0, schema, meta) == "red"
    assert mod._answer_for_patient(patient_b, 0, schema, meta) == "blue"
    assert mod._answer_for_patient(patient_none, 0, schema, meta) == "not answered"


def test_answer_for_patient_renders_multi_value_joined() -> None:
    schema = {"evs": [{"name": "E_1", "dtype": "M", "values": ["V_x", "V_y"]}]}
    meta = {
        "E_1": {
            "question_en": "Q",
            "value_meaning": {"V_x": {"en": "chest"}, "V_y": {"en": "back"}},
        }
    }
    patient_both = {"bin_pos": [], "cat_val": {}, "multi_val": {0: [0, 1]}}
    patient_none = {"bin_pos": [], "cat_val": {}, "multi_val": {}}
    assert mod._answer_for_patient(patient_both, 0, schema, meta) == "chest, back"
    assert mod._answer_for_patient(patient_none, 0, schema, meta) == "none"


def test_find_default_chief_complaint_prefers_cough_related() -> None:
    schema = {
        "evs": [
            {"name": "E_1", "dtype": "B", "values": []},
            {"name": "E_2", "dtype": "B", "values": []},
        ]
    }
    meta = {
        "E_1": {"question_en": "Do you have shortness of breath?"},
        "E_2": {"question_en": "Do you have a cough?"},
    }
    assert mod._find_default_chief_complaint(schema, meta) == "E_2"


def test_find_default_chief_complaint_falls_back_to_first_binary() -> None:
    schema = {
        "evs": [
            {"name": "E_1", "dtype": "C", "values": ["V_a"]},  # cat — skip
            {"name": "E_2", "dtype": "B", "values": []},
            {"name": "E_3", "dtype": "B", "values": []},
        ]
    }
    meta = {
        "E_2": {"question_en": "Something unrelated?"},
        "E_3": {"question_en": "Also unrelated?"},
    }
    # No cough in questions → first binary in schema order (E_2).
    assert mod._find_default_chief_complaint(schema, meta) == "E_2"


def test_enumerate_binary_grid_sets_bits_correctly() -> None:
    """Bit i of the row index → column for top_k_binary[i] gets 1.0."""
    columns_idx = {"CC": 0, "E_A": 1, "E_B": 2}
    schema: dict = {}  # unused in this path
    # Stub agent with a predict_proba that returns the input columns 1..2
    # so the caller can assert bit ↔ column mapping cleanly.
    fake_agent = SimpleNamespace(
        classifier=SimpleNamespace(
            predict_proba=lambda x: np.column_stack(
                [x[:, 1], x[:, 2], np.zeros(x.shape[0])]
            )
        )
    )
    probs, answers = mod._enumerate_binary_grid(
        fake_agent,  # type: ignore[arg-type]
        schema,
        columns_idx,
        chief_complaint_ev="CC",
        age_bucket=3,
        sex=0,
        top_k_binary=["E_A", "E_B"],
    )
    # 2 axes → 4 combos.
    assert probs.shape == (4, 3)
    # Every row should have CC=1.
    # answers[i]['E_A'] == 'yes' iff bit 0 of i set.
    assert answers[0] == {"E_A": "no", "E_B": "no"}
    assert answers[1] == {"E_A": "yes", "E_B": "no"}
    assert answers[2] == {"E_A": "no", "E_B": "yes"}
    assert answers[3] == {"E_A": "yes", "E_B": "yes"}
    # Column 1 (E_A) mirrored to probs[:, 0]: 0,1,0,1
    assert np.allclose(probs[:, 0], [0.0, 1.0, 0.0, 1.0])
    # Column 2 (E_B) mirrored to probs[:, 1]: 0,0,1,1
    assert np.allclose(probs[:, 1], [0.0, 0.0, 1.0, 1.0])
