"""Tests for the symptoms-server question/answer translation layer (canonical-based).

Builds synthetic :class:`CanonicalDataset` fixtures so the tests don't
depend on DDXPlus files. Covers:

* ``build_question`` rendering binary / categorical / numeric / multi.
* i18n resolution: keys present → localized text; keys missing → falls
  back to native_question_text / native_value_labels then to the raw id.
* ``synth_patient`` round-trip with raw ``answer_value`` and localized labels.
* ``localize_condition`` resolves slugs to display names with the full
  fallback chain.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from claritymed.core.i18n import loader as i18n_loader
from claritymed.core.i18n.loader import _reset_for_tests
from claritymed.core.symptoms.datasets import (
    CanonicalCondition,
    CanonicalDataset,
    CanonicalEvidence,
    CanonicalValue,
)
from claritymed.core.symptoms.schemas import DatasetSpec
from claritymed.ingest.symptoms.typed_basd import build_layout
from claritymed.servers.symptoms.questions import (
    QuestionPayloadError,
    build_question,
    localize_condition,
    synth_patient,
)


@pytest.fixture
def i18n_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "i18n"
    d.mkdir()
    monkeypatch.setattr(i18n_loader, "I18N_DIR", d)
    _reset_for_tests()
    return d


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


@pytest.fixture
def spec() -> DatasetSpec:
    return DatasetSpec(id="ddxplus", model_ids=["typed_basd_v1"])


def _canonical() -> CanonicalDataset:
    """3-evidence + 2-condition synthetic CanonicalDataset.

    Evidence layout mirrors the DDXPlus shape (binary, categorical with
    string values, numeric categorical) so the rendering logic is fully
    exercised without depending on the corpus.
    """
    layout = build_layout(
        [
            {"name": "E_91", "dtype": "B", "values": []},
            {"name": "E_55", "dtype": "C", "values": ["V_123", "V_14", "V_15"]},
            {"name": "E_pain", "dtype": "C", "values": ["0", "5", "10"]},
        ]
    )
    evidences = [
        CanonicalEvidence(
            id="E_91",
            idx=0,
            dtype="B",
            native_question_text={"en": "Do you have a fever?"},
        ),
        CanonicalEvidence(
            id="E_55",
            idx=1,
            dtype="C",
            values=[
                CanonicalValue(raw="V_123", local_idx=0),
                CanonicalValue(raw="V_14", local_idx=1),
                CanonicalValue(raw="V_15", local_idx=2),
            ],
            native_question_text={"en": "Where do you feel pain?"},
            native_value_labels={
                "V_123": {"en": "nowhere"},
                "V_14": {"en": "iliac wing (right)"},
                "V_15": {"en": "iliac wing (left)"},
            },
        ),
        CanonicalEvidence(
            id="E_pain",
            idx=2,
            dtype="C",
            values=[
                CanonicalValue(raw="0", local_idx=0),
                CanonicalValue(raw="5", local_idx=1),
                CanonicalValue(raw="10", local_idx=2),
            ],
            native_question_text={"en": "Rate the pain."},
        ),
    ]
    conditions = [
        CanonicalCondition(
            id="spontaneous_pneumothorax",
            idx=0,
            severity=2,
            icd10="J93",
            native_name={"en": "Spontaneous pneumothorax"},
        ),
        CanonicalCondition(
            id="common_cold",
            idx=1,
            severity=5,
            native_name={"en": "Common cold"},
        ),
    ]
    return CanonicalDataset.build(
        id="ddxplus",
        evidences=evidences,
        conditions=conditions,
        layout=layout,
        severity_vector=np.array([2.0, 5.0]),
    )


# --- build_question (no i18n keys present, fall back to native) ------------


def test_binary_question_falls_back_to_native_question(spec, i18n_dir) -> None:
    cd = _canonical()
    q = build_question(cd, spec, ev_idx=0, language="en")
    assert q.numeric is None
    assert q.question == "Do you have a fever?"
    labels = {opt.label for opt in q.options}
    assert labels == {"Yes", "No"}
    for opt in q.options:
        assert "·yes" in opt.description or "·no" in opt.description


def test_categorical_question_uses_native_value_labels(spec, i18n_dir) -> None:
    cd = _canonical()
    q = build_question(cd, spec, ev_idx=1, language="en")
    labels = {opt.label for opt in q.options}
    assert {"nowhere", "iliac wing (right)", "iliac wing (left)"} <= labels


def test_numeric_question_for_numeric_values(spec, i18n_dir) -> None:
    cd = _canonical()
    q = build_question(cd, spec, ev_idx=2, language="en")
    assert q.numeric is not None
    assert q.numeric.min == 0
    assert q.numeric.max == 10
    assert q.options == []


# --- build_question (i18n keys present, ZH-localized) ----------------------


def test_question_text_resolves_zh_i18n_key(spec, i18n_dir) -> None:
    _write_yaml(
        i18n_dir / "zh" / "symptoms_ddxplus.yaml",
        {"symptoms": {"ddxplus": {"E_91": {"question": "你发烧了吗？"}}}},
    )
    _write_yaml(
        i18n_dir / "zh" / "common.yaml",
        {"symptoms": {"binary": {"yes": "是", "no": "否"}}},
    )
    cd = _canonical()
    q = build_question(cd, spec, ev_idx=0, language="zh")
    assert q.question == "你发烧了吗？"
    labels = {opt.label for opt in q.options}
    assert labels == {"是", "否"}


def test_value_label_resolves_zh_i18n_key(spec, i18n_dir) -> None:
    _write_yaml(
        i18n_dir / "zh" / "symptoms_ddxplus.yaml",
        {
            "symptoms": {
                "ddxplus": {
                    "E_55": {
                        "question": "你哪里痛？",
                        "values": {
                            "V_123": "没有",
                            "V_14": "右髂",
                            "V_15": "左髂",
                        },
                    }
                }
            }
        },
    )
    cd = _canonical()
    q = build_question(cd, spec, ev_idx=1, language="zh")
    labels = {opt.label for opt in q.options}
    assert {"没有", "右髂", "左髂"} <= labels


def test_missing_zh_key_falls_back_to_native_value_labels(spec, i18n_dir) -> None:
    cd = _canonical()
    q = build_question(cd, spec, ev_idx=1, language="zh")
    labels = {opt.label for opt in q.options}
    assert "nowhere" in labels


def test_empty_string_translation_falls_back_to_native(spec, i18n_dir) -> None:
    _write_yaml(
        i18n_dir / "zh" / "symptoms_ddxplus.yaml",
        {"symptoms": {"ddxplus": {"E_91": {"question": ""}}}},
    )
    cd = _canonical()
    q = build_question(cd, spec, ev_idx=0, language="zh")
    assert q.question == "Do you have a fever?"


def test_empty_string_value_label_falls_back_to_native(spec, i18n_dir) -> None:
    _write_yaml(
        i18n_dir / "zh" / "symptoms_ddxplus.yaml",
        {
            "symptoms": {
                "ddxplus": {
                    "E_55": {
                        "question": "",
                        "values": {"V_123": "", "V_14": "", "V_15": ""},
                    }
                }
            }
        },
    )
    cd = _canonical()
    q = build_question(cd, spec, ev_idx=1, language="zh")
    labels = {opt.label for opt in q.options}
    assert "nowhere" in labels


# --- synth_patient round-trips --------------------------------------------


def test_synth_patient_binary_yes_via_raw_value(spec, i18n_dir) -> None:
    cd = _canonical()
    payload = synth_patient(
        cd, spec, ev_idx=0, answer="是", answer_value="yes", language="zh"
    )
    assert payload["bin_pos"] == {0}


def test_synth_patient_binary_no_via_localized_label(spec, i18n_dir) -> None:
    _write_yaml(
        i18n_dir / "zh" / "common.yaml",
        {"symptoms": {"binary": {"yes": "是", "no": "否"}}},
    )
    cd = _canonical()
    payload = synth_patient(
        cd, spec, ev_idx=0, answer="否", answer_value=None, language="zh"
    )
    assert payload["bin_pos"] == set()


def test_synth_patient_categorical_via_raw_value_id(spec, i18n_dir) -> None:
    cd = _canonical()
    payload = synth_patient(
        cd, spec, ev_idx=1, answer="左髂", answer_value="V_15", language="zh"
    )
    # V_15 is local index 2 in the sorted value list [V_123, V_14, V_15].
    assert payload["cat_val"] == {1: 2}


def test_synth_patient_categorical_via_localized_label_when_raw_missing(
    spec, i18n_dir
) -> None:
    cd = _canonical()
    payload = synth_patient(
        cd,
        spec,
        ev_idx=1,
        answer="iliac wing (right)",
        answer_value=None,
        language="en",
    )
    assert payload["cat_val"] == {1: 1}


def test_synth_patient_numeric_snaps_to_nearest(spec, i18n_dir) -> None:
    cd = _canonical()
    payload = synth_patient(cd, spec, ev_idx=2, answer=3, language="en")
    # Value list is [0, 5, 10]; 3 is closer to 5 than to 0.
    assert payload["cat_val"] == {2: 1}


def test_synth_patient_unmatched_categorical_raises_422(spec, i18n_dir) -> None:
    cd = _canonical()
    with pytest.raises(QuestionPayloadError) as exc:
        synth_patient(
            cd,
            spec,
            ev_idx=1,
            answer="completely unknown label",
            answer_value=None,
            language="en",
        )
    assert exc.value.status_code == 422


# --- condition name localization ------------------------------------------


def test_localize_condition_via_i18n_key(spec, i18n_dir) -> None:
    _write_yaml(
        i18n_dir / "zh" / "symptoms_ddxplus.yaml",
        {
            "symptoms": {
                "ddxplus": {
                    "conditions": {"spontaneous_pneumothorax": {"name": "自发性气胸"}}
                }
            }
        },
    )
    cd = _canonical()
    assert (
        localize_condition(cd, spec, "spontaneous_pneumothorax", "zh") == "自发性气胸"
    )


def test_localize_condition_falls_back_to_native_name(spec, i18n_dir) -> None:
    cd = _canonical()
    assert (
        localize_condition(cd, spec, "spontaneous_pneumothorax", "zh")
        == "Spontaneous pneumothorax"
    )


def test_localize_condition_unknown_slug_raises(spec, i18n_dir) -> None:
    cd = _canonical()
    with pytest.raises(KeyError):
        localize_condition(cd, spec, "not_a_condition", "en")
