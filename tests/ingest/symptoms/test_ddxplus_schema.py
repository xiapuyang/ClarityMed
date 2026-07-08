"""DDXPlus schema loader + parse_patient + prepare.py tests.

Uses tmp_path fixtures so the tests don't touch the real DDXPlus
download under ``~/.claritymed/data/symptoms/ddxplus/``. The fixture
mini-corpus exercises every evidence dtype + the missing-severity path.
"""

from __future__ import annotations

import hashlib
import json
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

from claritymed.ingest.symptoms.ddxplus.prepare import (
    DDXPLUS_SHA256,
    cmd_check,
)
from claritymed.ingest.symptoms.ddxplus.schema import (
    DDXPLUS_CONDITIONS_JSON,
    DDXPLUS_EVIDENCES_JSON,
    DDXPLUS_SPLIT_FILES,
    DdxplusSchemaError,
    Patient,
    load_evidence_schema,
    load_patients,
    load_pidx,
    parse_patient,
)

_PatientRow = namedtuple(
    "_PatientRow",
    [
        "AGE",
        "SEX",
        "PATHOLOGY",
        "EVIDENCES",
        "INITIAL_EVIDENCE",
        "DIFFERENTIAL_DIAGNOSIS",
    ],
)


def _write_evidences(data_dir: Path) -> None:
    """3 binary + 1 categorical + 1 ordinal-numeric evidence."""
    data = {
        "E_1": {"name": "E_1", "data_type": "B", "possible-values": []},
        "E_2": {"name": "E_2", "data_type": "B", "possible-values": []},
        "E_3": {"name": "E_3", "data_type": "B", "possible-values": []},
        "E_4": {
            "name": "E_4",
            "data_type": "C",
            "possible-values": ["mild", "moderate", "severe"],
        },
        "E_5": {
            "name": "E_5",
            "data_type": "C",
            "possible-values": ["0", "5", "10"],
        },
    }
    (data_dir / DDXPLUS_EVIDENCES_JSON).write_text(json.dumps(data))


def _write_conditions(data_dir: Path, *, missing_severity: bool = False) -> None:
    cond_a = {
        "condition_name": "Acute appendicitis",
        "severity": 1,
        "symptoms": {},
    }
    cond_b = {
        "condition_name": "Common cold",
        "severity": 5,
        "symptoms": {},
    }
    if missing_severity:
        del cond_b["severity"]
    payload = {"Acute appendicitis": cond_a, "Common cold": cond_b}
    (data_dir / DDXPLUS_CONDITIONS_JSON).write_text(json.dumps(payload))


# --- load_evidence_schema --------------------------------------------------


def test_load_evidence_schema_parses_fixture(tmp_path: Path) -> None:
    _write_evidences(tmp_path)
    schema = load_evidence_schema(tmp_path, use_ordinal=True)
    # 3 binary (1 slot each) + 1 categorical (1 + 3 = 4) + 1 numeric
    # categorical (1 + 3 + 1 ordinal = 5) = 12 slots total.
    assert schema["sym_size"] == 12
    assert schema["n_ev"] == 5
    # E_5 is the numeric one and should have an ordinal scalar.
    e5_idx = schema["index"]["E_5"]
    assert schema["has_ord"][e5_idx] is True


def test_load_evidence_schema_default_no_ordinal(tmp_path: Path) -> None:
    """Without --ordinal, numeric categoricals are plain one-hot."""
    _write_evidences(tmp_path)
    schema = load_evidence_schema(tmp_path)
    assert all(not h for h in schema["has_ord"])


def test_load_evidence_schema_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as exc:
        load_evidence_schema(tmp_path)
    assert "release_evidences.json" in str(exc.value)


def test_load_evidence_schema_unnamed_entry_rejected(tmp_path: Path) -> None:
    (tmp_path / DDXPLUS_EVIDENCES_JSON).write_text(
        json.dumps([{"data_type": "B", "possible-values": []}])
    )
    with pytest.raises(DdxplusSchemaError) as exc:
        load_evidence_schema(tmp_path)
    assert "name" in str(exc.value).lower()


# --- load_pidx -------------------------------------------------------------


def test_load_pidx_parses_severity(tmp_path: Path) -> None:
    _write_conditions(tmp_path)
    pidx, sev = load_pidx(tmp_path)
    assert pidx == {"Acute appendicitis": 0, "Common cold": 1}
    assert sev[pidx["Acute appendicitis"]] == 1.0
    assert sev[pidx["Common cold"]] == 5.0


def test_load_pidx_strict_severity_rejects_missing(tmp_path: Path) -> None:
    """Missing severity is fatal in strict mode (default) — would otherwise
    silently downgrade a Critical condition to the default-3 Moderate tier."""
    _write_conditions(tmp_path, missing_severity=True)
    with pytest.raises(DdxplusSchemaError) as exc:
        load_pidx(tmp_path)
    assert "severity" in str(exc.value).lower()


def test_load_pidx_non_strict_keeps_demo_default(tmp_path: Path) -> None:
    _write_conditions(tmp_path, missing_severity=True)
    pidx, sev = load_pidx(tmp_path, strict_severity=False)
    assert sev[pidx["Common cold"]] == 3.0  # default


def test_load_pidx_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_pidx(tmp_path)


def test_load_pidx_whitelist_filters_to_subset(tmp_path: Path) -> None:
    """Whitelist restricts pidx + shrinks the severity vector to match."""
    _write_conditions(tmp_path)
    pidx, sev = load_pidx(tmp_path, whitelist={"Common cold"})
    assert pidx == {"Common cold": 0}
    assert len(sev) == 1
    assert sev[0] == 5.0


def test_load_pidx_whitelist_unknown_name_raises(tmp_path: Path) -> None:
    """Typos in the whitelist fail loud — subset training must not silently
    shrink to an unintended scope because the operator misspelled a name."""
    _write_conditions(tmp_path)
    with pytest.raises(DdxplusSchemaError) as exc:
        load_pidx(tmp_path, whitelist={"Common cold", "Pneumnia"})  # typo
    assert "Pneumnia" in str(exc.value)


# --- parse_patient ---------------------------------------------------------


def test_parse_patient_encodes_all_evidence_types(tmp_path: Path) -> None:
    _write_evidences(tmp_path)
    _write_conditions(tmp_path)
    schema = load_evidence_schema(tmp_path)
    pidx, _ = load_pidx(tmp_path)
    row = _PatientRow(
        AGE=35,
        SEX="F",
        PATHOLOGY="Acute appendicitis",
        EVIDENCES=repr(["E_1", "E_4_@_severe", "E_5_@_5"]),
        INITIAL_EVIDENCE="E_1",
        DIFFERENTIAL_DIAGNOSIS=repr(
            [["Acute appendicitis", 0.7], ["Common cold", 0.1]]
        ),
    )
    p = parse_patient(row, schema, pidx)
    assert isinstance(p, Patient)
    assert 0 in p.bin_pos  # E_1 (sorted to index 0)
    # E_4 is index 3 in the sorted schema; value "severe" is local index 2.
    e4 = schema["index"]["E_4"]
    assert p.cat_val[e4] == 2
    # diff vector reflects DIFFERENTIAL_DIAGNOSIS string.
    assert p.diff[pidx["Acute appendicitis"]] == pytest.approx(0.7)
    assert p.diff[pidx["Common cold"]] == pytest.approx(0.1)
    assert p.d == pidx["Acute appendicitis"]
    assert p.sex == 1  # F


def test_parse_patient_handles_initial_with_value_token(tmp_path: Path) -> None:
    """INITIAL_EVIDENCE can carry a ``code_@_value`` form; strip the value."""
    _write_evidences(tmp_path)
    _write_conditions(tmp_path)
    schema = load_evidence_schema(tmp_path)
    pidx, _ = load_pidx(tmp_path)
    row = _PatientRow(
        AGE=30,
        SEX="M",
        PATHOLOGY="Common cold",
        EVIDENCES=repr(["E_2"]),
        INITIAL_EVIDENCE="E_4_@_mild",
        DIFFERENTIAL_DIAGNOSIS=repr([["Common cold", 1.0]]),
    )
    p = parse_patient(row, schema, pidx)
    assert p.init == schema["index"]["E_4"]


def test_parse_patient_to_dict_round_trip(tmp_path: Path) -> None:
    """Patient.to_dict produces the dict shape TypedEnv expects."""
    _write_evidences(tmp_path)
    _write_conditions(tmp_path)
    schema = load_evidence_schema(tmp_path)
    pidx, _ = load_pidx(tmp_path)
    row = _PatientRow(
        AGE=22,
        SEX="F",
        PATHOLOGY="Common cold",
        EVIDENCES=repr(["E_1"]),
        INITIAL_EVIDENCE="E_1",
        DIFFERENTIAL_DIAGNOSIS=repr([["Common cold", 1.0]]),
    )
    d = parse_patient(row, schema, pidx).to_dict()
    expected_keys = {
        "bin_pos",
        "cat_val",
        "multi_val",
        "pos",
        "init",
        "d",
        "age",
        "sex",
        "diff",
    }
    assert set(d) == expected_keys
    assert isinstance(d["diff"], np.ndarray)


# --- load_patients ---------------------------------------------------------


def _write_split_csv(data_dir: Path, split: str, rows: list[dict]) -> None:
    """Write a mini split CSV in the DDXPlus schema (columns match load_patients)."""
    import pandas as pd

    df = pd.DataFrame(
        rows,
        columns=[
            "AGE",
            "SEX",
            "PATHOLOGY",
            "EVIDENCES",
            "INITIAL_EVIDENCE",
            "DIFFERENTIAL_DIAGNOSIS",
        ],
    )
    df.to_csv(data_dir / DDXPLUS_SPLIT_FILES[split], index=False)


def test_load_patients_filters_out_of_scope_pathology(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Rows whose PATHOLOGY is outside pidx are silently dropped (subset case).

    A summary line is printed so operators can spot unexpected drops without
    a stack trace — but the return value contains only the in-scope patients.
    """
    _write_evidences(tmp_path)
    _write_conditions(tmp_path)
    _write_split_csv(
        tmp_path,
        "train",
        [
            {
                "AGE": 30,
                "SEX": "M",
                "PATHOLOGY": "Common cold",
                "EVIDENCES": repr(["E_1"]),
                "INITIAL_EVIDENCE": "E_1",
                "DIFFERENTIAL_DIAGNOSIS": repr([["Common cold", 1.0]]),
            },
            {
                "AGE": 40,
                "SEX": "F",
                "PATHOLOGY": "Acute appendicitis",  # out-of-scope under whitelist
                "EVIDENCES": repr(["E_2"]),
                "INITIAL_EVIDENCE": "E_2",
                "DIFFERENTIAL_DIAGNOSIS": repr([["Acute appendicitis", 1.0]]),
            },
        ],
    )
    schema = load_evidence_schema(tmp_path)
    pidx, _ = load_pidx(tmp_path, whitelist={"Common cold"})
    kept = load_patients(tmp_path, n=10, split="train", schema=schema, pidx=pidx)
    assert len(kept) == 1
    assert kept[0]["d"] == pidx["Common cold"]
    out = capsys.readouterr().out
    assert "Acute appendicitis" in out
    assert "filtered 1" in out


# --- prepare.py: cmd_check -------------------------------------------------


def test_cmd_check_reports_missing_files(tmp_path: Path, capsys) -> None:
    """Empty data dir → missing-file lines + non-zero exit."""
    rc = cmd_check(tmp_path)
    assert rc != 0
    captured = capsys.readouterr()
    assert "MISSING" in captured.out


def test_cmd_check_nonexistent_dir(tmp_path: Path) -> None:
    rc = cmd_check(tmp_path / "does-not-exist")
    assert rc == 2


def test_cmd_check_with_unpinned_hashes_passes(tmp_path: Path, capsys) -> None:
    """When all hashes are unpinned (placeholder values), present files
    are reported as UNVERIFIED but the check passes — so the table can
    land empty and be populated incrementally by an operator."""
    for filename in DDXPLUS_SHA256:
        (tmp_path / filename).write_bytes(b"placeholder")
    rc = cmd_check(tmp_path)
    captured = capsys.readouterr()
    assert "UNVERIFIED" in captured.out
    assert rc == 0


def test_cmd_check_pinned_hash_mismatch_fails(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    """When a pinned hash is set and the file doesn't match, fail loudly."""
    payload = b"actual content"
    real_sha = hashlib.sha256(payload).hexdigest()
    wrong_sha = "f" * 64
    monkeypatch.setitem(DDXPLUS_SHA256, DDXPLUS_EVIDENCES_JSON, wrong_sha)
    (tmp_path / DDXPLUS_EVIDENCES_JSON).write_bytes(payload)
    # write placeholders for the others so we only test this mismatch
    for filename in DDXPLUS_SHA256:
        if filename != DDXPLUS_EVIDENCES_JSON:
            (tmp_path / filename).write_bytes(b"x")
    rc = cmd_check(tmp_path)
    captured = capsys.readouterr()
    assert "MISMATCH" in captured.out
    assert real_sha in captured.out  # actual reported
    assert rc != 0
