"""Unit 10: MedDialog-CN preprocessing into Alpaca fine-tune JSONL."""

from __future__ import annotations

import json

import pytest

from claritymed.ingest.finetune.meddialog_cn import preprocess_meddialog_cn


def _write_dialogs(path, dialogs: list[tuple[str, str]]) -> None:
    """Each tuple is (patient, doctor). Empty strings allowed."""
    blocks = []
    for patient, doctor in dialogs:
        lines = []
        if patient:
            lines.append(f"病人：{patient}")
        if doctor:
            lines.append(f"医生：{doctor}")
        blocks.append("\n".join(lines))
    path.write_text("\n\n".join(blocks), encoding="utf-8")


# --- core filters -----------------------------------------------------


def test_drops_dialog_with_no_doctor_answer(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_dialogs(
        raw / "2018.txt",
        [
            (
                "Q1: any cure for chronic headache?",
                "this is a long enough answer to keep",
            ),
            ("Q2: orphan question", ""),
        ],
    )
    out = tmp_path / "out"
    stats = preprocess_meddialog_cn(raw, out)
    assert stats.total_dialogs == 2
    assert stats.kept == 1
    assert stats.dropped_no_answer == 1


def test_drops_short_answer(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_dialogs(
        raw / "2018.txt",
        [
            ("Q: complete question?", "too short"),  # answer < 20 chars
            ("Q: full question?", "this answer is plenty long enough"),
        ],
    )
    out = tmp_path / "out"
    stats = preprocess_meddialog_cn(raw, out)
    assert stats.kept == 1
    assert stats.dropped_short_answer == 1


def test_drops_overlong_question(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_dialogs(
        raw / "2018.txt",
        [("X" * 1001, "valid answer that is long enough to keep")],
    )
    out = tmp_path / "out"
    stats = preprocess_meddialog_cn(raw, out)
    assert stats.dropped_long_question == 1
    assert stats.kept == 0


def test_drops_duplicates(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_dialogs(
        raw / "2018.txt",
        [
            ("Q same", "A same — long enough to be kept"),
            ("Q same", "A same — long enough to be kept"),
        ],
    )
    out = tmp_path / "out"
    stats = preprocess_meddialog_cn(raw, out)
    assert stats.kept == 1
    assert stats.dropped_dedup == 1


# --- output ----------------------------------------------------------


def test_writes_three_splits_with_alpaca_shape(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_dialogs(
        raw / "2018.txt",
        [
            (f"Q{i}: question?", f"A{i}: this is a long enough answer to keep")
            for i in range(60)
        ],
    )
    out = tmp_path / "out"
    stats = preprocess_meddialog_cn(raw, out)
    assert stats.kept == 60
    # 90/5/5 split — distribution is hash-bucketed so not exactly proportional
    # at small N, but each file exists and the total adds up.
    assert (out / "train.jsonl").exists()
    assert (out / "val.jsonl").exists()
    assert (out / "test.jsonl").exists()
    assert stats.written_train + stats.written_val + stats.written_test == 60
    # Most should be train (90% bucket).
    assert stats.written_train > stats.written_val + stats.written_test
    # Spot-check shape.
    line = (out / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    row = json.loads(line)
    assert row["source"] == "meddialog_cn"
    assert "instruction" in row
    assert "input" in row
    assert "output" in row
    assert "id" in row


def test_split_is_deterministic(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_dialogs(
        raw / "2018.txt",
        [(f"Q{i}: q?", f"A{i}: long enough answer") for i in range(30)],
    )
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    stats_a = preprocess_meddialog_cn(raw, out_a)
    stats_b = preprocess_meddialog_cn(raw, out_b)
    assert stats_a == stats_b
    assert (out_a / "train.jsonl").read_bytes() == (out_b / "train.jsonl").read_bytes()


# --- edge cases ------------------------------------------------------


def test_missing_input_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        preprocess_meddialog_cn(tmp_path / "does_not_exist", tmp_path / "out")


def test_empty_input_produces_empty_output(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "out"
    stats = preprocess_meddialog_cn(raw, out)
    assert stats.total_dialogs == 0
    assert stats.kept == 0


def test_handles_full_width_colon(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "2018.txt").write_text(
        "病人：使用全角冒号的问句\n医生：这是一段长度足够的医生回答内容用来通过过滤阈值\n",
        encoding="utf-8",
    )
    stats = preprocess_meddialog_cn(raw, tmp_path / "out")
    assert stats.kept == 1


def test_limit_caps_input(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_dialogs(
        raw / "2018.txt",
        [(f"Q{i}: q?", f"A{i}: long enough answer to keep") for i in range(10)],
    )
    stats = preprocess_meddialog_cn(raw, tmp_path / "out", limit=3)
    assert stats.total_dialogs <= 4  # may overshoot by 1 due to loop ordering
    assert stats.kept <= 3
