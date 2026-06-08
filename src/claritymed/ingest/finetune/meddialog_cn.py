"""MedDialog-CN cleaning + Alpaca-style JSONL emission.

Input shape (per the public release): one dialog per file, lines tagged
``病人：…`` / ``医生：…``. We pair the first ``病人`` question with the first
``医生`` response in each dialog and emit::

    {"instruction": "...", "input": "", "output": "...",
     "source": "meddialog_cn", "id": "..."}

Filtering rules:

* Drop dialog if no doctor answer found.
* Drop if doctor answer < ``MIN_ANSWER_CHARS`` (default 20) — too short
  to be informative.
* Drop if patient question > ``MAX_QUESTION_CHARS`` (default 1000) —
  almost always indicates an error in upstream segmentation.
* Drop near-duplicates by SHA-1 of (question, answer).

Output is split 90 / 5 / 5 across train / val / test. Splits are
deterministic given the same input ordering (we hash on dialog id, not
randomly sample).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MIN_ANSWER_CHARS = 20
MAX_QUESTION_CHARS = 1000
DEFAULT_INSTRUCTION = "请根据病人的描述，给出专业的医学回复。"

# Tolerate both ASCII colon and CJK full-width colon (released data uses both).
_PATIENT_RE = re.compile(r"^\s*病人\s*[:：]\s*(.+?)\s*$")
_DOCTOR_RE = re.compile(r"^\s*医生\s*[:：]\s*(.+?)\s*$")


@dataclass(frozen=True)
class MedDialogStats:
    """Per-run counts. Returned to the CLI for stdout summary."""

    total_dialogs: int
    kept: int
    dropped_no_answer: int
    dropped_short_answer: int
    dropped_long_question: int
    dropped_dedup: int
    written_train: int
    written_val: int
    written_test: int


@dataclass(frozen=True)
class _Pair:
    dialog_id: str
    question: str
    answer: str


def preprocess_meddialog_cn(
    input_dir: Path,
    output_dir: Path,
    *,
    limit: int | None = None,
) -> MedDialogStats:
    """Read every ``*.txt`` in ``input_dir``, clean, split, write JSONL.

    Each input file is one dialog; the dialog id is the file stem so a
    line of audit can refer back to the source.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"MedDialog-CN input dir {input_dir} not found")
    output_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    counts = {
        "total": 0,
        "no_answer": 0,
        "short_answer": 0,
        "long_question": 0,
        "dedup": 0,
    }
    accepted: list[_Pair] = []

    for path in sorted(input_dir.glob("*.txt")):
        if limit is not None and counts["total"] >= limit:
            break
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise UnicodeDecodeError(
                exc.encoding,
                exc.object,
                exc.start,
                exc.end,
                f"{path.name}: input must be UTF-8 (got {exc.encoding})",
            ) from exc

        for dialog_id, pair in _iter_dialogs(text, path.stem):
            counts["total"] += 1
            if limit is not None and counts["total"] > limit:
                break
            verdict = _classify(pair, seen_hashes)
            if verdict == "ok":
                accepted.append(pair)
            elif verdict == "no_answer":
                counts["no_answer"] += 1
            elif verdict == "short_answer":
                counts["short_answer"] += 1
            elif verdict == "long_question":
                counts["long_question"] += 1
            elif verdict == "dedup":
                counts["dedup"] += 1
            # dialog_id is unused once classified but kept for log readability.
            _ = dialog_id

    train, val, test = _split_90_5_5(accepted)
    _write_jsonl(output_dir / "train.jsonl", train)
    _write_jsonl(output_dir / "val.jsonl", val)
    _write_jsonl(output_dir / "test.jsonl", test)

    return MedDialogStats(
        total_dialogs=counts["total"],
        kept=len(accepted),
        dropped_no_answer=counts["no_answer"],
        dropped_short_answer=counts["short_answer"],
        dropped_long_question=counts["long_question"],
        dropped_dedup=counts["dedup"],
        written_train=len(train),
        written_val=len(val),
        written_test=len(test),
    )


# --- internals -----------------------------------------------------


def _iter_dialogs(text: str, file_stem: str):
    """Yield ``(dialog_id, _Pair)`` per dialog separated by blank lines."""
    blocks = re.split(r"\n\s*\n", text.strip())
    for i, block in enumerate(blocks):
        question, answer = _first_qa(block)
        dialog_id = f"{file_stem}#d{i}"
        yield dialog_id, _Pair(dialog_id=dialog_id, question=question, answer=answer)


def _first_qa(block: str) -> tuple[str, str]:
    question = ""
    answer = ""
    for line in block.splitlines():
        if not question:
            m = _PATIENT_RE.match(line)
            if m:
                question = m.group(1).strip()
                continue
        if question and not answer:
            m = _DOCTOR_RE.match(line)
            if m:
                answer = m.group(1).strip()
                break
    return question, answer


def _classify(pair: _Pair, seen_hashes: set[str]) -> str:
    if not pair.answer:
        return "no_answer"
    if len(pair.answer) < MIN_ANSWER_CHARS:
        return "short_answer"
    if len(pair.question) > MAX_QUESTION_CHARS:
        return "long_question"
    h = hashlib.sha1(f"{pair.question}\n{pair.answer}".encode("utf-8")).hexdigest()
    if h in seen_hashes:
        return "dedup"
    seen_hashes.add(h)
    return "ok"


def _split_90_5_5(pairs: list[_Pair]) -> tuple[list[_Pair], list[_Pair], list[_Pair]]:
    """Hash-bucket split — deterministic, not random."""
    train: list[_Pair] = []
    val: list[_Pair] = []
    test: list[_Pair] = []
    for p in pairs:
        bucket = int(hashlib.sha1(p.dialog_id.encode()).hexdigest(), 16) % 100
        if bucket < 90:
            train.append(p)
        elif bucket < 95:
            val.append(p)
        else:
            test.append(p)
    return train, val, test


def _write_jsonl(path: Path, pairs: list[_Pair]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for p in pairs:
            row = {
                "id": p.dialog_id,
                "instruction": DEFAULT_INSTRUCTION,
                "input": p.question,
                "output": p.answer,
                "source": "meddialog_cn",
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
