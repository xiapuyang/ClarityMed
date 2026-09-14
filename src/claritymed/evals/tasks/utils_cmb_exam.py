"""Dataset preprocessor for the CMB-Exam Chinese MCQA task.

FreedomIntelligence/CMB stores each row's options as a JSON-encoded
string in a single ``option`` field; the YAML's Jinja template can't
parse JSON. We flatten the options into ``option_a``..``option_e`` here
and drop the multi-choice (``多项选择题``) rows so the exact_match scorer's
single-letter gold target stays well-defined. The original ``answer``
field is already a single letter (e.g. ``"D"``) for single-choice rows.
"""

from __future__ import annotations

import json
from typing import Any

import datasets

# CMB question_type field values. The val/train splits also contain
# 多项选择题 (multi-answer, answer is a letter set like "BCDE") which
# don't fit the single-letter exact_match scorer and are filtered out.
_SINGLE_CHOICE = "单项选择题"


def _flatten_options(raw: Any) -> dict[str, str]:
    """Return ``option_a``..``option_e`` from the raw JSON-or-dict field.

    CMB rows store options as a JSON string keyed by uppercase letters.
    Some questions have 4 options (A–D), some have 5 (A–E). Missing
    letters become empty strings so the YAML template can conditionally
    render ``(E)`` only when populated.
    """
    if isinstance(raw, str):
        parsed = json.loads(raw)
    else:
        parsed = raw or {}
    return {
        "option_a": parsed.get("A", ""),
        "option_b": parsed.get("B", ""),
        "option_c": parsed.get("C", ""),
        "option_d": parsed.get("D", ""),
        "option_e": parsed.get("E", ""),
    }


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Filter to single-choice rows and flatten the JSON ``option`` field."""

    def _is_single(doc: dict[str, Any]) -> bool:
        return doc.get("question_type") == _SINGLE_CHOICE

    def _flatten(doc: dict[str, Any]) -> dict[str, Any]:
        return _flatten_options(doc.get("option"))

    return dataset.filter(_is_single).map(_flatten)
