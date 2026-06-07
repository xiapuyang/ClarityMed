"""Tests for ``LabPanel`` / ``LabValue`` / ``ReferenceRange``."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from claritymed.core.schemas import LabPanel, LabValue, ReferenceRange


def _value(**overrides) -> LabValue:
    payload = {
        "name": "glucose",
        "value": 5.4,
        "unit": "mmol/L",
        "loinc": "2345-7",
        "reference_range": ReferenceRange(low=3.9, high=5.6, source="report_provided"),
        "flag": "normal",
        "ocr_confidence": 0.97,
    }
    payload.update(overrides)
    return LabValue(**payload)


def _panel(**overrides) -> LabPanel:
    payload = {
        "user_id": "alice",
        "collected_at": datetime(2026, 6, 1, tzinfo=timezone.utc),
        "values": [_value()],
        "source_doc": "uploads/lab_2026_06_01.pdf",
    }
    payload.update(overrides)
    return LabPanel(**payload)


def test_happy_path():
    p = _panel()
    assert p.values[0].value == 5.4


def test_qualitative_value_with_unknown_flag():
    v = _value(value="positive", flag="unknown", reference_range=None)
    assert v.value == "positive"


def test_ocr_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        _value(ocr_confidence=1.2)


def test_reference_range_low_gt_high_rejected():
    with pytest.raises(ValidationError):
        ReferenceRange(low=10.0, high=5.0, source="report_provided")


def test_empty_values_rejected():
    with pytest.raises(ValidationError):
        _panel(values=[])


def test_user_id_pattern_enforced():
    with pytest.raises(ValidationError):
        _panel(user_id="../escape")
