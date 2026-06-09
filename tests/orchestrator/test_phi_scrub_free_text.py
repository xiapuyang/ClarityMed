"""Tests for ``PhiGuard.scrub_free_text``.

Covers the M10 free-text PHI defense layer: regex-based scrubbing for phone,
email, ID, MRN, etc. NER pass is hook-only in v1 and skipped here.
"""

from __future__ import annotations

import importlib

import pytest

from claritymed.orchestrator import PhiGuard, ScrubReport


@pytest.fixture
def guard():
    # Reload config to pick up safety.yaml additions in tests run after edits.
    from claritymed import config as _cfg

    importlib.reload(_cfg)
    return PhiGuard.from_config()


def test_cn_phone_number_scrubbed(guard):
    text = "请联系 13800138000 安排复诊。"
    scrubbed, report = guard.scrub_free_text(text)
    assert "13800138000" not in scrubbed
    assert "[REDACTED:PHONE]" in scrubbed
    assert report.rule_hits.get("phone_cn") == 1


def test_email_scrubbed(guard):
    text = "Reach me at patient@example.com please."
    scrubbed, report = guard.scrub_free_text(text)
    assert "patient@example.com" not in scrubbed
    assert "[REDACTED:EMAIL]" in scrubbed
    assert report.rule_hits.get("email") == 1


def test_cn_id_card_scrubbed(guard):
    text = "身份证 11010119900307123X 请核对。"
    scrubbed, report = guard.scrub_free_text(text)
    assert "11010119900307123X" not in scrubbed
    assert "[REDACTED:ID]" in scrubbed
    assert report.rule_hits.get("id_cn") == 1


def test_mrn_scrubbed(guard):
    text = "Chart MRN: 1234567 was updated."
    scrubbed, report = guard.scrub_free_text(text)
    assert "1234567" not in scrubbed
    assert "[REDACTED:MRN]" in scrubbed


def test_multiple_pii_in_one_text(guard):
    text = "张三 13800138000，邮箱 a@b.cn，证件 11010119900307123X。"
    scrubbed, report = guard.scrub_free_text(text)
    assert "13800138000" not in scrubbed
    assert "a@b.cn" not in scrubbed
    assert "11010119900307123X" not in scrubbed
    assert sum(report.rule_hits.values()) >= 3


def test_no_pii_text_returns_unchanged(guard):
    text = "Blood glucose 5.6 mmol/L is within normal range."
    scrubbed, report = guard.scrub_free_text(text)
    assert scrubbed == text
    assert report.rule_hits == {}


def test_empty_string(guard):
    scrubbed, report = guard.scrub_free_text("")
    assert scrubbed == ""
    assert report.text_len_before == 0
    assert report.text_len_after == 0


def test_phone_regex_no_over_match():
    # Guard against the phone regex over-matching inside longer digit strings.
    # Uses regex-only service so the model layer (which correctly tags 16-digit
    # numbers as account_number) doesn't interfere with this boundary check.
    from claritymed.core.scrub.service import (
        FreeTextRule,
        PrivacyFilterConfig,
        ScrubConfig,
        ScrubService,
    )

    config = ScrubConfig(
        free_text_patterns=[
            FreeTextRule(
                name="phone_cn",
                regex=r"(?<!\d)1[3-9]\d{9}(?!\d)",
                replacement="[REDACTED:PHONE]",
            )
        ],
        privacy_filter=PrivacyFilterConfig(enabled=False),
    )
    svc = ScrubService(config)
    text = "Reference number 1234567890123456 should stay intact."
    scrubbed, _ = svc.scrub(text)
    assert "1234567890123456" in scrubbed


def test_report_no_original_spans_recorded(guard):
    """Audit-safety: report records counts only, not the redacted spans.

    This is enforced by the ScrubReport schema (frozen + extra=forbid + only
    int/dict fields). Verify the model has no field that could leak strings.
    """
    text = "phone 13800138000"
    _, report = guard.scrub_free_text(text)
    assert isinstance(report, ScrubReport)
    # Walk fields — none should be a str (which could contain redacted span).
    for field_name, value in report.model_dump().items():
        if isinstance(value, str):
            pytest.fail(f"ScrubReport.{field_name} is str — possible PII leak risk.")


def test_scrub_then_outbound_double_pass(guard):
    """Scrubbed text should pass clean through the structured outbound check.

    The [REDACTED:*] tokens are not new PHI; running ``check_outbound`` on a
    payload built from scrubbed text should produce no additional hits.
    """
    scrubbed, _ = guard.scrub_free_text("phone 13800138000 email a@b.cn")
    payload = {"free_text": scrubbed}
    cleaned, hits = guard.check_outbound(payload, provider_kind="cloud")
    assert cleaned == payload
    assert hits == []


def test_model_enabled_regex_always_runs(guard):
    """Regex pass always runs; phone must be scrubbed regardless of model state."""
    scrubbed, report = guard.scrub_free_text("联系 13800138000")
    assert "13800138000" not in scrubbed
    assert "[REDACTED:PHONE]" in scrubbed
