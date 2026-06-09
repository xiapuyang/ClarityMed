"""Tests for ``ScrubService``.

Covers the two-layer pipeline: regex pass + privacy-filter model pass.
The model layer is tested via a mock pipeline so the real model weights
are not required.
"""

from __future__ import annotations

import pytest

from claritymed.core.scrub.service import (
    FreeTextRule,
    PrivacyFilterConfig,
    ScrubConfig,
    ScrubReport,
    ScrubService,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PHONE_RULE = FreeTextRule(
    name="phone_cn",
    regex=r"(?<!\d)1[3-9]\d{9}(?!\d)",
    replacement="[REDACTED:PHONE]",
)

_EMAIL_RULE = FreeTextRule(
    name="email",
    regex=r"[\w.+-]+@[\w-]+\.[\w.-]+",
    replacement="[REDACTED:EMAIL]",
)


def _service_regex_only() -> ScrubService:
    config = ScrubConfig(
        free_text_patterns=[_PHONE_RULE, _EMAIL_RULE],
        privacy_filter=PrivacyFilterConfig(enabled=False),
    )
    return ScrubService(config)


def _service_model_enabled(mock_pipeline) -> ScrubService:
    config = ScrubConfig(
        free_text_patterns=[],
        privacy_filter=PrivacyFilterConfig(enabled=True),
    )
    svc = ScrubService(config)
    # Inject mock pipeline bypassing lazy load.
    svc._pipeline = mock_pipeline
    svc._pipeline_tried = True
    return svc


# ---------------------------------------------------------------------------
# Regex layer
# ---------------------------------------------------------------------------


def test_regex_phone_scrubbed():
    svc = _service_regex_only()
    scrubbed, report = svc.scrub("联系电话 13900139000")
    assert "13900139000" not in scrubbed
    assert "[REDACTED:PHONE]" in scrubbed
    assert report.rule_hits["phone_cn"] == 1
    assert report.model_hits == 0


def test_regex_email_scrubbed():
    svc = _service_regex_only()
    scrubbed, report = svc.scrub("email me at doc@hospital.org")
    assert "doc@hospital.org" not in scrubbed
    assert "[REDACTED:EMAIL]" in scrubbed
    assert report.rule_hits["email"] == 1


def test_regex_no_pii_unchanged():
    svc = _service_regex_only()
    text = "Blood glucose 5.6 mmol/L."
    scrubbed, report = svc.scrub(text)
    assert scrubbed == text
    assert report.rule_hits == {}
    assert report.model_hits == 0


def test_empty_string():
    svc = _service_regex_only()
    scrubbed, report = svc.scrub("")
    assert scrubbed == ""
    assert report.text_len_before == 0
    assert report.text_len_after == 0


def test_report_has_no_string_fields():
    """ScrubReport must contain counts only — no field may hold raw PII."""
    svc = _service_regex_only()
    _, report = svc.scrub("phone 13800138000 email a@b.cn")
    assert isinstance(report, ScrubReport)
    for field_name, value in report.model_dump().items():
        if isinstance(value, str):
            pytest.fail(f"ScrubReport.{field_name} is str — possible PII leak")


# ---------------------------------------------------------------------------
# Model layer — _apply_spans (pure, no real model)
# ---------------------------------------------------------------------------


def test_apply_spans_replaces_person():
    text = "My name is Alice Smith and I work here."
    spans = [{"entity_group": "private_person", "start": 11, "end": 22}]
    result = ScrubService._apply_spans(text, spans)
    assert "Alice Smith" not in result
    assert "[REDACTED:PERSON]" in result


def test_apply_spans_multiple_in_reverse():
    text = "Alice at alice@example.com"
    spans = [
        {"entity_group": "private_person", "start": 0, "end": 5},
        {"entity_group": "private_email", "start": 9, "end": 25},
    ]
    result = ScrubService._apply_spans(text, spans)
    assert "Alice" not in result
    assert "alice@example.com" not in result
    assert "[REDACTED:PERSON]" in result
    assert "[REDACTED:EMAIL]" in result


def test_apply_spans_empty_returns_text():
    text = "no PII here"
    assert ScrubService._apply_spans(text, []) == text


def test_apply_spans_unknown_label_uses_generic():
    text = "secret token abc123"
    spans = [{"entity_group": "future_label", "start": 7, "end": 19}]
    result = ScrubService._apply_spans(text, spans)
    assert "token abc123" not in result
    assert "[REDACTED]" in result


# ---------------------------------------------------------------------------
# Model layer — integration via injected mock pipeline
# ---------------------------------------------------------------------------


def test_model_layer_scrubs_name():
    class _MockPipeline:
        def __call__(self, text):
            return [{"entity_group": "private_person", "start": 0, "end": 5}]

    svc = _service_model_enabled(_MockPipeline())
    scrubbed, report = svc.scrub("Alice was seen today.")
    assert "Alice" not in scrubbed
    assert "[REDACTED:PERSON]" in scrubbed
    assert report.model_hits == 1


def test_model_layer_failure_falls_back_to_regex():
    """Pipeline crash must not propagate — returns regex-only output."""

    class _BrokenPipeline:
        def __call__(self, text):
            raise RuntimeError("GPU exploded")

    config = ScrubConfig(
        free_text_patterns=[_PHONE_RULE],
        privacy_filter=PrivacyFilterConfig(enabled=True),
    )
    svc = ScrubService(config)
    svc._pipeline = _BrokenPipeline()
    svc._pipeline_tried = True

    scrubbed, report = svc.scrub("call 13800138000 Alice")
    assert "13800138000" not in scrubbed
    assert "[REDACTED:PHONE]" in scrubbed
    assert report.model_hits == 0


def test_model_layer_disabled_no_pipeline_call():
    """When privacy_filter.enabled=False, pipeline is never touched."""
    called = []

    class _ShouldNotBeCalled:
        def __call__(self, text):
            called.append(text)
            return []

    config = ScrubConfig(
        free_text_patterns=[],
        privacy_filter=PrivacyFilterConfig(enabled=False),
    )
    svc = ScrubService(config)
    svc._pipeline = _ShouldNotBeCalled()
    svc._pipeline_tried = True

    svc.scrub("Alice Smith 13800138000")
    assert called == []


# ---------------------------------------------------------------------------
# Audit emission
# ---------------------------------------------------------------------------


def _patch_audit(monkeypatch) -> list:
    """Patch audit_event on the audit module (lazy-import target) and return capture list."""
    import claritymed.core.observability.audit as _audit_mod

    captured: list = []
    monkeypatch.setattr(
        _audit_mod,
        "audit_event",
        lambda kind, payload=None: captured.append(
            {"kind": kind, "payload": payload or {}}
        ),
    )
    return captured


def test_model_layer_emits_audit_on_success(monkeypatch):
    """_layer_model emits scrub.privacy_filter with status=ok."""
    captured = _patch_audit(monkeypatch)

    class _MockPipeline:
        def __call__(self, text):
            return [{"entity_group": "private_person", "start": 0, "end": 5}]

    svc = _service_model_enabled(_MockPipeline())
    svc.scrub("Alice was seen today.")

    assert len(captured) == 1
    ev = captured[0]
    assert ev["kind"] == "scrub.privacy_filter"
    assert ev["payload"]["status"] == "ok"
    assert ev["payload"]["hits"] == 1
    assert "duration_ms" in ev["payload"]
    assert "chars_in" in ev["payload"]
    assert "chars_out" in ev["payload"]


def test_model_layer_emits_audit_on_failure(monkeypatch):
    """_layer_model emits status=error audit when the pipeline crashes."""
    captured = _patch_audit(monkeypatch)

    class _BrokenPipeline:
        def __call__(self, text):
            raise RuntimeError("inference failed")

    config = ScrubConfig(
        free_text_patterns=[],
        privacy_filter=PrivacyFilterConfig(enabled=True),
    )
    svc = ScrubService(config)
    svc._pipeline = _BrokenPipeline()
    svc._pipeline_tried = True
    svc.scrub("some text")

    assert len(captured) == 1
    assert captured[0]["payload"]["status"] == "error"
    assert "duration_ms" in captured[0]["payload"]


def test_model_layer_audit_skips_when_no_context():
    """No request context → audit is silently skipped, scrub still works."""

    class _MockPipeline:
        def __call__(self, text):
            return []

    svc = _service_model_enabled(_MockPipeline())
    scrubbed, report = svc.scrub("safe text")
    assert scrubbed == "safe text"


def test_model_layer_emits_skipped_when_pipeline_unavailable(monkeypatch):
    """pipeline=None (model not installed) still emits status=skipped audit."""
    captured = _patch_audit(monkeypatch)

    config = ScrubConfig(
        free_text_patterns=[],
        privacy_filter=PrivacyFilterConfig(enabled=True),
    )
    svc = ScrubService(config)
    svc._pipeline = None
    svc._pipeline_tried = True

    svc.scrub("some text")

    assert len(captured) == 1
    assert captured[0]["kind"] == "scrub.privacy_filter"
    assert captured[0]["payload"]["status"] == "skipped"


# ---------------------------------------------------------------------------
# from_config round-trip
# ---------------------------------------------------------------------------


def test_from_config_loads_rules():
    svc = ScrubService.from_config()
    assert len(svc._config.free_text_patterns) > 0
    assert svc._config.privacy_filter.enabled is True
    assert svc._config.privacy_filter.onnx_file == "onnx/model_q4f16.onnx"
