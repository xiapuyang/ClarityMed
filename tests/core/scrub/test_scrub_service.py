"""Tests for ``ScrubService``.

Covers the two-layer pipeline: regex pass + privacy-filter model pass.
The model layer is tested via a mock pipeline so the real model weights
are not required.
"""

from __future__ import annotations

from typing import Any

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
# Model layer — _OnnxNerPipeline._aggregate (BIOES decoding, no real model)
# ---------------------------------------------------------------------------


def _make_pipeline() -> Any:
    """Return a minimal _OnnxNerPipeline stand-in exposing only _aggregate."""
    from claritymed.core.scrub.service import _OnnxNerPipeline

    # We only need _aggregate, so session/tokenizer are left as None.
    p = object.__new__(_OnnxNerPipeline)
    p._id2label = {
        0: "O",
        1: "B-private_person",
        2: "I-private_person",
        3: "E-private_person",
        4: "S-private_person",
        5: "B-private_email",
        6: "E-private_email",
    }
    return p


# "Alice" encoded as two subword tokens spanning chars 0-3 and 3-5
_ALICE_OFFSETS = [(0, 0), (0, 3), (3, 5), (5, 0)]  # CLS, Al, ice, SEP


def test_aggregate_bioes_single_token():
    """S- tag produces one span without opening a current span."""
    p = _make_pipeline()
    # "Hi" at 0-2, "Alice" single token at 3-8
    preds = [0, 0, 4]  # O, O, S-private_person
    offsets = [(0, 2), (3, 8), (9, 14)]  # not special
    spans = p._aggregate(preds, offsets)
    assert len(spans) == 1
    assert spans[0] == {"entity_group": "private_person", "start": 9, "end": 14}


def test_aggregate_bioes_b_e_pair():
    """B- then E- with same label produces one merged span."""
    p = _make_pipeline()
    preds = [1, 3]  # B-private_person, E-private_person
    offsets = [(0, 2), (3, 5)]
    spans = p._aggregate(preds, offsets)
    assert len(spans) == 1
    assert spans[0] == {"entity_group": "private_person", "start": 0, "end": 5}


def test_aggregate_bioes_b_i_e_sequence():
    """B- I- E- produces a single span covering all three tokens."""
    p = _make_pipeline()
    preds = [1, 2, 3]  # B, I, E
    offsets = [(0, 2), (2, 4), (4, 6)]
    spans = p._aggregate(preds, offsets)
    assert len(spans) == 1
    assert spans[0] == {"entity_group": "private_person", "start": 0, "end": 6}


def test_aggregate_bioes_e_without_b_creates_span():
    """E- with no open span still emits a span (graceful degradation)."""
    p = _make_pipeline()
    preds = [3]  # E-private_person, no preceding B-
    offsets = [(5, 10)]
    spans = p._aggregate(preds, offsets)
    assert len(spans) == 1
    assert spans[0]["start"] == 5
    assert spans[0]["end"] == 10


def test_aggregate_bioes_special_tokens_skipped():
    """Zero-length offset tokens (CLS/SEP) do not generate spans."""
    p = _make_pipeline()
    preds = [1, 4, 0]  # B- (CLS-like), S-, O
    offsets = [(0, 0), (1, 6), (7, 10)]  # first is special
    spans = p._aggregate(preds, offsets)
    # CLS B- should be ignored; only S- at (1,6) produces a span
    assert len(spans) == 1
    assert spans[0]["start"] == 1


def test_aggregate_bioes_multiple_spans():
    """Two separate entities produce two spans."""
    p = _make_pipeline()
    preds = [4, 0, 5, 6]  # S-person, O, B-email, E-email
    offsets = [(0, 5), (6, 8), (9, 15), (15, 20)]
    spans = p._aggregate(preds, offsets)
    assert len(spans) == 2
    assert spans[0]["entity_group"] == "private_person"
    assert spans[1]["entity_group"] == "private_email"
    assert spans[1]["start"] == 9
    assert spans[1]["end"] == 20


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


def test_apply_spans_private_date_not_redacted():
    """Dates are clinical context (onset, appointment) — they must not be scrubbed."""
    text = "Symptoms started on January 15, 2024."
    spans = [{"entity_group": "private_date", "start": 20, "end": 36}]
    result = ScrubService._apply_spans(text, spans)
    assert result == text  # unchanged


def test_apply_spans_date_skipped_but_other_labels_still_applied():
    """_SKIP_LABELS exemption is label-specific — other entities in the same pass are redacted."""
    text = "Alice seen on January 15."
    spans = [
        {"entity_group": "private_person", "start": 0, "end": 5},
        {"entity_group": "private_date", "start": 14, "end": 24},
    ]
    result = ScrubService._apply_spans(text, spans)
    assert "Alice" not in result
    assert "[REDACTED:PERSON]" in result
    assert "January 15" in result  # date preserved


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


def test_model_layer_populates_hit_types_in_report():
    """model_hit_types on ScrubReport reflects per-label counts from the pipeline."""

    class _MockPipeline:
        def __call__(self, text):
            return [
                {"entity_group": "private_person", "start": 0, "end": 5},
                {"entity_group": "private_email", "start": 10, "end": 28},
                {"entity_group": "private_person", "start": 30, "end": 35},
            ]

    svc = _service_model_enabled(_MockPipeline())
    _, report = svc.scrub("Alice at bob@example.com and Carol")
    assert report.model_hit_types == {"private_person": 2, "private_email": 1}
    assert report.model_hits == 3


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


def test_regex_layer_emits_audit_on_hits(monkeypatch):
    """scrub() emits scrub.regex audit event when regex layer finds PII."""
    captured = _patch_audit(monkeypatch)

    svc = _service_regex_only()
    svc.scrub("call me at 13554760115 or email me@example.com")

    regex_events = [e for e in captured if e["kind"] == "scrub.regex"]
    assert len(regex_events) == 1
    hits = regex_events[0]["payload"]["rule_hits"]
    assert hits["phone_cn"] == 1
    assert hits["email"] == 1


def test_regex_layer_no_audit_when_no_hits(monkeypatch):
    """scrub() does not emit scrub.regex when no PII is found."""
    captured = _patch_audit(monkeypatch)

    svc = _service_regex_only()
    svc.scrub("My hemoglobin is 105.")

    assert not any(e["kind"] == "scrub.regex" for e in captured)


def test_model_layer_emits_audit_on_success(monkeypatch):
    """_layer_model emits scrub.privacy_filter with status=ok and typed hit_types."""
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
    assert ev["payload"]["hit_types"] == {"private_person": 1}
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


# ---------------------------------------------------------------------------
# BIOES aggregation edge cases not covered above (state-machine transitions)
# ---------------------------------------------------------------------------


def test_aggregate_b_after_b_closes_previous():
    """B-X followed by B-Y closes the X span and starts a new Y span."""
    p = _make_pipeline()
    preds = [1, 5]  # B-private_person, B-private_email
    offsets = [(0, 5), (6, 11)]
    spans = p._aggregate(preds, offsets)
    assert len(spans) == 2
    assert spans[0]["entity_group"] == "private_person"
    assert spans[0]["start"] == 0 and spans[0]["end"] == 5
    assert spans[1]["entity_group"] == "private_email"


def test_aggregate_i_with_mismatched_label_starts_new_span():
    """I-Y appearing without a matching B-Y closes any open span and starts a Y span."""
    p = _make_pipeline()
    # B-private_person then I-private_email — mismatched I- treated as fresh span
    preds = [1, 6 - 4]  # B-private_person, I-private_person actually matches
    # Use B-person then I-email (id2label has no I-email; map id 6 to I-private_email)
    p._id2label[7] = "I-private_email"
    preds = [1, 7]
    offsets = [(0, 5), (6, 11)]
    spans = p._aggregate(preds, offsets)
    assert len(spans) == 2
    assert spans[0]["entity_group"] == "private_person"
    assert spans[1]["entity_group"] == "private_email"


def test_aggregate_e_with_mismatched_open_span_emits_both():
    """E-Y while a B-X span is open closes X and emits a separate Y token."""
    p = _make_pipeline()
    preds = [1, 6]  # B-private_person, E-private_email
    offsets = [(0, 5), (6, 11)]
    spans = p._aggregate(preds, offsets)
    # X span closed AND Y E-token emitted on its own
    assert len(spans) == 2
    assert spans[0]["entity_group"] == "private_person"
    assert spans[1]["entity_group"] == "private_email"
    assert spans[1]["start"] == 6 and spans[1]["end"] == 11


def test_aggregate_s_closes_existing_current():
    """S- arriving with an open B- span first closes the existing span."""
    p = _make_pipeline()
    preds = [1, 4]  # B-private_person, S-private_person
    offsets = [(0, 5), (6, 11)]
    spans = p._aggregate(preds, offsets)
    # First span closes when S- arrives, S- creates its own
    assert len(spans) == 2


def test_aggregate_o_closes_open_span():
    """O label closes any currently open span without emitting itself."""
    p = _make_pipeline()
    preds = [1, 0]  # B-private_person, O
    offsets = [(0, 5), (6, 10)]
    spans = p._aggregate(preds, offsets)
    assert len(spans) == 1
    assert spans[0]["entity_group"] == "private_person"
    assert spans[0]["end"] == 5  # NOT extended by O token


def test_aggregate_special_token_closes_open_span():
    """A special token (offset_start==offset_end) closes any open span."""
    p = _make_pipeline()
    preds = [1, 0]  # B-private_person, then a SEP-like zero-length offset
    offsets = [(0, 5), (10, 10)]
    spans = p._aggregate(preds, offsets)
    assert len(spans) == 1
    assert spans[0]["end"] == 5


def test_aggregate_emits_trailing_open_span_at_end():
    """An unclosed current span at end-of-sequence is still emitted."""
    p = _make_pipeline()
    preds = [1, 2]  # B-private_person, I-private_person — no closing tag
    offsets = [(0, 5), (5, 10)]
    spans = p._aggregate(preds, offsets)
    assert len(spans) == 1
    assert spans[0]["start"] == 0 and spans[0]["end"] == 10


# ---------------------------------------------------------------------------
# check_runtime_deps — branches by config
# ---------------------------------------------------------------------------


def test_check_runtime_deps_noop_when_disabled():
    """Disabled privacy filter → no-op, no imports attempted."""
    svc = ScrubService(ScrubConfig(privacy_filter=PrivacyFilterConfig(enabled=False)))
    # Must not raise.
    svc.check_runtime_deps()


def test_check_runtime_deps_onnx_path_passes_when_deps_installed():
    """onnxruntime + transformers are project deps, so this is the happy path."""
    svc = ScrubService(
        ScrubConfig(
            privacy_filter=PrivacyFilterConfig(
                enabled=True, onnx_file="onnx/model_q4f16.onnx"
            )
        )
    )
    svc.check_runtime_deps()  # must not raise


def test_check_runtime_deps_torch_path_passes_when_deps_installed():
    """Torch path: onnx_file=None — torch is a project dep."""
    svc = ScrubService(
        ScrubConfig(privacy_filter=PrivacyFilterConfig(enabled=True, onnx_file=None))
    )
    svc.check_runtime_deps()  # must not raise


def test_check_runtime_deps_onnx_missing_raises(monkeypatch):
    """ImportError on onnxruntime → ImportError with install hint."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("no onnxruntime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    svc = ScrubService(
        ScrubConfig(
            privacy_filter=PrivacyFilterConfig(enabled=True, onnx_file="x.onnx")
        )
    )
    with pytest.raises(ImportError, match="onnxruntime"):
        svc.check_runtime_deps()


def test_check_runtime_deps_torch_missing_raises(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    svc = ScrubService(
        ScrubConfig(privacy_filter=PrivacyFilterConfig(enabled=True, onnx_file=None))
    )
    with pytest.raises(ImportError, match="torch"):
        svc.check_runtime_deps()


def test_check_runtime_deps_transformers_missing_raises(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("no transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    svc = ScrubService(
        ScrubConfig(
            privacy_filter=PrivacyFilterConfig(
                enabled=True, onnx_file="onnx/model_q4f16.onnx"
            )
        )
    )
    with pytest.raises(ImportError, match="transformers"):
        svc.check_runtime_deps()


# ---------------------------------------------------------------------------
# ensure_downloaded — branches
# ---------------------------------------------------------------------------


def test_ensure_downloaded_noop_when_disabled():
    svc = ScrubService(ScrubConfig(privacy_filter=PrivacyFilterConfig(enabled=False)))
    assert svc.ensure_downloaded() is True


def test_ensure_downloaded_hf_hub_missing_returns_false(monkeypatch):
    """huggingface_hub not installed → log warning, return False."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            raise ImportError("missing hf hub")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    svc = ScrubService(
        ScrubConfig(
            privacy_filter=PrivacyFilterConfig(
                enabled=True, onnx_file="onnx/model.onnx"
            )
        )
    )
    assert svc.ensure_downloaded() is False


def test_ensure_downloaded_uses_cache_fast_path(monkeypatch):
    """Local cache hit → snapshot_download called once with local_files_only=True."""
    import huggingface_hub

    calls: list[dict] = []

    def fake_snapshot(repo_id, local_files_only=False, **kwargs):
        calls.append(
            {"repo_id": repo_id, "local_files_only": local_files_only, **kwargs}
        )
        return "/tmp/snapshot"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    svc = ScrubService(
        ScrubConfig(
            privacy_filter=PrivacyFilterConfig(
                enabled=True, onnx_file="onnx/model_q4f16.onnx"
            )
        )
    )
    assert svc.ensure_downloaded() is True
    assert len(calls) == 1
    assert calls[0]["local_files_only"] is True
    assert "config.json" in calls[0]["allow_patterns"]


def test_ensure_downloaded_torch_path_uses_ignore_patterns(monkeypatch):
    import huggingface_hub

    calls: list[dict] = []

    def fake_snapshot(repo_id, local_files_only=False, **kwargs):
        calls.append({"local_files_only": local_files_only, **kwargs})
        return "/tmp/snapshot"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    svc = ScrubService(
        ScrubConfig(privacy_filter=PrivacyFilterConfig(enabled=True, onnx_file=None))
    )
    assert svc.ensure_downloaded() is True
    assert "ignore_patterns" in calls[0]
    assert "*.msgpack" in calls[0]["ignore_patterns"]


def test_ensure_downloaded_falls_back_to_network_on_local_miss(monkeypatch):
    """LocalEntryNotFoundError → second snapshot_download call without local_files_only."""
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    state = {"calls": 0}

    def fake_snapshot(repo_id, local_files_only=False, **kwargs):
        state["calls"] += 1
        if local_files_only:
            raise LocalEntryNotFoundError("missing")
        return "/tmp/snapshot"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    svc = ScrubService(
        ScrubConfig(
            privacy_filter=PrivacyFilterConfig(
                enabled=True, onnx_file="onnx/model.onnx"
            )
        )
    )
    assert svc.ensure_downloaded() is True
    assert state["calls"] == 2


def test_ensure_downloaded_returns_false_on_generic_exception(monkeypatch):
    import huggingface_hub

    def fake_snapshot(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    svc = ScrubService(
        ScrubConfig(
            privacy_filter=PrivacyFilterConfig(
                enabled=True, onnx_file="onnx/model.onnx"
            )
        )
    )
    assert svc.ensure_downloaded() is False


# ---------------------------------------------------------------------------
# _get_pipeline dispatch (ONNX vs torch branch + caching)
# ---------------------------------------------------------------------------


def test_get_pipeline_dispatches_onnx_and_caches(monkeypatch):
    svc = ScrubService(
        ScrubConfig(
            privacy_filter=PrivacyFilterConfig(
                enabled=True, onnx_file="onnx/model.onnx"
            )
        )
    )
    sentinel = object()
    onnx_calls = {"n": 0}

    def fake_load_onnx(path):
        onnx_calls["n"] += 1
        return sentinel

    monkeypatch.setattr(svc, "_load_onnx_pipeline", fake_load_onnx)
    monkeypatch.setattr(svc, "_load_torch_pipeline", lambda: pytest.fail("nope"))

    assert svc._get_pipeline() is sentinel
    # Second call must not re-load.
    assert svc._get_pipeline() is sentinel
    assert onnx_calls["n"] == 1


def test_get_pipeline_dispatches_torch_when_onnx_file_none(monkeypatch):
    svc = ScrubService(
        ScrubConfig(privacy_filter=PrivacyFilterConfig(enabled=True, onnx_file=None))
    )
    sentinel = object()
    monkeypatch.setattr(svc, "_load_onnx_pipeline", lambda *a: pytest.fail("nope"))
    monkeypatch.setattr(svc, "_load_torch_pipeline", lambda: sentinel)
    assert svc._get_pipeline() is sentinel


# ---------------------------------------------------------------------------
# _load_onnx_pipeline / _load_torch_pipeline ImportError + Exception
# ---------------------------------------------------------------------------


def test_load_onnx_pipeline_returns_none_on_import_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    svc = ScrubService(
        ScrubConfig(
            privacy_filter=PrivacyFilterConfig(enabled=True, onnx_file="onnx/m.onnx")
        )
    )
    assert svc._load_onnx_pipeline("onnx/m.onnx") is None


def test_load_onnx_pipeline_returns_none_on_generic_exception(monkeypatch):
    """Any non-ImportError during ONNX load disables the layer (returns None)."""
    import huggingface_hub

    def fake_hf_download(*args, **kwargs):
        raise RuntimeError("network missing")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_download)
    svc = ScrubService(
        ScrubConfig(
            privacy_filter=PrivacyFilterConfig(enabled=True, onnx_file="onnx/m.onnx")
        )
    )
    assert svc._load_onnx_pipeline("onnx/m.onnx") is None


def test_load_torch_pipeline_returns_none_on_import_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    svc = ScrubService(
        ScrubConfig(privacy_filter=PrivacyFilterConfig(enabled=True, onnx_file=None))
    )
    assert svc._load_torch_pipeline() is None


def test_load_torch_pipeline_returns_none_on_generic_exception(monkeypatch):
    """transformers.pipeline raises both on the device and the cpu fallback → return None."""
    import transformers

    def fake_pipeline(*args, **kwargs):
        raise RuntimeError("model broken")

    monkeypatch.setattr(transformers, "pipeline", fake_pipeline)
    svc = ScrubService(
        ScrubConfig(
            privacy_filter=PrivacyFilterConfig(
                enabled=True, onnx_file=None, device="cpu"
            )
        )
    )
    # On cpu we don't fall back, so first failure → return None via outer except.
    assert svc._load_torch_pipeline() is None


def test_load_torch_pipeline_falls_back_to_cpu(monkeypatch):
    """Non-cpu device fails → retried on cpu."""
    import transformers

    state = {"attempts": []}

    def fake_pipeline(*args, **kwargs):
        device = kwargs.get("device")
        state["attempts"].append(device)
        if device != "cpu":
            raise RuntimeError("op unsupported on mps")
        return "cpu_pipe"

    monkeypatch.setattr(transformers, "pipeline", fake_pipeline)
    # Patch resolve_device to return non-cpu without touching real hardware.
    import claritymed.core.scrub.service as svc_mod

    monkeypatch.setattr(svc_mod, "resolve_device", lambda _x: "mps")

    svc = ScrubService(
        ScrubConfig(
            privacy_filter=PrivacyFilterConfig(
                enabled=True, onnx_file=None, device="auto"
            )
        )
    )
    result = svc._load_torch_pipeline()
    assert result == "cpu_pipe"
    assert state["attempts"] == ["mps", "cpu"]


# ---------------------------------------------------------------------------
# _patch_tqdm_lock — exception path
# ---------------------------------------------------------------------------


def test_patch_tqdm_lock_swallows_exception(monkeypatch, caplog):
    """If tqdm.tqdm.set_lock blows up, the helper warns instead of crashing."""
    from claritymed.core.scrub.service import _patch_tqdm_lock

    import tqdm

    def boom(_lock):
        raise RuntimeError("tqdm internals changed")

    monkeypatch.setattr(tqdm.tqdm, "set_lock", boom)

    with caplog.at_level("WARNING"):
        _patch_tqdm_lock()  # must NOT raise

    assert any("tqdm lock patch failed" in rec.message for rec in caplog.records)
