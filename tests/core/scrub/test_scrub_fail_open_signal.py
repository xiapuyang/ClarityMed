"""Regression tests for the ``model_failed`` ScrubReport flag.

Locks in the contract introduced for ce:review P1 #10 — when the privacy-
filter model layer is enabled but its pipeline cannot load or raises
during inference, ``ScrubReport.model_failed`` is True so the
orchestrator can fail loud on cloud turns rather than silently leaking
regex-only output.
"""

from __future__ import annotations

from claritymed.core.scrub.service import (
    PrivacyFilterConfig,
    ScrubConfig,
    ScrubService,
)


class _RaisingPipeline:
    """Mock pipeline that raises on every call."""

    def __call__(self, _text: str):
        raise RuntimeError("ONNX session crashed")


def test_model_failed_false_when_pipeline_disabled():
    """Regex-only config never sets model_failed."""
    svc = ScrubService(ScrubConfig(privacy_filter=PrivacyFilterConfig(enabled=False)))
    _scrubbed, report = svc.scrub("any text")
    assert report.model_failed is False


def test_model_failed_true_when_pipeline_load_returns_none():
    """When the pipeline is None (load failure), model_failed is True."""
    svc = ScrubService(ScrubConfig(privacy_filter=PrivacyFilterConfig(enabled=True)))
    # Force the lazy-load path to return None without exercising real weights.
    svc._pipeline = None
    svc._pipeline_tried = True
    _scrubbed, report = svc.scrub("My name is Alice.")
    assert report.model_failed is True
    assert report.model_hits == 0


def test_model_failed_true_when_pipeline_raises():
    """When pipeline inference raises, regex output is still returned but
    model_failed is set so callers can branch."""
    svc = ScrubService(ScrubConfig(privacy_filter=PrivacyFilterConfig(enabled=True)))
    svc._pipeline = _RaisingPipeline()
    svc._pipeline_tried = True
    scrubbed, report = svc.scrub("My name is Alice.")
    assert report.model_failed is True
    assert report.model_hits == 0
    # Regex output (no rules configured here) is returned verbatim.
    assert scrubbed == "My name is Alice."


def test_model_failed_false_on_clean_inference():
    """When the model runs to completion, model_failed stays False."""

    class _NoopPipeline:
        def __call__(self, _text: str):
            return []  # no spans detected

    svc = ScrubService(ScrubConfig(privacy_filter=PrivacyFilterConfig(enabled=True)))
    svc._pipeline = _NoopPipeline()
    svc._pipeline_tried = True
    _scrubbed, report = svc.scrub("Hello")
    assert report.model_failed is False
