"""Tests for ``ClaritymedBaselineLM`` — pydantic-ai Agent → lm-eval LM bridge."""

from __future__ import annotations

import pytest
from lm_eval.api.instance import Instance
from pydantic_ai.models.test import TestModel

from claritymed.core.schemas import ProviderConfig
from claritymed.evals.lm.baseline import ClaritymedBaselineLM

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

# A provider config that satisfies the schema. ``build_model`` is patched in
# every test so the real local server is never touched — TestModel takes over.
_PROVIDER_DICT = {
    "id": "ollama",
    "kind": "local",
    "model": "qwen3:14b",
    "base_url": "http://127.0.0.1:11434/v1",
}


def _provider() -> ProviderConfig:
    return ProviderConfig.model_validate(_PROVIDER_DICT)


def _patch_model(monkeypatch, *, output_text: str) -> None:
    """Replace ``build_model`` with a TestModel returning ``output_text``."""

    def _fake_build_model(_provider):
        return TestModel(custom_output_text=output_text)

    monkeypatch.setattr(
        "claritymed.evals.lm.baseline.build_model",
        _fake_build_model,
    )


def _instance(context: str, gen_kwargs: dict | None = None, idx: int = 0) -> Instance:
    return Instance(
        request_type="generate_until",
        doc={},
        arguments=(context, gen_kwargs if gen_kwargs is not None else {}),
        idx=idx,
    )


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_returns_completion_in_input_order(monkeypatch):
    _patch_model(monkeypatch, output_text="B")
    lm = ClaritymedBaselineLM(_provider())

    requests = [_instance("Q1?"), _instance("Q2?", idx=1), _instance("Q3?", idx=2)]
    out = lm.generate_until(requests)

    assert out == ["B", "B", "B"]
    assert len(lm.latencies_ms) == 3
    assert all(ms >= 0 for ms in lm.latencies_ms)


def test_provider_id_and_model_name_exposed(monkeypatch):
    _patch_model(monkeypatch, output_text="A")
    lm = ClaritymedBaselineLM(_provider())

    assert lm.provider_id == "ollama"
    assert lm.model_name == "qwen3:14b"


# ---------------------------------------------------------------------------
# gen_kwargs → ModelSettings forwarding
# ---------------------------------------------------------------------------


def test_full_completion_returned_verbatim_no_client_truncation(monkeypatch):
    """Reasoning models emit thinking + answer in one content blob. The
    adapter must NOT truncate — the downstream filter pulls the answer
    out of the trailing portion."""
    long_text = (
        "First the model thinks: option A is interesting, option B too. "
        "After analysis, Answer: C"
    )
    _patch_model(monkeypatch, output_text=long_text)
    lm = ClaritymedBaselineLM(_provider())

    out = lm.generate_until(
        [_instance("Q?", gen_kwargs={"max_gen_toks": 2048, "temperature": 0})]
    )
    assert out == [long_text]


def test_gen_kwargs_forwarded_via_model_settings(monkeypatch):
    """``max_gen_toks`` / ``until`` / ``temperature`` go to the wire so the
    API enforces the cap, instead of us truncating after generation."""
    captured: dict = {}

    class _Recorder:
        def run_sync(self, ctx, *, model_settings=None, **_):
            captured["settings"] = model_settings

            class _R:
                output = "A"

            return _R()

    monkeypatch.setattr(
        "claritymed.evals.lm.baseline.build_model",
        lambda _p: TestModel(custom_output_text="placeholder"),
    )
    monkeypatch.setattr(
        "claritymed.evals.lm.baseline.Agent", lambda *_a, **_kw: _Recorder()
    )

    lm = ClaritymedBaselineLM(_provider())
    lm.generate_until(
        [
            _instance(
                "Q?",
                gen_kwargs={
                    "max_gen_toks": 2048,
                    "until": ["\n\n"],
                    "temperature": 0,
                },
            )
        ]
    )
    settings = captured["settings"]
    assert settings is not None
    assert settings.get("max_tokens") == 2048
    assert settings.get("stop_sequences") == ["\n\n"]
    assert settings.get("temperature") == 0.0


def test_no_gen_kwargs_means_no_model_settings(monkeypatch):
    """When the task supplies nothing, the agent runs with its default."""
    captured: dict = {}

    class _Recorder:
        def run_sync(self, ctx, *, model_settings=None, **_):
            captured["settings"] = model_settings

            class _R:
                output = "A"

            return _R()

    monkeypatch.setattr(
        "claritymed.evals.lm.baseline.build_model",
        lambda _p: TestModel(custom_output_text="placeholder"),
    )
    monkeypatch.setattr(
        "claritymed.evals.lm.baseline.Agent", lambda *_a, **_kw: _Recorder()
    )

    lm = ClaritymedBaselineLM(_provider())
    lm.generate_until([_instance("Q?", gen_kwargs={})])
    # Adapter passes no model_settings kwarg → recorder's default None
    # (i.e. the agent's own default settings would apply at runtime).
    assert captured["settings"] is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_completion(monkeypatch):
    # TestModel(custom_output_text="") triggers pydantic-ai's empty-response
    # retry, which isn't what we want to exercise here. Stub the Agent directly
    # to return an output object with an empty string.
    class _EmptyAgent:
        def run_sync(self, *_a, **_kw):
            class _R:
                output = ""

            return _R()

    monkeypatch.setattr(
        "claritymed.evals.lm.baseline.build_model",
        lambda _p: TestModel(custom_output_text="placeholder"),
    )
    monkeypatch.setattr(
        "claritymed.evals.lm.baseline.Agent",
        lambda *_a, **_kw: _EmptyAgent(),
    )
    lm = ClaritymedBaselineLM(_provider())

    out = lm.generate_until([_instance("Q?")])
    assert out == [""]


def test_completion_returned_verbatim_regardless_of_length(monkeypatch):
    """No client-side cap — the API-level ``max_tokens`` is what bounds
    generation. Whatever the agent returns lands in the output."""
    _patch_model(monkeypatch, output_text="x" * 50)
    lm = ClaritymedBaselineLM(_provider())

    out = lm.generate_until([_instance("Q?", gen_kwargs={})])
    assert out == ["x" * 50]


def test_adapter_passes_strict_mcqa_instructions_by_default(monkeypatch):
    """The default system instruction tells compliant models to reply
    with just a letter — strongest signal short of training."""
    captured: dict = {}

    def _record_agent(*args, **kwargs):
        captured["instructions"] = kwargs.get("instructions")

        class _R:
            output = "A"

        class _Fake:
            def run_sync(self, *_a, **_kw):
                return _R()

        return _Fake()

    monkeypatch.setattr(
        "claritymed.evals.lm.baseline.build_model",
        lambda _p: TestModel(custom_output_text="placeholder"),
    )
    monkeypatch.setattr("claritymed.evals.lm.baseline.Agent", _record_agent)

    ClaritymedBaselineLM(_provider())
    assert captured["instructions"] is not None
    assert "single capital letter" in captured["instructions"].lower() or (
        "single letter" in captured["instructions"].lower()
    )


def test_adapter_accepts_custom_instructions(monkeypatch):
    """Phase 3 task YAMLs may use yes/no/maybe (PubMedQA) instead of A-D —
    callers override the instruction without subclassing."""
    captured: dict = {}

    def _record_agent(*args, **kwargs):
        captured["instructions"] = kwargs.get("instructions")

        class _Fake:
            def run_sync(self, *_a, **_kw):
                class _R:
                    output = "yes"

                return _R()

        return _Fake()

    monkeypatch.setattr(
        "claritymed.evals.lm.baseline.build_model",
        lambda _p: TestModel(custom_output_text="placeholder"),
    )
    monkeypatch.setattr("claritymed.evals.lm.baseline.Agent", _record_agent)

    ClaritymedBaselineLM(_provider(), instructions="Reply with yes, no, or maybe.")
    assert captured["instructions"] == "Reply with yes, no, or maybe."


def test_adapter_accepts_no_instructions(monkeypatch):
    """``instructions=None`` runs the model without any system message."""
    captured: dict = {}

    def _record_agent(*args, **kwargs):
        captured["instructions"] = kwargs.get("instructions")

        class _Fake:
            def run_sync(self, *_a, **_kw):
                class _R:
                    output = "X"

                return _R()

        return _Fake()

    monkeypatch.setattr(
        "claritymed.evals.lm.baseline.build_model",
        lambda _p: TestModel(custom_output_text="placeholder"),
    )
    monkeypatch.setattr("claritymed.evals.lm.baseline.Agent", _record_agent)

    ClaritymedBaselineLM(_provider(), instructions=None)
    assert captured["instructions"] is None


def test_missing_gen_kwargs_falls_back_to_defaults(monkeypatch):
    _patch_model(monkeypatch, output_text="hello")
    lm = ClaritymedBaselineLM(_provider())

    # Arguments tuple with only context — runner-side weirdness, but the
    # adapter should still cope.
    req = Instance(
        request_type="generate_until",
        doc={},
        arguments=("Q?",),
        idx=0,
    )
    assert lm.generate_until([req]) == ["hello"]


def test_empty_args_rejected(monkeypatch):
    _patch_model(monkeypatch, output_text="x")
    lm = ClaritymedBaselineLM(_provider())

    req = Instance(
        request_type="generate_until",
        doc={},
        arguments=(),
        idx=0,
    )
    with pytest.raises(ValueError, match="no args"):
        lm.generate_until([req])


def test_provider_error_propagates(monkeypatch):
    class _Boom:
        def run_sync(self, *_a, **_kw):
            raise RuntimeError("upstream blew up")

    monkeypatch.setattr(
        "claritymed.evals.lm.baseline.Agent",
        lambda *_a, **_kw: _Boom(),
    )
    monkeypatch.setattr(
        "claritymed.evals.lm.baseline.build_model",
        lambda _p: TestModel(custom_output_text="ignored"),
    )

    lm = ClaritymedBaselineLM(_provider())
    with pytest.raises(RuntimeError, match="upstream blew up"):
        lm.generate_until([_instance("Q?")])

    # Latency was still recorded for the failed call — the runner can use
    # this to attribute the wall-clock budget even when the call errors.
    assert len(lm.latencies_ms) == 1


# ---------------------------------------------------------------------------
# Unsupported request types
# ---------------------------------------------------------------------------


def test_loglikelihood_raises_not_implemented(monkeypatch):
    _patch_model(monkeypatch, output_text="x")
    lm = ClaritymedBaselineLM(_provider())

    with pytest.raises(NotImplementedError, match="generate_until"):
        lm.loglikelihood([_instance("Q?")])


def test_loglikelihood_rolling_raises_not_implemented(monkeypatch):
    _patch_model(monkeypatch, output_text="x")
    lm = ClaritymedBaselineLM(_provider())

    with pytest.raises(NotImplementedError):
        lm.loglikelihood_rolling([_instance("Q?")])


# ---------------------------------------------------------------------------
# Per-request latency tracking
# ---------------------------------------------------------------------------


def test_latencies_recorded_per_request(monkeypatch):
    _patch_model(monkeypatch, output_text="A")
    lm = ClaritymedBaselineLM(_provider())

    lm.generate_until([_instance("Q1?"), _instance("Q2?")])
    assert len(lm.latencies_ms) == 2

    # Second call appends — runner can drain or replace.
    lm.generate_until([_instance("Q3?")])
    assert len(lm.latencies_ms) == 3
