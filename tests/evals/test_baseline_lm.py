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
# Truncation
# ---------------------------------------------------------------------------


def test_until_substring_truncates_completion(monkeypatch):
    _patch_model(monkeypatch, output_text="Answer: B\nExplanation: ...")
    lm = ClaritymedBaselineLM(_provider())

    out = lm.generate_until(
        [_instance("Q?", gen_kwargs={"until": ["\n"], "max_gen_toks": 32})]
    )
    assert out == ["Answer: B"]


def test_max_gen_toks_caps_completion(monkeypatch):
    # 100 chars of A. max_gen_toks=2 ⇒ char_cap = 2 * 4 = 8.
    _patch_model(monkeypatch, output_text="A" * 100)
    lm = ClaritymedBaselineLM(_provider())

    out = lm.generate_until(
        [_instance("Q?", gen_kwargs={"max_gen_toks": 2, "until": []})]
    )
    assert out == ["A" * 8]


def test_until_empty_no_truncation_applied(monkeypatch):
    _patch_model(monkeypatch, output_text="abc")
    lm = ClaritymedBaselineLM(_provider())

    out = lm.generate_until(
        [_instance("Q?", gen_kwargs={"until": [], "max_gen_toks": 128})]
    )
    assert out == ["abc"]


def test_until_accepts_single_string(monkeypatch):
    _patch_model(monkeypatch, output_text="A\nB")
    lm = ClaritymedBaselineLM(_provider())

    out = lm.generate_until(
        [_instance("Q?", gen_kwargs={"until": "\n", "max_gen_toks": 32})]
    )
    assert out == ["A"]


def test_first_until_match_wins(monkeypatch):
    _patch_model(monkeypatch, output_text="Answer: A\n\nMore text")
    lm = ClaritymedBaselineLM(_provider())

    out = lm.generate_until(
        [_instance("Q?", gen_kwargs={"until": ["\n\n", "\n"], "max_gen_toks": 64})]
    )
    # Either stop works; both produce "Answer: A" because \n is hit first.
    assert out == ["Answer: A"]


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


def test_default_max_gen_toks_when_absent(monkeypatch):
    _patch_model(monkeypatch, output_text="x" * 50)
    lm = ClaritymedBaselineLM(_provider())

    out = lm.generate_until([_instance("Q?", gen_kwargs={"until": []})])
    # No truncation because default 256 * 4 = 1024 chars > 50.
    assert out == ["x" * 50]


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
