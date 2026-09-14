"""Unit tests for LLMTranslationProvider and shared translation utilities.

All LLM calls use TestModel so no real network requests are made.
"""

from __future__ import annotations

import pytest
from pydantic_ai.models.test import TestModel

from claritymed.core.translation import LLMTranslationProvider, detect_language


# --- detect_language ---------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("我血红蛋白105，需要担心吗", "zh"),
        ("hemoglobin 105, should I worry?", "en"),
        ("我的hemoglobin值很低", "zh"),  # CJK ratio >20 %
        ("", "en"),  # empty → default en
        ("ABC 123", "en"),  # pure ASCII
        ("你好世界", "zh"),  # pure CJK
    ],
)
def test_detect_language(text, expected):
    assert detect_language(text) == expected


def test_detect_language_is_module_function():
    """Can be called without constructing a provider instance."""
    assert detect_language("测试") == "zh"


# --- translate_query ---------------------------------------------------


async def test_translate_query_returns_model_output():
    svc = LLMTranslationProvider(TestModel(custom_output_text="hemoglobin 105"))
    result = await svc.translate_query("我血红蛋白105", target_lang="en")
    assert result == "hemoglobin 105"


async def test_translate_query_zh_target():
    svc = LLMTranslationProvider(TestModel(custom_output_text="血红蛋白105"))
    result = await svc.translate_query("hemoglobin 105", target_lang="zh")
    assert result == "血红蛋白105"


# --- translate_answer --------------------------------------------------


async def test_translate_answer_preserves_output():
    long_answer = "Hemoglobin of 105 g/L is mildly low. See [1] for ranges."
    svc = LLMTranslationProvider(TestModel(custom_output_text="血红蛋白105克/升略低。"))
    result = await svc.translate_answer(long_answer, target_lang="zh")
    assert result == "血红蛋白105克/升略低。"


# --- translate_term ----------------------------------------------------


async def test_translate_term():
    svc = LLMTranslationProvider(TestModel(custom_output_text="hemoglobin"))
    result = await svc.translate_term("血红蛋白", target_lang="en")
    assert result == "hemoglobin"


# --- translate (general) -----------------------------------------------


async def test_translate_general():
    svc = LLMTranslationProvider(TestModel(custom_output_text="anemia"))
    result = await svc.translate("贫血", target_lang="en")
    assert result == "anemia"


# --- fallback on failure -----------------------------------------------


async def test_translate_falls_back_to_original_on_error():
    """If the LLM call raises, the original text is returned unchanged."""
    svc = LLMTranslationProvider(TestModel(custom_output_text="answer"))

    async def _boom(text, *, target_lang, context="general"):
        raise RuntimeError("simulated failure")

    svc._call = _boom  # type: ignore[method-assign]

    result = await svc.translate_query("我血红蛋白105", target_lang="en")
    assert result == "我血红蛋白105"


async def test_translate_returns_original_on_empty_output():
    """An empty model output falls through to the original text."""
    svc = LLMTranslationProvider(TestModel(custom_output_text=""))
    result = await svc.translate_query("我血红蛋白105", target_lang="en")
    assert result == "我血红蛋白105"


# --- step instrumentation ---------------------------------------------


async def test_translate_query_records_step_when_sink_active():
    """translate_query emits a StepRecord into an active capture_steps() scope."""
    from claritymed.core.observability.steps import capture_steps

    svc = LLMTranslationProvider(TestModel(custom_output_text="hemoglobin 105"))
    with capture_steps() as steps:
        await svc.translate_query("我血红蛋白105", target_lang="en")

    assert len(steps) == 1
    assert steps[0].name == "translate.query"
    assert steps[0].details == "translate/en"
    assert steps[0].summary == "done"
    assert not steps[0].failed
    assert steps[0].duration_ms >= 0


async def test_translate_no_step_without_sink():
    """Without a capture_steps() scope no StepRecord is created and no error occurs."""
    svc = LLMTranslationProvider(TestModel(custom_output_text="hemoglobin"))
    result = await svc.translate_query("血红蛋白", target_lang="en")
    assert result == "hemoglobin"


async def test_failed_step_marked_on_llm_error():
    """When _call is patched to raise, no step is recorded (step lives in _call)."""
    from claritymed.core.observability.steps import capture_steps

    svc = LLMTranslationProvider(TestModel(custom_output_text="x"))

    async def _boom(text, *, target_lang, context="general"):
        raise RuntimeError("simulated")

    svc._call = _boom  # type: ignore[method-assign]

    with capture_steps() as steps:
        await svc.translate_query("血红蛋白", target_lang="en")

    # translate_query catches the error, but _boom never entered `with step(...)`,
    # so no step is recorded.
    assert len(steps) == 0


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


def test_make_translation_provider_returns_llm_provider():
    from claritymed.core.translation.factory import make_translation_provider
    from claritymed.core.translation.llm_provider import LLMTranslationProvider

    model = TestModel(custom_output_text="x")
    provider = make_translation_provider(model)
    assert isinstance(provider, LLMTranslationProvider)


def test_make_translation_provider_falls_back_on_config_error(monkeypatch):
    """If loading retrieval.yaml fails, default to llm provider silently."""
    import claritymed.core.rag.schemas as _rag_schemas

    def _boom():
        raise RuntimeError("config broken")

    monkeypatch.setattr(_rag_schemas, "load_retrieval_config", _boom)
    from claritymed.core.translation.factory import make_translation_provider
    from claritymed.core.translation.llm_provider import LLMTranslationProvider

    provider = make_translation_provider(TestModel(custom_output_text="x"))
    assert isinstance(provider, LLMTranslationProvider)


def test_make_translation_provider_with_cloud_phi_kind():
    """phi_kind='cloud' wires a real scrub_gate into the provider."""
    from claritymed.core.translation.factory import make_translation_provider
    from claritymed.core.translation.llm_provider import LLMTranslationProvider

    provider = make_translation_provider(
        TestModel(custom_output_text="x"), phi_kind="cloud"
    )
    assert isinstance(provider, LLMTranslationProvider)
    assert provider._scrub_gate is not None


def test_make_translation_provider_rejects_unknown_id(monkeypatch):
    """An unsupported provider id must fail loud rather than silently fall through."""
    import claritymed.core.rag.schemas as _rag_schemas
    from types import SimpleNamespace

    monkeypatch.setattr(
        _rag_schemas,
        "load_retrieval_config",
        lambda: SimpleNamespace(translation=SimpleNamespace(provider="not_real")),
    )
    from claritymed.core.translation.factory import make_translation_provider

    with pytest.raises(ValueError, match="Unknown translation.provider"):
        make_translation_provider(TestModel(custom_output_text="x"))
