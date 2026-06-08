"""Unit tests for TranslationService.

All LLM calls use TestModel so no real network requests are made.
"""

from __future__ import annotations

import pytest
from pydantic_ai.models.test import TestModel

from claritymed.core.translation import TranslationService


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
    assert TranslationService.detect_language(text) == expected


def test_detect_language_is_static():
    """Can be called without constructing an instance."""
    assert TranslationService.detect_language("测试") == "zh"


# --- translate_query ---------------------------------------------------


async def test_translate_query_returns_model_output():
    svc = TranslationService(TestModel(custom_output_text="hemoglobin 105"))
    result = await svc.translate_query("我血红蛋白105", target_lang="en")
    assert result == "hemoglobin 105"


async def test_translate_query_zh_target():
    svc = TranslationService(TestModel(custom_output_text="血红蛋白105"))
    result = await svc.translate_query("hemoglobin 105", target_lang="zh")
    assert result == "血红蛋白105"


# --- translate_answer --------------------------------------------------


async def test_translate_answer_preserves_output():
    long_answer = "Hemoglobin of 105 g/L is mildly low. See [1] for ranges."
    svc = TranslationService(TestModel(custom_output_text="血红蛋白105克/升略低。"))
    result = await svc.translate_answer(long_answer, target_lang="zh")
    assert result == "血红蛋白105克/升略低。"


# --- translate_term ----------------------------------------------------


async def test_translate_term():
    svc = TranslationService(TestModel(custom_output_text="hemoglobin"))
    result = await svc.translate_term("血红蛋白", target_lang="en")
    assert result == "hemoglobin"


# --- translate (general) -----------------------------------------------


async def test_translate_general():
    svc = TranslationService(TestModel(custom_output_text="anemia"))
    result = await svc.translate("贫血", target_lang="en")
    assert result == "anemia"


# --- fallback on failure -----------------------------------------------


async def test_translate_falls_back_to_original_on_error():
    """If the LLM call raises, the original text is returned unchanged."""
    from pydantic_ai.models.test import TestModel as _TM

    svc = TranslationService(_TM(custom_output_text="answer"))

    # Patch _call to raise
    async def _boom(text, *, target_lang, context="general"):
        raise RuntimeError("simulated failure")

    svc._call = _boom  # type: ignore[method-assign]

    result = await svc.translate_query("我血红蛋白105", target_lang="en")
    assert result == "我血红蛋白105"


async def test_translate_returns_original_on_empty_output():
    """An empty model output falls through to the original text."""
    svc = TranslationService(TestModel(custom_output_text=""))
    result = await svc.translate_query("我血红蛋白105", target_lang="en")
    assert result == "我血红蛋白105"
