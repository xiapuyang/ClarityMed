"""Tests for ``ModeRouter``: rules, hybrid LLM fallback, accuracy on fixtures."""

from __future__ import annotations

import pytest

from claritymed import config as _cfg
from claritymed.orchestrator.router import ModeRouter, RoutingDecision


@pytest.fixture
def router():
    return ModeRouter(_cfg.load_modes_config())


def test_explicit_mode_prefix_returns_explicit(router):
    d = router.classify_rule_only("/mode ask")
    assert d.mode == "ask"
    assert d.source == "explicit"
    assert d.confidence == 1.0


def test_upload_prefix_routes_to_ingest(router):
    d = router.classify_rule_only("/upload ./report.pdf")
    assert d.mode == "ingest"
    assert d.source == "explicit"


def test_library_prefix_routes_to_rag(router):
    d = router.classify_rule_only("/library")
    assert d.mode == "rag"


def test_english_question_routes_to_ask(router):
    d = router.classify_rule_only("What does my hemoglobin level mean?")
    assert d.mode == "ask"
    assert d.confidence >= 0.9


def test_chinese_question_routes_to_ask(router):
    d = router.classify_rule_only("血红蛋白偏低是怎么回事？")
    assert d.mode == "ask"


def test_upload_verb_routes_to_ingest(router):
    d = router.classify_rule_only("上传一份化验单", has_attachment=True)
    assert d.mode == "ingest"
    assert d.confidence >= 0.9


def test_library_verb_routes_to_rag(router):
    d = router.classify_rule_only("add to my library")
    assert d.mode == "rag"


def test_ambiguous_input_returns_ambiguous(router):
    d = router.classify_rule_only("i'm not sure what to say")
    assert d.mode == "ambiguous"
    assert d.confidence < 0.5


async def test_hybrid_calls_llm_when_ambiguous(router):
    async def stub_llm(text):
        return RoutingDecision(mode="ask", confidence=0.88, source="llm", reason="stub")

    d = await router.classify("i just got a thing", llm_classify=stub_llm)
    assert d.source == "llm"
    assert d.mode == "ask"


async def test_hybrid_skips_llm_when_high_confidence(router):
    """High-confidence rule decisions short-circuit before any LLM call."""
    calls = 0

    async def spy_llm(text):
        nonlocal calls
        calls += 1
        return RoutingDecision(
            mode="ask", confidence=0.5, source="llm", reason="should not run"
        )

    d = await router.classify("/upload x", llm_classify=spy_llm)
    assert d.source == "explicit"
    assert calls == 0


async def test_hybrid_low_llm_confidence_returns_ambiguous(router):
    async def stub_llm(text):
        return RoutingDecision(mode="ask", confidence=0.3, source="llm", reason="weak")

    d = await router.classify("vague", llm_classify=stub_llm)
    assert d.mode == "ambiguous"


def test_rule_only_accuracy_on_hand_curated_samples(router):
    """The plan requires >= 85% rule-only accuracy on a 30-sample fixture."""
    samples = [
        ("/mode ask", "ask"),
        ("/mode ingest", "ingest"),
        ("/mode rag", "rag"),
        ("/upload ./x.pdf", "ingest"),
        ("/library", "rag"),
        ("/upload", "ingest"),
        ("/ask why", "ask"),
        ("what is my blood pressure target?", "ask"),
        ("how do i interpret a high creatinine?", "ask"),
        ("why does my back hurt?", "ask"),
        ("when should I worry about a fever?", "ask"),
        ("is paracetamol safe with my meds?", "ask"),
        ("can I take ibuprofen now?", "ask"),
        ("血红蛋白偏低是怎么回事？", "ask"),
        ("怎么读懂这份化验单？", "ask"),
        ("为什么我的血压在早上更高？", "ask"),
        ("是否需要复查？", "ask"),
        ("能否减少药量？", "ask"),
        ("上传这份化验报告", "ingest"),
        ("录入我的过敏史", "ingest"),
        ("save this report", "ingest"),
        ("upload my discharge summary", "ingest"),
        ("add to my records", "ingest"),
        ("存档这份病历", "ingest"),
        ("save this paper to my library", "rag"),
        ("add to my library", "rag"),
        ("收藏这篇论文", "rag"),
        ("参考资料管理", "rag"),
        ("把这篇文章存到资料库", "rag"),
        ("library", "rag"),
    ]

    correct = 0
    for text, expected in samples:
        d = router.classify_rule_only(text)
        if d.mode == expected:
            correct += 1
    accuracy = correct / len(samples)
    assert accuracy >= 0.85, f"router accuracy {accuracy:.2%} below 85% target"
