"""``ToolAnnouncedButSkippedRule`` aggregation + threshold tests."""

from __future__ import annotations

from claritymed.core.audit.rules.tool_announced import ToolAnnouncedButSkippedRule


def _mode_ask(provider: str, model: str) -> dict:
    return {
        "kind": "mode.ask",
        "payload": {"provider_id": provider, "model": model},
    }


def _skipped(provider: str, model: str, snippet: str = "我将检索") -> dict:
    return {
        "kind": "mode.ask.tool_announced_but_skipped",
        "payload": {
            "provider_id": provider,
            "model": model,
            "snippet": snippet,
            "tool": "retrieve_medical_literature",
        },
        "created_at": "2026-06-09T10:00:00Z",
    }


def test_rule_counts_per_provider_model_pair():
    rule = ToolAnnouncedButSkippedRule()
    for _ in range(10):
        rule.accept(_mode_ask("ollama", "qwen3:14b"))
    for _ in range(3):
        rule.accept(_skipped("ollama", "qwen3:14b"))
    report = rule.report()
    assert report.counts["ollama/qwen3:14b"] == 3
    assert report.total_relevant == 10


def test_rule_threshold_deterministic_recommendation():
    """At >= 10% skip rate the rule recommends deterministic mode."""
    rule = ToolAnnouncedButSkippedRule()
    for _ in range(10):
        rule.accept(_mode_ask("ollama", "qwen3:14b"))
    for _ in range(3):  # 30%
        rule.accept(_skipped("ollama", "qwen3:14b"))
    findings = rule.report().findings
    assert any("deterministic" in f for f in findings)


def test_rule_threshold_prompt_tune_recommendation():
    """5-10% range recommends tightening the prompt."""
    rule = ToolAnnouncedButSkippedRule()
    for _ in range(100):
        rule.accept(_mode_ask("ollama", "qwen3:14b"))
    for _ in range(7):  # 7%
        rule.accept(_skipped("ollama", "qwen3:14b"))
    findings = rule.report().findings
    assert any("prompt" in f.lower() for f in findings)
    assert not any("deterministic" in f for f in findings)


def test_rule_threshold_watch_below_5pct():
    """2-5% is "watch only" — no action recommended."""
    rule = ToolAnnouncedButSkippedRule()
    for _ in range(100):
        rule.accept(_mode_ask("ollama", "qwen3:14b"))
    for _ in range(3):  # 3%
        rule.accept(_skipped("ollama", "qwen3:14b"))
    findings = rule.report().findings
    assert any("watch" in f.lower() for f in findings)


def test_rule_no_findings_below_2pct():
    rule = ToolAnnouncedButSkippedRule()
    for _ in range(1000):
        rule.accept(_mode_ask("ollama", "qwen3:14b"))
    for _ in range(5):  # 0.5%
        rule.accept(_skipped("ollama", "qwen3:14b"))
    # Below the watch threshold → no findings emitted, but counts still recorded.
    report = rule.report()
    assert report.findings == []
    assert report.counts["ollama/qwen3:14b"] == 5


def test_rule_groups_separately_by_provider_and_model():
    rule = ToolAnnouncedButSkippedRule()
    for _ in range(50):
        rule.accept(_mode_ask("ollama", "qwen3:14b"))
    for _ in range(50):
        rule.accept(_mode_ask("anthropic", "claude-sonnet-4-5"))
    for _ in range(10):  # 20% on ollama
        rule.accept(_skipped("ollama", "qwen3:14b"))
    counts = rule.report().counts
    assert counts["ollama/qwen3:14b"] == 10
    assert "anthropic/claude-sonnet-4-5" in counts
    assert counts["anthropic/claude-sonnet-4-5"] == 0


def test_rule_caps_samples_at_ten():
    rule = ToolAnnouncedButSkippedRule()
    for i in range(25):
        rule.accept(_skipped("ollama", "qwen3:14b", snippet=f"sample-{i}"))
    report = rule.report()
    assert len(report.samples) == 10
    assert "sample-0" in report.samples[0]["snippet"]


def test_rule_ignores_unrelated_event_kinds():
    rule = ToolAnnouncedButSkippedRule()
    rule.accept({"kind": "rag.retrieval", "payload": {}})
    rule.accept({"kind": "ocr.extract", "payload": {}})
    rule.accept({"kind": "mode.ask", "payload": {"model": "x"}})
    report = rule.report()
    assert report.total_relevant == 1
