"""Pattern coverage for ``detect_announcement``.

These cases come from real model output (good cases) plus phrasings
the patterns must NOT match (negative cases) so we don't lose recall
or precision when the regex grows. Add a case before extending the
pattern list — the test is the spec.
"""

from __future__ import annotations

import pytest

from claritymed.core.observability.tool_announce import detect_announcement


# --- positives: should return a snippet --------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # The exact phrase the user screenshot showed:
        "我将首先检索关于成人贫血定义、轻度贫血的常见原因。",
        # Variations seen in other model outputs:
        "我会先查询一下相关指南。",
        "让我查阅一下医学文献。",
        "我先了解一下你的基本情况。",
        "稍等，我去查询相关参考范围。",
        "好的，我要检索一下贫血的诊断标准。",
        # English forms:
        "Let me first look up the relevant guidelines for you.",
        "I'll search the literature for normal hemoglobin ranges.",
        "I will retrieve the relevant clinical references.",
        "I'm going to check the guidelines on this.",
        "Let me consult the literature briefly.",
    ],
)
def test_announcement_phrases_detected(text: str) -> None:
    snippet = detect_announcement(text)
    assert snippet is not None
    assert snippet


@pytest.mark.parametrize(
    "text",
    [
        # Plain answer, no announcement:
        "你的血红蛋白 105 g/L 属于轻度贫血范围。",
        # Asking a question without announcing future tool use:
        "请问你的年龄和性别？这会影响参考范围的判断。",
        # English answer without announcement:
        "Your hemoglobin of 105 g/L is in the mild anemia range.",
        "What is your age and biological sex?",
        "",
    ],
)
def test_non_announcement_phrases_not_detected(text: str) -> None:
    assert detect_announcement(text) is None


def test_returns_match_snippet_for_audit_review() -> None:
    snippet = detect_announcement("我先检索一下，然后再给你答复。")
    assert snippet is not None
    assert "检索" in snippet
