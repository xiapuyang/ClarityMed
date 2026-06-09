"""Detect "tool announced but not actually invoked" patterns.

Some models in ``rag.mode=tool`` write sentences like "我将首先检索…" /
"I'll first search…" then end the turn without ever firing the tool
call — they treat retrieval as a future-turn promise. The prompt v3
explicitly forbids this, but compliance varies by model. This module
gives ``AskService`` a cheap detector so we can audit how often each
provider does it and decide whether to switch that provider to
deterministic mode.

The detector is a regex over the final answer text combined with the
per-turn tool-call counter on ``TurnState``. We accept some false
positives on idiomatic phrasing in exchange for a one-shot keyword
check that costs essentially nothing. Tune the patterns from real
audit data — do NOT preemptively cover every conceivable phrasing.
"""

from __future__ import annotations

import re

# Chinese announce patterns. ``我(将|会|要|来|去)?(先|首先)?(检索|查询|...)``
# covers "我将检索 / 我会先查询 / 我去查 / 我来检索一下" etc. Add
# observed misses here rather than guessing.
_ANNOUNCE_ZH = re.compile(
    r"我(将|会|要|来|去)?(先|首先|马上|稍后)?(检索|查询|查找|查阅|查一下|查看|了解|搜索)"
    r"|让我(先|首先)?(查|检索|查询|搜索|了解)"
    r"|稍等[,，].*?(查|检索|查询)"
)

# English announce patterns. Case-insensitive. ``i'?ll|i will|let me``
# + optional ``first/now`` + a retrieval verb captures obvious surface
# forms; the verb list is open for tuning.
_ANNOUNCE_EN = re.compile(
    r"\b(?:i(?:'ll| will|'m going to| am going to)|let me)\s+"
    r"(?:first\s+|now\s+|go\s+)?"
    r"(?:search|look\s*up|retrieve|consult|check\s+(?:the\s+)?(?:literature|guidelines|sources?))",
    re.IGNORECASE,
)


def detect_announcement(text: str) -> str | None:
    """Return the first announce snippet found in ``text``, or ``None``.

    Returning the snippet (not just a bool) lets the audit payload carry
    a human-reviewable sample so we can grow the pattern list against
    the real corpus. Callers should truncate before emitting to keep
    audit lines lean.
    """
    if not text:
        return None
    m = _ANNOUNCE_ZH.search(text)
    if m:
        return m.group(0)
    m = _ANNOUNCE_EN.search(text)
    if m:
        return m.group(0)
    return None
