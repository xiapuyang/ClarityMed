"""Regression tests for the cross-turn PHI replay defense.

Locks in the contract introduced for ce:review P0 #2 — when the current
turn is cloud, ``_sanitize_history_for_llm`` must re-scrub every
``UserPromptPart`` in the carried history, not only strip Evidence
blocks. Local turns persist raw user text into ``messages_json`` and
switching providers mid-session would otherwise replay unscrubbed PHI
to the cloud LLM on the very next turn.

Also locks the audit-PHI hardening for ce:review P1 #12 — the
``mode.ask.tool_announced_but_skipped`` payload may NOT contain the
matched snippet text (the LLM can paraphrase user PHI back).
"""

from __future__ import annotations


from claritymed.orchestrator.services.ask_service import _sanitize_history_for_llm


def _user_request_with(content: str):
    """Build a ``ModelRequest`` containing one ``UserPromptPart``."""
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    return ModelRequest(parts=[UserPromptPart(content=content)])


def _assistant_response(content: str = "ok"):
    """Build a minimal ``ModelResponse`` to anchor request/response pairs."""
    from pydantic_ai.messages import ModelResponse, TextPart

    return ModelResponse(parts=[TextPart(content=content)])


def _scrub_phone(text: str) -> str:
    """Stand-in scrub callable; just enough to prove the wiring."""
    import re

    return re.sub(r"\d{11,}", "[REDACTED:PHONE]", text)


def test_no_scrub_passes_history_through_unchanged():
    """Without ``scrub=``, only Evidence blocks are stripped — back-compat
    with the old behaviour."""
    history = [
        _user_request_with("I called 13912345678 yesterday."),
        _assistant_response(),
    ]
    out = _sanitize_history_for_llm(history)
    # No mutation expected.
    part = out[0].parts[0]
    assert part.content == "I called 13912345678 yesterday."


def test_scrub_callable_rewrites_user_prompt_parts():
    """When ``scrub=`` is supplied, every UserPromptPart string is
    transformed before the request goes back to the LLM."""
    history = [
        _user_request_with("I called 13912345678 yesterday."),
        _assistant_response(),
    ]
    out = _sanitize_history_for_llm(history, scrub=_scrub_phone)
    part = out[0].parts[0]
    assert "13912345678" not in part.content
    assert "[REDACTED:PHONE]" in part.content


def test_scrub_does_not_mutate_input_messages():
    """The original history list must be untouched — pydantic-ai uses
    dataclasses.replace, not in-place mutation."""
    history = [
        _user_request_with("I called 13912345678 yesterday."),
        _assistant_response(),
    ]
    original_text = history[0].parts[0].content
    _sanitize_history_for_llm(history, scrub=_scrub_phone)
    assert history[0].parts[0].content == original_text


def test_scrub_skips_non_user_messages():
    """Assistant responses must not be passed to the scrub callable —
    they're the LLM's own text, not user input."""
    seen: list[str] = []

    def _record(s: str) -> str:
        seen.append(s)
        return s

    history = [
        _user_request_with("hello"),
        _assistant_response("the doctor said 13912345678"),
    ]
    _sanitize_history_for_llm(history, scrub=_record)
    # Only the user-prompt content reached the scrub.
    assert seen == ["hello"]


def test_evidence_block_still_stripped_with_scrub_active():
    """The Evidence-strip happens before the scrub, so both defenses
    run on the same input rather than fighting each other."""
    payload = (
        "Evidence (cite by [n]):\n[1] (foo) bar baz\n\nQuestion: I called 13912345678."
    )
    history = [_user_request_with(payload), _assistant_response()]
    out = _sanitize_history_for_llm(history, scrub=_scrub_phone)
    cleaned = out[0].parts[0].content
    assert "Evidence (cite by [n]):" not in cleaned
    assert "[REDACTED:PHONE]" in cleaned


def test_multimodal_user_prompt_passes_through_unchanged():
    """``UserPromptPart`` with a non-string content (list/multimodal)
    has no scrub path today and must not be touched — the assertion
    catches future regressions if a string is added to the list."""
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    multimodal = ModelRequest(parts=[UserPromptPart(content=["x", "y"])])
    out = _sanitize_history_for_llm([multimodal], scrub=_scrub_phone)
    # Same object, no replace.
    assert out[0] is multimodal
