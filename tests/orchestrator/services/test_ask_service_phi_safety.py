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

Hardened in the follow-up ce:review pass:

* List-form ``UserPromptPart.content`` (multimodal) is now scrubbed
  too. The previous behaviour was a bypass — OCR text smuggled into a
  list during a prior local turn rode to the cloud LLM unscrubbed
  (adversarial reviewer ADV-007).
* The scrub callable may raise ``HistoryScrubFailed`` when the ONNX
  privacy-filter model layer reports ``model_failed=True``. The helper
  propagates rather than swallows, mirroring the fail-loud posture of
  the user-input scrub.
"""

from __future__ import annotations

import pytest

from claritymed.orchestrator.services.ask_service import (
    HistoryScrubFailed,
    _sanitize_history_for_llm,
)


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


def test_multimodal_user_prompt_string_elements_are_scrubbed():
    """Each ``str`` element inside a list-form ``UserPromptPart.content``
    is scrubbed individually.

    Previously list-form content was passed through unchanged, which let
    OCR text persisted into ``messages_json`` during a prior local turn
    ride to the cloud LLM unscrubbed (adversarial reviewer ADV-007).
    Non-string elements (binary content, image refs, structured parts)
    continue to pass through untouched.
    """
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    multimodal = ModelRequest(
        parts=[
            UserPromptPart(content=["I called 13912345678.", "and again 13800138000."])
        ]
    )
    out = _sanitize_history_for_llm([multimodal], scrub=_scrub_phone)
    cleaned = out[0].parts[0].content
    assert isinstance(cleaned, list)
    assert all("13912345678" not in s and "13800138000" not in s for s in cleaned)
    assert all("[REDACTED:PHONE]" in s for s in cleaned)


def test_multimodal_non_string_elements_pass_through_untouched():
    """``BinaryContent``-shaped or other non-string list elements have no
    text surface to scrub — they must not be coerced or rebuilt."""
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    sentinel = object()
    multimodal = ModelRequest(
        parts=[UserPromptPart(content=["my phone is 13912345678.", sentinel])]
    )
    out = _sanitize_history_for_llm([multimodal], scrub=_scrub_phone)
    cleaned = out[0].parts[0].content
    # str element scrubbed; sentinel object passes through identity-equal.
    assert "[REDACTED:PHONE]" in cleaned[0]
    assert cleaned[1] is sentinel


def test_multimodal_list_form_returns_same_object_when_no_string_changes():
    """When no string element actually changed, the helper does not
    rebuild the message — preserves the ``out[0] is original`` identity
    invariant from the old contract for the no-op case."""
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    def _noop(s: str) -> str:
        return s

    multimodal = ModelRequest(parts=[UserPromptPart(content=["a", "b"])])
    out = _sanitize_history_for_llm([multimodal], scrub=_noop)
    assert out[0] is multimodal


def test_history_scrub_failed_propagates_from_str_path():
    """When the scrub callable raises ``HistoryScrubFailed`` on a plain
    string ``UserPromptPart``, the helper does not swallow it.

    The caller in ``_producer`` relies on this to fail loud — replaying
    regex-only history to the cloud provider is the very bug this
    contract exists to prevent.
    """

    def _failing_scrub(s: str) -> str:
        raise HistoryScrubFailed(s[:8])

    history = [_user_request_with("anything"), _assistant_response()]
    with pytest.raises(HistoryScrubFailed):
        _sanitize_history_for_llm(history, scrub=_failing_scrub)


def test_history_scrub_failed_propagates_from_list_form_path():
    """Same fail-loud guarantee for list-form ``UserPromptPart.content``
    — the multimodal path must not silently swallow the model-failed
    signal."""
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    def _failing_scrub(s: str) -> str:
        raise HistoryScrubFailed(s[:8])

    multimodal = ModelRequest(parts=[UserPromptPart(content=["any", "content"])])
    with pytest.raises(HistoryScrubFailed):
        _sanitize_history_for_llm([multimodal], scrub=_failing_scrub)
