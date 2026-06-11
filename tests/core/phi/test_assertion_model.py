"""Tests for ``PhiAssertionModel`` — layer 3 of the PHI defense."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)

from claritymed.context import apply_context
from claritymed.core.phi.assertion_model import PhiAssertionModel, _PHI_SAFE_ATTR
from claritymed.core.phi.guard import PhiGuard
from claritymed.errors import NonRetryableLLMError, PhiLeakDetected


class _RecordingInner:
    """Stand-in pydantic-ai Model that records calls and never raises."""

    model_name = "stub"
    system = "stub-system"

    def __init__(self):
        self.calls: list[list] = []

    async def request(
        self, messages: list, model_settings: Any, model_request_parameters: Any
    ) -> Any:
        self.calls.append(messages)
        return _FakeResponse()

    async def __aenter__(self) -> "_RecordingInner":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None


class _FakeResponse:
    finish_reason = "stop"
    usage = None


@pytest.fixture
def _ctx():
    """Provide ContextVars so audit_event doesn't refuse."""
    tokens = apply_context("20260611000000ABCDEF12", "alice", "en")
    yield
    from claritymed.context import reset_context

    reset_context(tokens)


@pytest.fixture
def guard() -> PhiGuard:
    return PhiGuard.from_config()


async def test_phi_leak_subclasses_non_retryable():
    """PhiLeakDetected must be a NonRetryableLLMError so pydantic-ai's
    retry loop terminates rather than re-issuing the same offending
    prompt N times."""
    assert issubclass(PhiLeakDetected, NonRetryableLLMError)


async def test_clean_message_passes_through(guard: PhiGuard, _ctx):
    inner = _RecordingInner()
    wrapper = PhiAssertionModel(inner, guard=guard)
    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(content="What are normal HbA1c ranges?"),
            ]
        )
    ]
    await wrapper.request(messages, None, _params())
    assert inner.calls, "inner should be called for clean messages"


async def test_phone_number_in_user_prompt_raises(guard: PhiGuard, _ctx):
    inner = _RecordingInner()
    wrapper = PhiAssertionModel(inner, guard=guard)
    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(content="Please call me at 415-555-1234"),
            ]
        )
    ]
    with pytest.raises(PhiLeakDetected):
        await wrapper.request(messages, None, _params())
    assert not inner.calls, "inner must NOT be called when PHI is detected"


async def test_email_in_tool_return_raises(guard: PhiGuard, _ctx):
    inner = _RecordingInner()
    wrapper = PhiAssertionModel(inner, guard=guard)
    messages = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="lookup",
                    tool_call_id="call-1",
                    content={"email": "alice@example.com", "result": "ok"},
                ),
            ]
        )
    ]
    with pytest.raises(PhiLeakDetected):
        await wrapper.request(messages, None, _params())


async def test_phi_safe_attribute_skips_scan(guard: PhiGuard, _ctx):
    """A part flagged ``_phi_safe=True`` bypasses the content scan —
    the false-positive escape hatch."""
    inner = _RecordingInner()
    wrapper = PhiAssertionModel(inner, guard=guard)
    part = UserPromptPart(content="Please call me at 415-555-1234")
    setattr(part, _PHI_SAFE_ATTR, True)
    msg = ModelRequest(parts=[part])
    await wrapper.request([msg], None, _params())
    assert inner.calls


async def test_phi_safe_on_message_skips_all_parts(guard: PhiGuard, _ctx):
    inner = _RecordingInner()
    wrapper = PhiAssertionModel(inner, guard=guard)
    msg = ModelRequest(
        parts=[
            UserPromptPart(content="My SSN is 123-45-6789"),
        ]
    )
    setattr(msg, _PHI_SAFE_ATTR, True)
    await wrapper.request([msg], None, _params())
    assert inner.calls


async def test_system_prompt_phi_raises(guard: PhiGuard, _ctx):
    inner = _RecordingInner()
    wrapper = PhiAssertionModel(inner, guard=guard)
    messages = [
        ModelRequest(
            parts=[
                SystemPromptPart(content="Patient phone: 415-555-1234"),
            ]
        )
    ]
    with pytest.raises(PhiLeakDetected):
        await wrapper.request(messages, None, _params())


async def test_assistant_text_history_phi_raises(guard: PhiGuard, _ctx):
    """Assistant message echoes (TextPart) are scanned too — the
    'your HGB is 105' leak path the review surfaced."""
    inner = _RecordingInner()
    wrapper = PhiAssertionModel(inner, guard=guard)
    messages = [
        ModelResponse(parts=[TextPart(content="Email me at alice@example.com")]),
    ]
    with pytest.raises(PhiLeakDetected):
        await wrapper.request(messages, None, _params())


async def test_getattr_delegates_to_inner(guard: PhiGuard):
    inner = _RecordingInner()
    wrapper = PhiAssertionModel(inner, guard=guard)
    assert wrapper.model_name == "stub"
    assert wrapper.system == "stub-system"


def _params() -> Any:
    """Minimal stand-in for ``ModelRequestParameters`` — only attribute
    access happens inside the wrapper's scan path."""

    class _Params:
        function_tools = []

    return _Params()
