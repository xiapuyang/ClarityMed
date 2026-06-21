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
    tokens = apply_context("20260611000000ABCDEF12", "test", "en")
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


# ---------------------------------------------------------------------------
# Edge branches not exercised above
# ---------------------------------------------------------------------------


async def test_scan_text_returns_false_on_empty_string(guard):
    from claritymed.core.phi.assertion_model import _scan_text_for_phi

    assert _scan_text_for_phi("", guard) is False


async def test_message_without_parts_attr_is_skipped(guard, _ctx):
    """ModelResponse with no parts attribute → loop continue branch."""
    inner = _RecordingInner()
    wrapper = PhiAssertionModel(inner, guard=guard)

    class _Bare:
        pass

    await wrapper.request([_Bare()], None, _params())  # must not raise
    assert inner.calls


async def test_tool_return_dict_content_serialized_for_scan(guard, _ctx):
    """ToolReturnPart carrying a dict is JSON-serialized so the regex layer
    still sees a phone number in nested fields."""
    inner = _RecordingInner()
    wrapper = PhiAssertionModel(inner, guard=guard)
    messages = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="lookup",
                    content={"phone": "13800138000"},
                    tool_call_id="abc",
                )
            ]
        )
    ]
    with pytest.raises(PhiLeakDetected):
        await wrapper.request(messages, None, _params())


async def test_raise_leak_swallows_audit_failure(monkeypatch, guard, caplog):
    """If audit_event raises, the PhiLeakDetected must still propagate."""
    from claritymed.core.phi import assertion_model as _am

    def _boom(*a, **kw):
        raise RuntimeError("audit broken")

    monkeypatch.setattr(_am, "audit_event", _boom)
    with caplog.at_level("WARNING"):
        with pytest.raises(PhiLeakDetected):
            _am._raise_leak("UserPromptPart")
    assert any("audit emission failed" in r.message for r in caplog.records)


async def test_inner_missing_raises_attributeerror():
    """Guard against infinite recursion when _inner is missing."""
    wrapper = PhiAssertionModel.__new__(PhiAssertionModel)
    with pytest.raises(AttributeError):
        wrapper._inner  # noqa: B018


async def test_aenter_aexit_delegate_to_inner(guard):
    class _Spy(_RecordingInner):
        def __init__(self):
            super().__init__()
            self.entered = False
            self.exited = False

        async def __aenter__(self):
            self.entered = True
            return self

        async def __aexit__(self, *args):
            self.exited = True
            return None

    inner = _Spy()
    wrapper = PhiAssertionModel(inner, guard=guard)
    async with wrapper as same:
        assert same is wrapper
    assert inner.entered and inner.exited


async def test_request_stream_raises_on_phi(guard, _ctx):
    from contextlib import asynccontextmanager

    class _StreamInner(_RecordingInner):
        @asynccontextmanager
        async def request_stream(self, *args, **kwargs):
            yield "stream-handle"

    inner = _StreamInner()
    wrapper = PhiAssertionModel(inner, guard=guard)
    messages = [ModelRequest(parts=[UserPromptPart(content="call 13800138000")])]
    with pytest.raises(PhiLeakDetected):
        async with wrapper.request_stream(messages, None, _params()) as _:
            pass


async def test_request_stream_passes_clean_message_with_run_context(guard, _ctx):
    """When clean, request_stream forwards run_context to the inner."""
    from contextlib import asynccontextmanager

    received: dict = {}

    class _StreamInner(_RecordingInner):
        @asynccontextmanager
        async def request_stream(self, messages, settings, params, run_context=None):
            received["had_run_context"] = run_context is not None
            yield "stream-handle"

    inner = _StreamInner()
    wrapper = PhiAssertionModel(inner, guard=guard)
    msgs = [ModelRequest(parts=[UserPromptPart(content="clean question")])]
    async with wrapper.request_stream(msgs, None, _params(), "ctx-sentinel") as h:
        assert h == "stream-handle"
    assert received["had_run_context"] is True


async def test_trusted_tool_return_skipped_by_name(guard, _ctx):
    """ToolReturnPart whose tool_name is in the trusted allowlist bypasses
    the content scan even when the content looks NER-suspicious. Covers
    the random-slug false positive in ``save_to_library`` returns."""
    inner = _RecordingInner()
    wrapper = PhiAssertionModel(inner, guard=guard)
    messages = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="save_to_library",
                    tool_call_id="call-x",
                    content={"library_path": "papers/2026-06-13-abc23xyz"},
                ),
            ]
        )
    ]
    await wrapper.request(messages, None, _params())  # must not raise
    assert inner.calls


async def test_trusted_tool_return_marker_persists(guard, _ctx):
    """First scan must set ``_phi_safe`` on the part so subsequent scans
    short-circuit via the standard marker path (no allowlist re-check)."""
    inner = _RecordingInner()
    wrapper = PhiAssertionModel(inner, guard=guard)
    part = ToolReturnPart(
        tool_name="save_record",
        tool_call_id="call-y",
        content={"record_path": "labs/2026-06-13-jq3mz8tx"},
    )
    messages = [ModelRequest(parts=[part])]
    await wrapper.request(messages, None, _params())
    assert getattr(part, _PHI_SAFE_ATTR, False) is True


async def test_untrusted_tool_return_still_scans(guard, _ctx):
    """A tool_name outside the allowlist still gets content-scanned —
    the allowlist is an opt-in promise, not a default."""
    inner = _RecordingInner()
    wrapper = PhiAssertionModel(inner, guard=guard)
    messages = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="external_lookup",
                    tool_call_id="call-z",
                    content={"phone": "13800138000"},
                ),
            ]
        )
    ]
    with pytest.raises(PhiLeakDetected):
        await wrapper.request(messages, None, _params())


async def test_request_stream_passes_clean_message_without_run_context(guard, _ctx):
    from contextlib import asynccontextmanager

    received: dict = {}

    class _StreamInner(_RecordingInner):
        @asynccontextmanager
        async def request_stream(self, messages, settings, params):
            received["called"] = True
            yield "stream-handle"

    inner = _StreamInner()
    wrapper = PhiAssertionModel(inner, guard=guard)
    msgs = [ModelRequest(parts=[UserPromptPart(content="clean question")])]
    async with wrapper.request_stream(msgs, None, _params(), None) as h:
        assert h == "stream-handle"
    assert received["called"] is True
