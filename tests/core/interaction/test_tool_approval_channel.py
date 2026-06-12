"""Contract tests for ``ToolApprovalChannel`` protocol + headless default."""

from __future__ import annotations

import pytest

from claritymed.core.interaction import (
    ApprovalDecision,
    HeadlessToolApprovalChannel,
    InteractiveChannelUnavailable,
    ToolApprovalChannel,
)


def test_approval_decision_once():
    decision = ApprovalDecision(decision="once")
    assert decision.decision == "once"


def test_approval_decision_always_tool():
    decision = ApprovalDecision(decision="always_tool")
    assert decision.decision == "always_tool"


def test_approval_decision_deny():
    decision = ApprovalDecision(decision="deny")
    assert decision.decision == "deny"


def test_headless_channel_implements_protocol():
    assert isinstance(HeadlessToolApprovalChannel(), ToolApprovalChannel)


async def test_headless_channel_raises_unavailable():
    channel = HeadlessToolApprovalChannel()
    with pytest.raises(InteractiveChannelUnavailable):
        await channel.request("save_allergy", {"substance": "penicillin"})


async def test_headless_channel_breadcrumb_ignored():
    channel = HeadlessToolApprovalChannel()
    with pytest.raises(InteractiveChannelUnavailable):
        await channel.request("save_record", {}, breadcrumb="Tool 1/3")
