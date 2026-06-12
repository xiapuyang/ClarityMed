"""Contract tests for ``ToolApprovalChannel`` protocol + headless default."""

from __future__ import annotations

import pytest

from claritymed.core.interaction import (
    ApprovalDecision,
    HeadlessToolApprovalChannel,
    InteractiveChannelUnavailable,
    ToolApprovalChannel,
)


def test_approval_decision_defaults():
    decision = ApprovalDecision(decision="once")
    assert decision.decision == "once"
    assert decision.modified_args is None


def test_approval_decision_with_overrides():
    decision = ApprovalDecision(decision="modify", modified_args={"name": "X"})
    assert decision.modified_args == {"name": "X"}


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
