"""Tests for ``TextualToolApprovalChannel``.

Uses a fake ``App`` to avoid spinning up a real Textual event loop.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from claritymed.cli.tui.tool_approval_channel import TextualToolApprovalChannel
from claritymed.core.interaction import ApprovalDecision, InteractiveChannelUnavailable


def _fake_app(result) -> MagicMock:
    app = MagicMock()
    app.push_screen_wait = AsyncMock(return_value=result)
    return app


async def test_request_returns_decision_from_modal():
    decision = ApprovalDecision(decision="once")
    app = _fake_app(decision)
    channel = TextualToolApprovalChannel(app)

    result = await channel.request("save_allergy", {"name": "penicillin"})

    assert result.decision == "once"
    app.push_screen_wait.assert_called_once()


async def test_request_none_result_returns_deny():
    """Modal dismissed without a value (None) → deny by default."""
    app = _fake_app(None)
    channel = TextualToolApprovalChannel(app)

    result = await channel.request("save_allergy", {"name": "penicillin"})

    assert result.decision == "deny"


async def test_request_propagates_tool_name_and_args():
    received_modal: list = []

    async def _push(modal):
        received_modal.append(modal)
        return ApprovalDecision(decision="always_tool")

    app = MagicMock()
    app.push_screen_wait = _push
    channel = TextualToolApprovalChannel(app, language="zh")

    result = await channel.request(
        "save_medication", {"name": "metformin"}, breadcrumb="ingest"
    )

    assert result.decision == "always_tool"
    assert len(received_modal) == 1
    modal = received_modal[0]
    from claritymed.cli.tui.modals.tool_approval_modal import ToolApprovalModal

    assert isinstance(modal, ToolApprovalModal)


async def test_request_raises_channel_unavailable_on_push_exception():
    """push_screen_wait failing → InteractiveChannelUnavailable."""
    app = MagicMock()
    app.push_screen_wait = AsyncMock(side_effect=RuntimeError("app torn down"))
    channel = TextualToolApprovalChannel(app)

    with pytest.raises(InteractiveChannelUnavailable, match="Textual tool-approval"):
        await channel.request("save_allergy", {})
