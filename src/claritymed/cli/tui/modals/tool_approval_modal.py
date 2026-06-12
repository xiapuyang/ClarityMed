"""Tool approval modal — the v1 PHI write gate's UI.

The modal renders a single tool's name + args and accepts one of five
decisions: ``once`` / ``always_tool`` / ``always_pattern`` / ``modify`` /
``deny``. Multi-tool LLM responses are walked sequentially through this
modal under an asyncio.Lock owned by the agent loop (Unit 7's review-
driven scope cut: no BatchApprovalModal in v1).

Wire-up:

* ``AskService`` (Unit 8) catches ``DeferredToolRequests`` from
  pydantic-ai, iterates the tool calls, and ``push_screen_wait`` a
  ``ToolApprovalModal`` per call. The dispatcher receives the decision,
  records it in ``SettingsStore`` (for ``always_tool`` / ``always_pattern``),
  and returns the structured ``ApprovalDecision`` back to the agent
  loop as a ``ToolApproved`` / ``ToolDenied`` / ``ToolApproved`` (with
  override_args) entry in a ``DeferredToolResults`` payload.

This is the minimum-viable Unit 7 surface. The per-field ``modify_args``
form, the deletion-only ``[y/n]`` shape, and the multi-tool breadcrumb
(``Tool N/M``) are spelled out in the plan but kept for follow-up to
keep the unit ship-able in one session.
"""

from __future__ import annotations

import json
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from claritymed.core.interaction.tool_approval_channel import (
    ApprovalDecision,
    Decision,
)

__all__ = ["ApprovalDecision", "Decision", "ToolApprovalModal"]


class ToolApprovalModal(ModalScreen[ApprovalDecision]):
    """Single-tool approval gate. Modal dismisses with an ``ApprovalDecision``.

    Keyboard:
      * ``y`` — allow once.
      * ``a`` — always allow this tool (any args), TTL 7d.
      * ``p`` — always allow this tool with this args shape, TTL 7d.
      * ``m`` — modify args (v1: returns ``modify`` with no edits;
        Unit 10's LibraryView wires the per-field form).
      * ``n`` / ``escape`` — deny once.
    """

    DEFAULT_CSS = """
    ToolApprovalModal {
        align: center middle;
    }
    ToolApprovalModal > Vertical {
        background: $surface;
        border: thick $primary;
        padding: 1 2;
        width: 80;
        height: auto;
    }
    ToolApprovalModal Horizontal#buttons {
        align-horizontal: right;
        height: auto;
        margin-top: 1;
    }
    """

    # Tools that destroy data are restricted to ``y``/``n`` only — an
    # always-allow rule for ``delete_record`` would let the LLM silently
    # destroy records on subsequent turns with no confirmation. The
    # modal sniffs ``tool_name`` at compose time and hides ``a`` / ``p``
    # for these.
    _DESTRUCTIVE_TOOLS: frozenset[str] = frozenset({"delete_record"})

    BINDINGS = [
        ("escape", "deny", "Deny"),
        ("y", "once", "Once"),
        ("a", "always_tool", "Always tool"),
        ("p", "always_pattern", "Always pattern"),
        ("m", "modify", "Modify"),
        ("n", "deny", "Deny"),
    ]

    def __init__(
        self,
        tool_name: str,
        args: dict[str, Any],
        *,
        breadcrumb: str | None = None,
    ) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._args = args
        self._breadcrumb = breadcrumb
        self._destructive = tool_name in self._DESTRUCTIVE_TOOLS

    def compose(self) -> ComposeResult:
        title = (
            f"{self._breadcrumb}: {self._tool_name}"
            if self._breadcrumb
            else f"Tool: {self._tool_name}"
        )
        with Vertical():
            yield Label(title, id="title")
            yield Static(self._render_args(), id="args")
            yield Label("Proceed?")
            with Horizontal(id="buttons"):
                yield Button("Deny (n)", id="deny")
                # Destructive ops cannot have rules persisted: every
                # delete must be confirmed individually so a stale
                # always-allow rule cannot silently wipe data on a
                # later turn.
                if not self._destructive:
                    yield Button("Modify (m)", id="modify")
                    yield Button("Always pat. (p)", id="always_pattern")
                    yield Button("Always tool (a)", id="always_tool")
                yield Button("Once (y)", id="once", variant="primary")

    def _render_args(self) -> str:
        # Keep PHI text inside the modal — never logged outside the TUI.
        return json.dumps(self._args, indent=2, ensure_ascii=False)

    # --- actions -----------------------------------------------------

    def action_once(self) -> None:
        self.dismiss(ApprovalDecision(decision="once"))

    def action_always_tool(self) -> None:
        if self._destructive:
            return  # destructive tools: y/n only
        self.dismiss(ApprovalDecision(decision="always_tool"))

    def action_always_pattern(self) -> None:
        if self._destructive:
            return
        self.dismiss(ApprovalDecision(decision="always_pattern"))

    def action_modify(self) -> None:
        if self._destructive:
            return
        # Stub: minimal modal returns ``modify`` without an edit form.
        # The per-field form is Unit 10's LibraryView companion.
        self.dismiss(ApprovalDecision(decision="modify"))

    def action_deny(self) -> None:
        self.dismiss(ApprovalDecision(decision="deny"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or "deny"
        if button_id == "once":
            self.action_once()
        elif button_id == "always_tool":
            self.action_always_tool()
        elif button_id == "always_pattern":
            self.action_always_pattern()
        elif button_id == "modify":
            self.action_modify()
        else:
            self.action_deny()
