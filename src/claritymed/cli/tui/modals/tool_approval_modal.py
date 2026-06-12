"""Tool approval modal — the PHI write gate's UI.

The modal renders a single tool's name + human-readable description and
accepts one of three decisions: ``once`` / ``always_tool`` / ``deny``.
Multi-tool LLM responses are walked sequentially through this modal under
an asyncio.Lock owned by the agent loop.

Wire-up:

* ``AskService`` catches ``DeferredToolRequests`` from pydantic-ai,
  iterates the tool calls, and ``push_screen_wait``s a
  ``ToolApprovalModal`` per call. The dispatcher receives the decision,
  records it in ``SettingsStore`` (for ``always_tool``), and returns the
  structured ``ApprovalDecision`` back to the agent loop.
"""

from __future__ import annotations

from typing import Any

from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from claritymed.core.interaction.tool_approval_channel import (
    ApprovalDecision,
    Decision,
)

__all__ = ["ApprovalDecision", "Decision", "ToolApprovalModal"]

# Human-readable summaries for the seven ingest tools.
# Each lambda receives the raw args dict and returns a one-line description.
_TOOL_SUMMARY: dict[str, Any] = {
    "update_profile_field": lambda a: (
        f"Update profile: {a.get('field', '?')} → {a.get('value', '?')}"
    ),
    "save_allergy": lambda a: (
        f"Save allergy: {a.get('substance', '?')} ({a.get('severity', '?')})"
    ),
    "save_condition": lambda a: f"Save condition: {a.get('display', '?')}",
    "save_medication": lambda a: (
        f"Save medication: {a.get('name', '?')}"
        + (f"  {a['dose']}" if a.get("dose") else "")
    ),
    "save_record": lambda a: (f"Save {a.get('kind', 'record')}: {a.get('title', '?')}"),
    "save_to_library": lambda a: f"Save to library: {a.get('title', '?')}",
    "delete_record": lambda a: f"Delete record: {a.get('record_path', '?')}",
}


def _human_description(tool_name: str, args: dict[str, Any]) -> str:
    renderer = _TOOL_SUMMARY.get(tool_name)
    if renderer is not None:
        try:
            return renderer(args)
        except Exception:  # noqa: BLE001
            pass
    # Fallback for unknown tools: compact key=value list.
    pairs = ", ".join(f"{k}={v}" for k, v in args.items())
    return f"{tool_name}({pairs})"


class ToolApprovalModal(ModalScreen[ApprovalDecision]):
    """Single-tool approval gate. Modal dismisses with an ``ApprovalDecision``.

    Keyboard:
      * ``y`` / ``enter`` — allow once.
      * ``a`` — always allow this tool (any args), TTL 7d.
      * ``n`` / ``escape`` — deny.
      * ``←`` / ``→`` — cycle focus between buttons.
    """

    DEFAULT_CSS = """
    ToolApprovalModal {
        align: center middle;
    }
    ToolApprovalModal > Vertical {
        background: $surface;
        border: thick $primary;
        padding: 1 2;
        width: 72;
        height: auto;
    }
    ToolApprovalModal #description {
        color: $text;
        margin-bottom: 1;
    }
    ToolApprovalModal Horizontal#buttons {
        align-horizontal: right;
        height: auto;
        margin-top: 1;
    }
    ToolApprovalModal Button:focus {
        border: tall $accent;
        background: $surface-lighten-1;
        color: $text;
    }
    """

    # Destructive tools: no always-allow rule so a stale rule can't silently
    # wipe data on a later turn. Restricted to allow-once / deny only.
    _DESTRUCTIVE_TOOLS: frozenset[str] = frozenset({"delete_record"})

    BINDINGS = [
        ("escape", "deny", "Deny"),
        ("y", "once", "Allow"),
        ("a", "always_tool", "Always allow"),
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

    def on_mount(self) -> None:
        self.query_one("#once", Button).focus()

    def on_key(self, event: events.Key) -> None:
        if event.key not in ("left", "right"):
            return
        event.stop()
        buttons = list(self.query("Button"))
        if not buttons:
            return
        try:
            idx = buttons.index(self.focused)
        except ValueError:
            idx = len(buttons) - 1
        if event.key == "right":
            buttons[(idx + 1) % len(buttons)].focus()
        else:
            buttons[(idx - 1) % len(buttons)].focus()

    def compose(self) -> ComposeResult:
        title = (
            f"{self._breadcrumb}: {self._tool_name}"
            if self._breadcrumb
            else f"Tool: {self._tool_name}"
        )
        with Vertical():
            yield Label(title, id="title")
            yield Static(
                _human_description(self._tool_name, self._args), id="description"
            )
            with Horizontal(id="buttons"):
                yield Button("Deny (n)", id="deny")
                # Destructive ops cannot have persistent rules — every
                # delete must be confirmed individually.
                if not self._destructive:
                    yield Button("Always allow (a)", id="always_tool")
                yield Button("Allow (y)", id="once")

    # --- actions -----------------------------------------------------

    def action_once(self) -> None:
        self.dismiss(ApprovalDecision(decision="once"))

    def action_always_tool(self) -> None:
        if self._destructive:
            return
        self.dismiss(ApprovalDecision(decision="always_tool"))

    def action_deny(self) -> None:
        self.dismiss(ApprovalDecision(decision="deny"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or "deny"
        if button_id == "once":
            self.action_once()
        elif button_id == "always_tool":
            self.action_always_tool()
        else:
            self.action_deny()
