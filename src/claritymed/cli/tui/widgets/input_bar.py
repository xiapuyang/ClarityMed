"""Bottom input bar — single ``Input`` plus a mode-aware placeholder and a
slash-command autocomplete popup.

The bar emits ``InputBar.Submitted(value)`` when the user presses Enter on
a non-empty value. The app handles routing and clearing.

Slash autocomplete:

* Typing ``/`` opens a popup above the input listing matching commands
  from ``slash_commands.KNOWN_COMMANDS``.
* ``Up`` / ``Down`` move the selection cursor.
* ``Tab`` completes to the highlighted command. ``Enter`` completes if the
  input does not yet exactly match; if it does, Enter submits normally.
* ``Esc`` closes the popup without changing the input.
"""

from __future__ import annotations

import logging
import time

from textual import events
from textual.containers import Container
from textual.message import Message
from textual.widgets import Input, Static

from claritymed.cli.tui.slash_commands import KNOWN_COMMANDS

logger = logging.getLogger(__name__)

# Commands that take an argument get a trailing space so the user can keep
# typing without backspacing. Pure-trigger commands land without the space.
_COMMANDS_WITH_ARG: frozenset[str] = frozenset({"upload", "mode", "user"})
_POPUP_MAX_ROWS: int = 8
_SELECTED_PREFIX: str = "▸ "
_UNSELECTED_PREFIX: str = "  "


class InputBar(Container):
    """Container around a single Input plus a slash-command popup."""

    DEFAULT_CSS = """
    InputBar {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
        background: $surface;
    }
    InputBar #slash-popup {
        height: auto;
        max-height: 10;
        background: $boost;
        color: $text;
        border: solid $primary;
        padding: 0 1;
        display: none;
    }
    InputBar #slash-popup.visible {
        display: block;
    }
    InputBar Input {
        background: $boost;
    }
    InputBar #streaming-indicator {
        display: none;
        color: $warning;
        padding: 0 1;
        height: 1;
    }
    InputBar.streaming #streaming-indicator {
        display: block;
    }
    InputBar.streaming Input {
        border: tall $warning;
        opacity: 0.7;
    }
    """

    class Submitted(Message):
        """Emitted when the user submits a non-empty input."""

        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def __init__(self) -> None:
        super().__init__()
        self._popup_visible: bool = False
        self._popup_matches: list[str] = []
        self._popup_selected: int = 0
        # When None, all KNOWN_COMMANDS are eligible for autocomplete. The
        # app narrows this to a subset based on the current user's role
        # (e.g. ``/user`` is admin-only).
        self._visible_commands: frozenset[str] | None = None
        # When we programmatically set ``Input.value`` (e.g. during a Tab
        # completion or a ``clear()``), Textual queues an ``Input.Changed``
        # message. If we left the popup logic to re-run on that message,
        # it would re-open the popup we just dismissed. Mark the expected
        # value here so ``on_input_changed`` can ignore exactly one event.
        self._suppress_next_value: str | None = None
        self._stream_start: float | None = None
        self._stream_timer = None
        # Drop-bug diagnostics: when Ghostty's drag-drop occasionally lands
        # as raw keystrokes instead of bracketed paste, App.on_paste does
        # NOT fire but the Input still grows. Tracking the prior length
        # lets ``on_input_changed`` log a sudden burst (≥10 chars at once
        # or 0→path-shaped) so future failures can be diagnosed by
        # cross-referencing against ``on_paste:`` lines in app.log.
        self._prev_value_len: int = 0

    def compose(self):
        yield Static("⟳  Responding…  (0s · Esc to cancel)", id="streaming-indicator")
        yield Static("", id="slash-popup")
        yield _SlashInput(placeholder="What would you like to know?", id="input")

    # ----- public API the App calls --------------------------------------

    def set_streaming(self, active: bool) -> None:
        if active:
            self._stream_start = time.monotonic()
            self.add_class("streaming")
            self._tick_streaming_label()
            self._stream_timer = self.set_interval(1.0, self._tick_streaming_label)
        else:
            if self._stream_timer is not None:
                self._stream_timer.stop()
                self._stream_timer = None
            self._stream_start = None
            self.remove_class("streaming")

    def _tick_streaming_label(self) -> None:
        if self._stream_start is None:
            return
        elapsed = int(time.monotonic() - self._stream_start)
        if elapsed < 60:
            elapsed_str = f"{elapsed}s"
        else:
            m, s = divmod(elapsed, 60)
            elapsed_str = f"{m}m {s}s"
        self.query_one("#streaming-indicator", Static).update(
            f"⟳  Responding…  ({elapsed_str} · Esc to cancel)"
        )

    def set_placeholder(self, text: str) -> None:
        self.query_one("#input", Input).placeholder = text

    def focus_input(self) -> None:
        self.query_one("#input", Input).focus()

    def clear(self) -> None:
        self._hide_popup()
        self._suppress_next_value = ""
        self.query_one("#input", Input).value = ""

    def value(self) -> str:
        return self.query_one("#input", Input).value

    def set_command_filter(self, visible: frozenset[str] | None) -> None:
        """Restrict autocomplete to ``visible`` (None = all known commands).

        Hides any currently-open popup so a stale match cannot survive a
        role change (e.g. demoted admin shouldn't see ``/user`` highlighted).
        """
        self._visible_commands = visible
        if self._popup_visible:
            self._hide_popup()

    # ----- Textual event handlers ----------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        val = event.value
        prev_len = self._prev_value_len
        new_len = len(val)
        self._prev_value_len = new_len
        # Sudden growth signals a paste-like burst. A single keystroke is
        # +1, IME composition lands as one combined update — anything ≥10
        # is either a bracketed paste (logged by App.on_paste) or the
        # drag-drop-as-raw-keystrokes failure mode we're trying to catch.
        # The 0→non-empty transition is also logged because raw-keystroke
        # drops arrive char-by-char and only the first event has prev=0.
        growth = new_len - prev_len
        if growth >= 10 or (prev_len == 0 and new_len > 0):
            logger.info(
                "input_changed: prev_len=%d new_len=%d growth=%d head=%r",
                prev_len,
                new_len,
                growth,
                val[:120],
            )
        if (
            self._suppress_next_value is not None
            and event.value == self._suppress_next_value
        ):
            self._suppress_next_value = None
            return
        self._refresh_popup(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # If the popup is open and the highlighted match doesn't already
        # equal the input, complete first instead of submitting. Otherwise
        # behave as before: emit Submitted and let the app route.
        if self._popup_visible and self._popup_matches:
            if self._try_complete_from_popup(submit_on_exact=True):
                return  # completion took the keystroke; user presses Enter again
        text = event.value.strip()
        if not text:
            return
        self.post_message(self.Submitted(text))

    # ----- popup internals -----------------------------------------------

    def _refresh_popup(self, value: str) -> None:
        if not value.startswith("/"):
            self._hide_popup()
            return
        # Only autocomplete on the command head — the chars between ``/`` and
        # the first space. Anything past the space is argument typing and
        # shouldn't re-filter the menu.
        body = value[1:]
        head = body.split(" ", 1)[0].lower()
        eligible = (
            KNOWN_COMMANDS
            if self._visible_commands is None
            else tuple(c for c in KNOWN_COMMANDS if c in self._visible_commands)
        )
        matches = [c for c in eligible if c.startswith(head)]
        if not matches:
            self._hide_popup()
            return
        # When the value already includes a space, the user is typing args;
        # leave the popup hidden so it doesn't fight with the cursor.
        if " " in body:
            self._hide_popup()
            return
        self._popup_matches = matches
        if self._popup_selected >= len(matches):
            self._popup_selected = 0
        self._render_popup()
        if not self._popup_visible:
            self._popup_visible = True
            self.query_one("#slash-popup", Static).add_class("visible")

    def _render_popup(self) -> None:
        lines: list[str] = []
        for i, cmd in enumerate(self._popup_matches[:_POPUP_MAX_ROWS]):
            prefix = (
                _SELECTED_PREFIX if i == self._popup_selected else _UNSELECTED_PREFIX
            )
            lines.append(f"{prefix}/{cmd}")
        self.query_one("#slash-popup", Static).update("\n".join(lines))

    def _hide_popup(self) -> None:
        if not self._popup_visible and not self._popup_matches:
            return
        self._popup_visible = False
        self._popup_matches = []
        self._popup_selected = 0
        popup = self.query_one("#slash-popup", Static)
        popup.remove_class("visible")
        popup.update("")

    def _popup_move(self, delta: int) -> None:
        if not self._popup_visible or not self._popup_matches:
            return
        n = len(self._popup_matches)
        self._popup_selected = (self._popup_selected + delta) % n
        self._render_popup()

    def _try_complete_from_popup(self, *, submit_on_exact: bool = False) -> bool:
        """Complete the input to the highlighted command.

        Returns ``True`` when the keystroke has been consumed (popup was
        open and either we completed or — when ``submit_on_exact`` is set
        — we let an exact-match Enter pass through to ``Submitted``).
        Returns ``False`` when the popup wasn't open.
        """
        if not self._popup_visible or not self._popup_matches:
            return False
        selected = self._popup_matches[self._popup_selected]
        target = f"/{selected}" + (" " if selected in _COMMANDS_WITH_ARG else "")
        inp = self.query_one("#input", Input)
        current = inp.value
        if submit_on_exact and current.rstrip() == f"/{selected}":
            # Exact match — close popup, let the Input.Submitted bubble up
            # so the caller's flow runs unchanged.
            self._hide_popup()
            return False
        self._suppress_next_value = target
        inp.value = target
        inp.cursor_position = len(target)
        self._hide_popup()
        return True


class _SlashInput(Input):
    """Input subclass that lets the InputBar steer Up/Down/Tab/Esc while
    the slash-command popup is visible. When the popup is hidden these
    keys keep their default Input behaviour."""

    def _on_paste(self, event: events.Paste) -> None:
        # Diagnostic-only — do NOT call super().
        #
        # Textual's _get_dispatch_methods walks the MRO and yields every
        # class in the chain that defines _on_paste in its own __dict__,
        # invoking each independently. Input._on_paste is already in the
        # dispatch list; calling super()._on_paste(event) here would run
        # it a second time and insert the pasted text twice.
        logger.info(
            "_SlashInput._on_paste: len=%d head=%r",
            len(event.text) if event.text else 0,
            event.text[:120] if event.text else "",
        )
        # Drag-drop file ingestion lives in App.on_paste, but when this
        # Input is focused the paste lands here first and Textual's
        # Input._on_paste (called next via the MRO walker) calls
        # event.stop() after inserting, so the event never bubbles up.
        # Detect drop-shaped payloads, zero out event.text so the
        # inherited handler inserts nothing, and forward the original
        # text to the App handler directly.
        from claritymed.cli.tui.app import _looks_like_drop_attempt

        text = event.text or ""
        if text and _looks_like_drop_attempt(text):
            event.text = ""
            self.app.on_paste(events.Paste(text))

    def on_key(self, event: events.Key) -> None:
        bar = self.parent
        if not isinstance(bar, InputBar):
            return
        if not bar._popup_visible:
            return
        if event.key == "up":
            bar._popup_move(-1)
            event.stop()
            event.prevent_default()
        elif event.key == "down":
            bar._popup_move(1)
            event.stop()
            event.prevent_default()
        elif event.key == "tab":
            bar._try_complete_from_popup()
            event.stop()
            event.prevent_default()
        elif event.key == "escape":
            bar._hide_popup()
            event.stop()
            event.prevent_default()
