"""Bottom input bar — multi-line ``TextArea`` plus a mode-aware placeholder
and a slash-command autocomplete popup.

Migrated from Textual's single-line ``Input`` because medical pastes often
span multiple lines and ``Input`` truncated everything past the first
newline. The TextArea gives real multi-line editing: Enter submits,
Shift+Enter (or Ctrl+J / Alt+Enter as terminal-compat fallbacks) inserts
a newline, and the widget grows vertically as the user adds lines —
capped by ``max-height`` so a runaway paste never eats the conversation
pane.

Long pastes (thresholds: ``paste.placeholder_min_lines`` /
``paste.placeholder_min_chars``) collapse to a ``[Pasted text #N +M
lines]`` placeholder; the body is stashed on the InputBar and spliced
back into the submitted value at ``on_input_bar_submitted`` time so the
LLM sees the real text, not the visual placeholder.

The bar emits ``InputBar.Submitted(value)`` when the user presses Enter
on a non-empty value. The app handles routing and clearing.

Slash autocomplete: typing ``/`` opens a popup above the input; Up/Down
navigate, Tab completes, Enter submits if the current first-line value
already matches the highlighted command, Esc closes the popup.
"""

from __future__ import annotations

import logging
import re
import time

from textual import events
from textual.containers import Container
from textual.message import Message
from textual.widgets import Static, TextArea

from claritymed.cli.tui.slash_commands import KNOWN_COMMANDS

logger = logging.getLogger(__name__)

# Commands that take an argument get a trailing space so the user can keep
# typing without backspacing. Pure-trigger commands land without the space.
_COMMANDS_WITH_ARG: frozenset[str] = frozenset({"upload", "mode", "user"})
_POPUP_MAX_ROWS: int = 8
_SELECTED_PREFIX: str = "▸ "
_UNSELECTED_PREFIX: str = "  "

# Terminal keys that map to "insert newline" in the TextArea. Modern
# terminals (Ghostty, iTerm2, kitty, Alacritty) send ``shift+enter`` via
# the kitty keyboard protocol; older terminals fall back to ``ctrl+j``
# (literal LF) or ``alt+enter``. Bare ``enter`` always submits.
_NEWLINE_KEYS: frozenset[str] = frozenset({"shift+enter", "ctrl+j", "alt+enter"})

# Matches both ``[Pasted text #N]`` and ``[Pasted text #N +M lines]`` so
# the submit-time expansion handles single-line stashes too (paste of a
# single very long line still folds to a placeholder).
_PASTE_PLACEHOLDER_RE = re.compile(r"\[Pasted text #(\d+)(?: \+\d+ lines)?\]")


class InputBar(Container):
    """Container around a single TextArea plus a slash-command popup."""

    class HelpRequested(Message):
        """Emitted when the user presses ``?`` on an empty input bar."""

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
    InputBar TextArea {
        background: $boost;
        height: auto;
        min-height: 3;
        max-height: 12;
        border: tall $border-blurred;
    }
    InputBar TextArea:focus {
        border: tall $border;
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
    InputBar.streaming TextArea {
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
        # When we programmatically set ``TextArea.text`` (e.g. during Tab
        # completion or ``clear()``), Textual queues a ``TextArea.Changed``
        # message. If we left the popup logic to re-run on that message,
        # it would re-open the popup we just dismissed. Mark the expected
        # value here so ``on_text_area_changed`` ignores exactly one event.
        self._suppress_next_value: str | None = None
        self._stream_start: float | None = None
        self._stream_timer = None
        # Drop-bug diagnostics: when Ghostty's drag-drop occasionally lands
        # as raw keystrokes instead of bracketed paste, App.on_paste does
        # NOT fire but the input still grows. Tracking the prior length
        # lets ``on_text_area_changed`` log a sudden burst (≥10 chars at
        # once or 0→non-empty) so future failures can be diagnosed by
        # cross-referencing against ``on_paste:`` lines in app.log.
        self._prev_value_len: int = 0
        # Paste-stash: id → original body. Long pastes are folded into a
        # ``[Pasted text #N +M lines]`` visual placeholder; at submit
        # time, ``expand_pastes`` splices the body back so the LLM sees
        # what the user actually pasted.
        self._pastes: dict[int, str] = {}
        self._next_paste_id: int = 0

    def compose(self):
        yield Static("⟳  Responding…  (0s · Esc to cancel)", id="streaming-indicator")
        yield Static("", id="slash-popup")
        yield _SlashTextArea(id="input")

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
        # TextArea has no built-in placeholder. Stash the hint as the
        # widget's border subtitle so the mode prompt still shows up
        # somewhere visible without adding a layered Static overlay.
        # NoMatches mid-teardown is the only expected failure — anything
        # else is a real bug worth logging so we can fix it.
        from textual.css.query import NoMatches

        try:
            ta = self.query_one("#input", _SlashTextArea)
        except NoMatches:
            return
        except Exception:  # noqa: BLE001
            logger.warning("set_placeholder query_one failed", exc_info=True)
            return
        ta.border_subtitle = text or ""

    def focus_input(self) -> None:
        self.query_one("#input", _SlashTextArea).focus()

    def clear(self) -> None:
        self._hide_popup()
        self._pastes.clear()
        self._next_paste_id = 0
        self._suppress_next_value = ""
        self.query_one("#input", _SlashTextArea).text = ""

    def value(self) -> str:
        return self.query_one("#input", _SlashTextArea).text

    def expand_pastes(self, value: str) -> str:
        """Splice paste-stash bodies back into ``[Pasted text #N…]`` refs.

        Called by the submit path so the LLM receives the original text,
        not the visual placeholder. Unknown ids (e.g. a placeholder the
        user typed by hand) are left intact.
        """
        if not self._pastes:
            return value

        def _sub(m: re.Match[str]) -> str:
            pid = int(m.group(1))
            body = self._pastes.get(pid)
            return body if body is not None else m.group(0)

        return _PASTE_PLACEHOLDER_RE.sub(_sub, value)

    def stash_paste(self, body: str) -> str:
        """Stash a long paste body and return the visual placeholder.

        The placeholder uses 1-based ids per session and includes a line
        count when the body is multi-line — same shape claude-code's
        ``[Pasted text #N +M lines]`` uses, so the on-screen hint
        matches what TUI users see elsewhere.
        """
        self._next_paste_id += 1
        pid = self._next_paste_id
        self._pastes[pid] = body
        line_count = body.count("\n") + 1
        if line_count > 1:
            return f"[Pasted text #{pid} +{line_count} lines]"
        return f"[Pasted text #{pid}]"

    def set_command_filter(self, visible: frozenset[str] | None) -> None:
        """Restrict autocomplete to ``visible`` (None = all known commands).

        Hides any currently-open popup so a stale match cannot survive a
        role change (e.g. demoted admin shouldn't see ``/user`` highlighted).
        """
        self._visible_commands = visible
        if self._popup_visible:
            self._hide_popup()

    # ----- Textual event handlers ----------------------------------------

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        val = event.text_area.text
        prev_len = self._prev_value_len
        new_len = len(val)
        self._prev_value_len = new_len
        growth = new_len - prev_len
        # Sudden growth signals a paste-like burst. A single keystroke is
        # +1, IME composition lands as one combined update — anything ≥10
        # is either a bracketed paste (logged by App.on_paste) or the
        # drag-drop-as-raw-keystrokes failure mode we're trying to catch.
        # The 0→non-empty transition is also logged because raw-keystroke
        # drops arrive char-by-char and only the first event has prev=0.
        if growth >= 10 or (prev_len == 0 and new_len > 0):
            logger.info(
                "input_changed: prev_len=%d new_len=%d growth=%d head=%r",
                prev_len,
                new_len,
                growth,
                val[:120],
            )
        if self._suppress_next_value is not None and val == self._suppress_next_value:
            self._suppress_next_value = None
            return
        # Slash popup decisions only care about the first line — the
        # command head can't span newlines.
        first_line = val.split("\n", 1)[0]
        self._refresh_popup(first_line)

    # ----- submit (called by _SlashTextArea on Enter) --------------------

    def submit_current(self) -> None:
        """Submit the current TextArea contents.

        Mirrors the prior ``on_input_submitted`` flow: if the slash popup
        is open with a highlighted match that doesn't already equal the
        input, Enter completes the command first (the user presses Enter
        a second time to submit). Otherwise the value — with paste
        placeholders expanded — is posted as ``Submitted``.
        """
        if self._popup_visible and self._popup_matches:
            if self._try_complete_from_popup(submit_on_exact=True):
                return  # completion took the keystroke
        raw = self.query_one("#input", _SlashTextArea).text.strip()
        if not raw:
            return
        self.post_message(self.Submitted(self.expand_pastes(raw)))

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
        — we let an exact-match Enter pass through to the caller's submit).
        Returns ``False`` when the popup wasn't open or the input already
        equals the selected command (so submit can proceed).
        """
        if not self._popup_visible or not self._popup_matches:
            return False
        selected = self._popup_matches[self._popup_selected]
        target = f"/{selected}" + (" " if selected in _COMMANDS_WITH_ARG else "")
        ta = self.query_one("#input", _SlashTextArea)
        current = ta.text
        if submit_on_exact and current.rstrip() == f"/{selected}":
            # Exact match — close popup and let the caller submit.
            self._hide_popup()
            return False
        self._suppress_next_value = target
        ta.text = target
        ta.move_cursor((0, len(target)))
        self._hide_popup()
        return True


class _SlashTextArea(TextArea):
    """TextArea subclass that:

    * Submits on bare Enter (delegates to ``InputBar.submit_current``).
    * Inserts a newline on Shift+Enter / Ctrl+J / Alt+Enter.
    * Opens the help modal on ``?`` when the input is empty (otherwise
      ``?`` falls through and inserts).
    * Lets the InputBar steer Up/Down/Tab/Esc while the slash-command
      popup is visible; default cursor / focus behaviour otherwise.
    * Forwards drag-drop (path-shaped) pastes to ``App.on_paste``.
    * Folds long pastes into a stashed ``[Pasted text #N]`` placeholder.
    """

    def __init__(self, **kwargs) -> None:
        # tab_behavior="focus" preserves the previous Input semantics —
        # Tab moves focus rather than indenting the document. soft_wrap
        # lets long single-line pastes visually wrap within max-height
        # before they trip the placeholder threshold.
        super().__init__(tab_behavior="focus", soft_wrap=True, **kwargs)

    async def _on_paste(self, event: events.Paste) -> None:
        # Diagnostic + paste-routing handler. Runs before TextArea's own
        # ``_on_paste`` via the MRO walker; setting ``event.text = ""``
        # neutralises the inherited insert when we've already handled the
        # paste ourselves (drop forward, oversize reject, placeholder).
        #
        # ``event.stop()`` at the end matches what ``Input._on_paste``
        # always did — without it, the Paste bubbles to the InputBar /
        # App and (in test harnesses where Paste is posted directly to
        # this widget) re-enters dispatch on the same widget, causing
        # a duplicate insert. TextArea's stock handler omits the stop()
        # so we add it back here.
        text = event.text or ""
        logger.info(
            "_SlashTextArea._on_paste: len=%d head=%r",
            len(text),
            text[:120],
        )
        if not text:
            event.stop()
            return
        from claritymed.cli.tui.app import _looks_like_drop_attempt

        if _looks_like_drop_attempt(text):
            # File/URI drop — let App.on_paste run the ingest pipeline
            # and zero out the text so the inherited handler is a no-op.
            event.text = ""
            self.app.on_paste(events.Paste(text))
            event.stop()
            return

        from claritymed import config as _cfg

        limit = _cfg.paste_max_text_chars()
        if len(text) > limit:
            event.text = ""
            self.app.notify(
                f"Paste too long ({len(text)} chars; limit {limit}). "
                "Paste a shorter snippet or split it into messages.",
                severity="error",
                timeout=8.0,
            )
            event.stop()
            return

        line_count = text.count("\n") + 1
        if (
            line_count >= _cfg.paste_placeholder_min_lines()
            or len(text) >= _cfg.paste_placeholder_min_chars()
        ):
            bar = self._input_bar()
            if bar is not None:
                placeholder = bar.stash_paste(text)
                event.text = ""
                self.insert(placeholder)
                event.stop()
                return
        # Short, inlineable paste — let TextArea's default _on_paste run
        # via the MRO walker and insert the raw text (newlines preserved).
        event.stop()

    def on_key(self, event: events.Key) -> None:
        bar = self._input_bar()
        if bar is None:
            return
        # ``?`` on an empty input opens the help modal instead of inserting
        # a literal question mark. Once the user is mid-input, ``?`` falls
        # through and inserts as expected (e.g. typing a question).
        if event.key == "question_mark" and not self.text:
            bar.post_message(InputBar.HelpRequested())
            event.stop()
            event.prevent_default()
            return
        # Multi-line typing keys. ``prevent_default()`` cuts the MRO walk
        # at the next class boundary, so TextArea._on_key never reaches
        # the ``insert_values["enter"] = "\n"`` insert path — Enter and
        # Shift+Enter end up doing exactly what the user expects.
        if event.key in _NEWLINE_KEYS:
            self.insert("\n")
            event.stop()
            event.prevent_default()
            return
        if event.key == "enter":
            bar.submit_current()
            event.stop()
            event.prevent_default()
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

    def _input_bar(self) -> "InputBar | None":
        """Walk parents until we find the InputBar, or None if detached.

        ``self.parent`` is the InputBar's compose container — InputBar is
        usually two hops up. Returning None lets callers gracefully no-op
        during widget teardown when the tree is being dismantled.
        """
        node = self.parent
        while node is not None and not isinstance(node, InputBar):
            node = node.parent
        return node
