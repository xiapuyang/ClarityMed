"""Slash-command autocomplete on the input bar.

Verifies the popup shows when ``/`` is typed, filters by prefix, navigates
with Up/Down, completes on Tab, and submits-on-exact-match on Enter so
the existing routing flow still works.
"""

from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Input, Static

from claritymed.cli.tui.widgets import InputBar


class _Host(App):
    """Tiny host that mounts only the InputBar so the tests don't pay for
    the full ClarityMedApp surface (status bar, conversation, modals)."""

    def compose(self):
        yield InputBar()


def _input(app: _Host) -> Input:
    return app.query_one("#input", Input)


def _popup(app: _Host) -> Static:
    return app.query_one("#slash-popup", Static)


@pytest.mark.asyncio
async def test_typing_slash_opens_popup_with_all_commands():
    async with _Host().run_test() as pilot:
        app: _Host = pilot.app  # type: ignore[assignment]
        bar = app.query_one(InputBar)
        bar.focus_input()
        await pilot.press("/")
        await pilot.pause()
        assert bar._popup_visible
        rendered = str(_popup(app).renderable)
        # All known commands should be listed under "/".
        for cmd in (
            "clear",
            "help",
            "lang",
            "library",
            "provider",
            "quit",
            "upload",
            "user",
        ):
            assert f"/{cmd}" in rendered


@pytest.mark.asyncio
async def test_popup_filters_by_prefix():
    async with _Host().run_test() as pilot:
        app: _Host = pilot.app  # type: ignore[assignment]
        bar = app.query_one(InputBar)
        bar.focus_input()
        await pilot.press("/", "c")
        await pilot.pause()
        assert bar._popup_visible
        assert bar._popup_matches == ["clear"]
        rendered = str(_popup(app).renderable)
        assert "/clear" in rendered
        # No leakage from other commands.
        assert "/library" not in rendered
        assert "/help" not in rendered


@pytest.mark.asyncio
async def test_unknown_prefix_hides_popup():
    async with _Host().run_test() as pilot:
        app: _Host = pilot.app  # type: ignore[assignment]
        bar = app.query_one(InputBar)
        bar.focus_input()
        await pilot.press("/", "z")
        await pilot.pause()
        assert not bar._popup_visible


@pytest.mark.asyncio
async def test_down_arrow_navigates_selection_wraps():
    async with _Host().run_test() as pilot:
        app: _Host = pilot.app  # type: ignore[assignment]
        bar = app.query_one(InputBar)
        bar.focus_input()
        await pilot.press("/")
        await pilot.pause()
        initial = bar._popup_selected
        await pilot.press("down")
        await pilot.pause()
        assert bar._popup_selected == (initial + 1) % len(bar._popup_matches)
        await pilot.press("up")
        await pilot.pause()
        assert bar._popup_selected == initial


@pytest.mark.asyncio
async def test_tab_completes_to_selected_command():
    """Tab on a non-arg command like /help fills the input and hides the popup."""
    async with _Host().run_test() as pilot:
        app: _Host = pilot.app  # type: ignore[assignment]
        bar = app.query_one(InputBar)
        bar.focus_input()
        await pilot.press("/", "h")
        await pilot.pause()
        assert bar._popup_matches == ["help"]
        await pilot.press("tab")
        await pilot.pause()
        assert _input(app).value == "/help"
        assert not bar._popup_visible


@pytest.mark.asyncio
async def test_tab_completes_arg_command_with_trailing_space():
    """Arg-taking commands (/upload, /user, /provider) complete with a trailing
    space so the user can keep typing without backspacing."""
    async with _Host().run_test() as pilot:
        app: _Host = pilot.app  # type: ignore[assignment]
        bar = app.query_one(InputBar)
        bar.focus_input()
        await pilot.press("/", "u", "p")
        await pilot.pause()
        assert bar._popup_matches == ["upload"]
        await pilot.press("tab")
        await pilot.pause()
        assert _input(app).value == "/upload "
        assert not bar._popup_visible


@pytest.mark.asyncio
async def test_enter_on_exact_match_submits():
    """If the input already equals the highlighted command, Enter should
    submit through the existing flow (no double-Enter required)."""
    submitted: list[str] = []

    class _Spy(_Host):
        def on_input_bar_submitted(self, message: InputBar.Submitted) -> None:
            submitted.append(message.value)

    async with _Spy().run_test() as pilot:
        app = pilot.app
        bar = app.query_one(InputBar)
        bar.focus_input()
        for ch in "/help":
            await pilot.press(ch)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert submitted == ["/help"]


@pytest.mark.asyncio
async def test_escape_closes_popup_without_changing_input():
    async with _Host().run_test() as pilot:
        app: _Host = pilot.app  # type: ignore[assignment]
        bar = app.query_one(InputBar)
        bar.focus_input()
        await pilot.press("/", "h")
        await pilot.pause()
        before = _input(app).value
        await pilot.press("escape")
        await pilot.pause()
        assert not bar._popup_visible
        assert _input(app).value == before


@pytest.mark.asyncio
async def test_set_command_filter_hides_user_for_non_admin():
    """When the app narrows the visible set, /user must drop out of
    autocomplete entirely — typing /u then only matches /upload."""
    async with _Host().run_test() as pilot:
        app: _Host = pilot.app  # type: ignore[assignment]
        bar = app.query_one(InputBar)
        # Mimic what ClarityMedApp._apply_command_filter does for a non-admin.
        from claritymed.cli.tui.slash_commands import KNOWN_COMMANDS

        bar.set_command_filter(frozenset(c for c in KNOWN_COMMANDS if c != "user"))
        bar.focus_input()
        await pilot.press("/")
        await pilot.pause()
        assert "user" not in bar._popup_matches
        await pilot.press("u")
        await pilot.pause()
        # /u prefix → only /upload remains for a non-admin.
        assert bar._popup_matches == ["upload"]


@pytest.mark.asyncio
async def test_popup_hides_once_user_starts_typing_args():
    """After completing a /<cmd> with a space, the popup should step out of
    the way — further chars are arguments, not command prefixes."""
    async with _Host().run_test() as pilot:
        app: _Host = pilot.app  # type: ignore[assignment]
        bar = app.query_one(InputBar)
        bar.focus_input()
        await pilot.press("/", "u", "p")
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()
        assert _input(app).value == "/upload "
        await pilot.press("f", "o", "o")
        await pilot.pause()
        assert not bar._popup_visible
        assert _input(app).value == "/upload foo"


@pytest.mark.asyncio
async def test_typing_non_slash_input_does_not_show_popup():
    async with _Host().run_test() as pilot:
        app: _Host = pilot.app  # type: ignore[assignment]
        bar = app.query_one(InputBar)
        bar.focus_input()
        await pilot.press("h", "e", "l", "l", "o")
        await pilot.pause()
        assert not bar._popup_visible


@pytest.mark.asyncio
async def test_paste_inserts_text_exactly_once():
    """Regression for the Cmd+V double-paste bug.

    Textual's ``_get_dispatch_methods`` yields every ``_on_paste`` it finds
    in the MRO. Our ``_SlashInput`` subclass adds its own ``_on_paste`` for
    diagnostic logging; if that override calls ``super()._on_paste(event)``,
    ``Input._on_paste`` runs twice (once via super, once via the MRO walker)
    and the pasted text shows up duplicated in the Input value.
    """
    from textual import events

    async with _Host().run_test() as pilot:
        app: _Host = pilot.app  # type: ignore[assignment]
        bar = app.query_one(InputBar)
        bar.focus_input()
        inp = _input(app)
        inp.post_message(events.Paste("hello"))
        await pilot.pause()
        assert inp.value == "hello"


@pytest.mark.asyncio
async def test_drag_drop_path_forwards_to_app_when_input_focused():
    """Regression for the "raw path landed in input" bug.

    Bracketed-paste of a file path delivered to a focused Input was being
    inserted verbatim because Textual's ``Input._on_paste`` calls
    ``event.stop()`` after inserting, which prevented the paste from
    bubbling to ``App.on_paste`` — the only place that ingests dropped
    files. ``_SlashInput._on_paste`` now detects drop-shaped payloads,
    clears ``event.text`` so the inherited handler inserts nothing, and
    forwards the original text to the App handler directly.
    """
    from textual import events

    forwarded: list[str] = []

    class _DropHost(_Host):
        def on_paste(self, event: events.Paste) -> None:
            forwarded.append(event.text)

    async with _DropHost().run_test() as pilot:
        app: _DropHost = pilot.app  # type: ignore[assignment]
        bar = app.query_one(InputBar)
        bar.focus_input()
        inp = _input(app)
        path_text = "/Users/sharp/Downloads/malignant\\ \\(3\\).png"
        inp.post_message(events.Paste(path_text))
        await pilot.pause()
        assert inp.value == ""
        assert forwarded == [path_text]
