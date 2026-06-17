"""Modal tests — push_screen + assert the dismiss payload."""

from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Button, OptionList

from claritymed.cli.tui.modals import (
    ApprovalModal,
    LibraryModal,
    UploadModal,
)
from claritymed.cli.tui.modals.provider_modal import ProviderModal


class _ModalHostApp(App):
    """Minimal host that pushes one modal and remembers its dismiss payload."""

    def __init__(self, modal) -> None:
        super().__init__()
        self._modal = modal
        self.result = "<unset>"

    def on_mount(self) -> None:
        def _capture(value):
            self.result = value

        self.push_screen(self._modal, _capture)


@pytest.mark.asyncio
async def test_upload_modal_cancel_dismisses_with_none():
    app = _ModalHostApp(UploadModal(initial_path=""))
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#cancel", Button).press()
        await pilot.pause()
    assert app.result is None


@pytest.mark.asyncio
async def test_upload_modal_rejects_missing_file(tmp_path):
    app = _ModalHostApp(UploadModal(initial_path=str(tmp_path / "ghost.txt")))
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#confirm", Button).press()
        await pilot.pause()
        # Still on the modal — dismiss never fired.
        assert app.result == "<unset>"


@pytest.mark.asyncio
async def test_upload_modal_confirms_with_valid_file(tmp_path):
    sample = tmp_path / "note.txt"
    sample.write_text("hello world", encoding="utf-8")
    app = _ModalHostApp(UploadModal(initial_path=str(sample)))
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#confirm", Button).press()
        await pilot.pause()
    assert app.result is not None
    text, public = app.result
    assert text == "hello world"
    # Default radio is "record" → public=False.
    assert public is False


@pytest.mark.asyncio
async def test_approval_modal_accept():
    app = _ModalHostApp(
        ApprovalModal(provider_id="claude", model="anthropic:claude-sonnet-4-5")
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#accept", Button).press()
        await pilot.pause()
    assert app.result is True


@pytest.mark.asyncio
async def test_approval_modal_deny():
    app = _ModalHostApp(
        ApprovalModal(provider_id="claude", model="anthropic:claude-sonnet-4-5")
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#deny", Button).press()
        await pilot.pause()
    assert app.result is False


@pytest.mark.asyncio
async def test_library_modal_lists_system_collections_with_strategy_none():
    """List view (empty query) renders without needing a live RAG strategy —
    every configured system collection plus a [USER] row should appear."""
    app = _ModalHostApp(
        LibraryModal(
            user_id="test",
            language="en",
            strategy=None,
            initial_query="",
        )
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        # Sweep through the rendered ListView labels.
        from textual.widgets import ListView

        listview = modal.query_one("#results", ListView)
        rendered = " ".join(
            str(child.renderable)
            for item in listview.children
            for child in item.children
            if hasattr(child, "renderable")
        )
        assert "[USER]" in rendered
        # statpearls_en + textbooks_en are seeded in configs/retrieval.yaml.
        assert "statpearls_en" in rendered or "textbooks_en" in rendered
        modal.query_one("#close", Button).press()
        await pilot.pause()
    assert app.result is None


@pytest.mark.asyncio
async def test_library_modal_blocks_search_when_strategy_missing():
    """Submitting a query with strategy=None surfaces a 'RAG disabled' banner
    instead of crashing."""
    from textual.widgets import Input, Static

    app = _ModalHostApp(
        LibraryModal(
            user_id="test",
            language="en",
            strategy=None,
            initial_query="",
        )
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#query", Input).value = "hypertension"
        modal.query_one("#query", Input).post_message(
            Input.Submitted(modal.query_one("#query", Input), "hypertension", None)
        )
        # The Input.Submitted handler runs a worker; pause until it settles.
        for _ in range(10):
            await pilot.pause()
            if "disabled" in str(modal.query_one("#status", Static).renderable):
                break
        assert "disabled" in str(modal.query_one("#status", Static).renderable)
        modal.query_one("#close", Button).press()
        await pilot.pause()
    assert app.result is None


# ---------------------------------------------------------------------------
# ProviderModal
# ---------------------------------------------------------------------------


def _stub_providers():
    from claritymed.core.schemas.models import ProviderConfig

    return [
        ProviderConfig(id="local_llm", kind="local", model="qwen3:14b"),
        ProviderConfig(id="cloud_llm", kind="cloud", model="openai:gpt-4o"),
    ]


@pytest.mark.asyncio
async def test_provider_modal_cancel_dismisses_with_none(monkeypatch):
    monkeypatch.setattr(
        "claritymed.stores.models.list_available_providers",
        _stub_providers,
    )
    app = _ModalHostApp(ProviderModal(current_provider_id="local_llm"))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
    assert app.result is None


@pytest.mark.asyncio
async def test_provider_modal_selects_provider(monkeypatch):
    monkeypatch.setattr(
        "claritymed.stores.models.list_available_providers",
        _stub_providers,
    )
    app = _ModalHostApp(ProviderModal(current_provider_id="local_llm"))
    async with app.run_test() as pilot:
        await pilot.pause()
        # current is local_llm (index 0); move down to cloud_llm (index 1)
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()
    assert app.result == "cloud_llm"


@pytest.mark.asyncio
async def test_provider_modal_marks_current_provider_selected(monkeypatch):
    monkeypatch.setattr(
        "claritymed.stores.models.list_available_providers",
        _stub_providers,
    )
    app = _ModalHostApp(ProviderModal(current_provider_id="local_llm"))
    async with app.run_test() as pilot:
        await pilot.pause()
        picker = app.screen.query_one("#picker", OptionList)
        # current provider (local_llm) is at index 0 and should be pre-highlighted
        assert picker.highlighted == 0


@pytest.mark.asyncio
async def test_provider_modal_no_providers_shows_label(monkeypatch):
    monkeypatch.setattr("claritymed.stores.models.list_available_providers", lambda: [])
    from textual.widgets import Label

    app = _ModalHostApp(ProviderModal())
    async with app.run_test() as pilot:
        await pilot.pause()
        labels = [lbl.renderable for lbl in app.screen.query(Label)]
        assert any("No providers" in str(lbl) for lbl in labels)


# ---------------------------------------------------------------------------
# UserModal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_modal_cancel_dismisses_with_none(alice, bob):
    from claritymed.cli.tui.modals.user_modal import UserModal

    app = _ModalHostApp(UserModal(current_user_id="alice"))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
    assert app.result is None


@pytest.mark.asyncio
async def test_user_modal_selects_user(alice, bob):
    from claritymed.cli.tui.modals.user_modal import UserModal

    app = _ModalHostApp(UserModal(current_user_id="alice"))
    async with app.run_test() as pilot:
        await pilot.pause()
        # alice is highlighted (index 0 in sorted ids); down → bob.
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()
    assert app.result == "bob"


@pytest.mark.asyncio
async def test_user_modal_marks_current_user_highlighted(alice, bob):
    from claritymed.cli.tui.modals.user_modal import UserModal

    app = _ModalHostApp(UserModal(current_user_id="bob"))
    async with app.run_test() as pilot:
        await pilot.pause()
        picker = app.screen.query_one("#picker", OptionList)
        # Sorted ids: ["alice", "bob"] → bob is at index 1.
        assert picker.highlighted == 1


@pytest.mark.asyncio
async def test_user_modal_no_users_shows_label():
    from textual.widgets import Label

    from claritymed.cli.tui.modals.user_modal import UserModal

    # No fixture invoked → list_user_ids() returns [].
    app = _ModalHostApp(UserModal(current_user_id=None))
    async with app.run_test() as pilot:
        await pilot.pause()
        labels = [lbl.renderable for lbl in app.screen.query(Label)]
        assert any("No users" in str(lbl) for lbl in labels)
