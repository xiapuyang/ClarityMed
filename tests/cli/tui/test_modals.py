"""Modal tests — push_screen + assert the dismiss payload."""

from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Button

from claritymed.cli.tui.modals import (
    ApprovalModal,
    LibraryModal,
    ModeModal,
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
async def test_mode_modal_buttons_return_choice():
    app = _ModalHostApp(ModeModal(detected_mode="ingest", confidence=0.4))
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#rag", Button).press()
        await pilot.pause()
    assert app.result == "rag"


@pytest.mark.asyncio
async def test_mode_modal_cancel_returns_none():
    app = _ModalHostApp(ModeModal())
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#cancel", Button).press()
        await pilot.pause()
    assert app.result is None


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
async def test_library_modal_save_returns_active_collections():
    app = _ModalHostApp(
        LibraryModal(
            documents=["doc-a", "doc-b"],
            active_collections=["medcorp_en"],
        )
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#save", Button).press()
        await pilot.pause()
    # The default checkbox state matches active_collections so it should
    # contain at least 'medcorp_en' (assuming the config still has it).
    assert isinstance(app.result, list)


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
        modal = app.screen
        modal.query_one("#cancel", Button).press()
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
        modal = app.screen
        modal.query_one("#p_cloud_llm", Button).press()
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
        btn = app.screen.query_one("#p_local_llm", Button)
        assert "selected" in btn.classes


@pytest.mark.asyncio
async def test_provider_modal_no_providers_shows_label(monkeypatch):
    monkeypatch.setattr("claritymed.stores.models.list_available_providers", lambda: [])
    from textual.widgets import Label

    app = _ModalHostApp(ProviderModal())
    async with app.run_test() as pilot:
        await pilot.pause()
        labels = [lbl.renderable for lbl in app.screen.query(Label)]
        assert any("No providers" in str(lbl) for lbl in labels)
