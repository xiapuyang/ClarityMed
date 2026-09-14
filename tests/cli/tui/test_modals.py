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
from claritymed.core.upload import UploadBundle, UploadPart
from claritymed.core.upload.bundle import hash_inline_text


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


def _ok_bundle(*, source: str = "report.pdf", chars: int = 500) -> UploadBundle:
    """Bundle that clears all gates — single OK file part."""
    content = "x" * chars
    return UploadBundle(
        parts=(
            UploadPart(
                kind="file",
                source=source,
                content=content,
                source_hash=hash_inline_text(content),
                status="ok",
                chars=chars,
            ),
        )
    )


def _failing_bundle() -> UploadBundle:
    """Bundle that fails validation (OCR-failed part)."""
    return UploadBundle(
        parts=(
            UploadPart(
                kind="file",
                source="broken.pdf",
                content="",
                source_hash="deadbeef" * 8,
                status="ocr_failed",
                chars=0,
            ),
        )
    )


@pytest.mark.asyncio
async def test_upload_modal_cancel_dismisses_with_none():
    app = _ModalHostApp(UploadModal(_ok_bundle()))
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#cancel", Button).press()
        await pilot.pause()
    assert app.result is None


@pytest.mark.asyncio
async def test_upload_modal_confirms_with_valid_bundle():
    app = _ModalHostApp(UploadModal(_ok_bundle(source="note.txt")))
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#confirm", Button).press()
        await pilot.pause()
    # Approved bundles dismiss with True; the caller already holds the bundle.
    assert app.result is True


@pytest.mark.asyncio
async def test_upload_modal_disables_yes_for_invalid_bundle():
    """OCR-failed parts must block the Yes button so the user can't
    accidentally dispatch a doomed upload."""
    app = _ModalHostApp(UploadModal(_failing_bundle()))
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        confirm = modal.query_one("#confirm", Button)
        assert confirm.disabled is True
        # Pressing the disabled button should not dismiss.
        confirm.press()
        await pilot.pause()
        assert app.result == "<unset>"
        # `y` keybind respects the same gate as the button.
        await pilot.press("y")
        await pilot.pause()
        assert app.result == "<unset>"
        # Default focus on cancel when the bundle is invalid.
        assert modal.focused is modal.query_one("#cancel", Button)


@pytest.mark.asyncio
async def test_upload_modal_renders_parts_preview():
    """Every part is listed in the preview with status + source name."""
    bundle = UploadBundle(
        parts=(
            UploadPart(
                kind="image",
                source="scan.png",
                content="ok content " * 20,
                source_hash="a" * 64,
                status="ok",
                chars=200,
            ),
            UploadPart(
                kind="file",
                source="broken.pdf",
                content="",
                source_hash="b" * 64,
                status="ocr_failed",
                chars=0,
            ),
        )
    )
    app = _ModalHostApp(UploadModal(bundle, language="en"))
    async with app.run_test() as pilot:
        await pilot.pause()
        from textual.widgets import Static

        modal = app.screen
        rendered = str(modal.query_one("#parts", Static).renderable)
        assert "scan.png" in rendered
        assert "broken.pdf" in rendered
        # Status icon for the failed row should be the failure glyph.
        assert "✗" in rendered
        modal.query_one("#cancel", Button).press()
        await pilot.pause()


@pytest.mark.asyncio
async def test_upload_modal_arrow_keys_cycle_buttons():
    """←/→ should move focus between Yes/No so the active choice is
    visible before the user hits Enter — matches ToolApprovalModal."""
    app = _ModalHostApp(UploadModal(_ok_bundle()))
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        # Default focus = confirm (Yes is the primary action when valid).
        assert modal.focused is modal.query_one("#confirm", Button)
        await pilot.press("left")
        assert modal.focused is modal.query_one("#cancel", Button)
        await pilot.press("right")
        assert modal.focused is modal.query_one("#confirm", Button)
        # Cancel before the host app tears down.
        await pilot.press("escape")
        await pilot.pause()


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
