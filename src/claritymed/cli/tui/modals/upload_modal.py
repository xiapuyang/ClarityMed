"""Upload modal — yes/no approve gate for an ``UploadBundle``.

The modal is purely presentational: the caller does all I/O upstream
(read file from disk in path mode, resolve placeholders in mixed-text
mode), assembles an ``UploadBundle``, then hands it here. We render
the per-part preview, run ``bundle.validate()``, and dismiss with
``True`` on approve / ``None`` on cancel.

If validation fails the Yes button is disabled and the gate reasons
are listed under the part rows so the user sees exactly which part is
the problem. Mirrors the ``ToolApprovalModal`` pattern (explicit deny,
no silent skipping) — same focus-cycling keybindings.
"""

from __future__ import annotations

import logging

from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from claritymed.core.i18n import t
from claritymed.core.upload import PartStatus, UploadBundle, UploadPart

logger = logging.getLogger(__name__)


# Per-part status → icon glyph. These are intentionally NOT in i18n
# YAML — the glyphs are universal and embedding them inline keeps the
# render fast (no t() call per row per status).
_STATUS_ICON: dict[PartStatus, str] = {
    "ok": "✓",
    "low_content": "⚠",
    "ocr_pending": "⟳",
    "ocr_failed": "✗",
}


class UploadModal(ModalScreen):
    """Yes / No approval for adding a bundle to the public library."""

    DEFAULT_CSS = """
    UploadModal {
        align: center middle;
    }
    UploadModal > Vertical {
        background: $surface;
        border: thick $primary;
        padding: 1 2;
        width: 80;
        height: auto;
    }
    UploadModal #parts {
        color: $text;
        margin-top: 1;
        margin-bottom: 1;
    }
    UploadModal #error {
        color: $error;
        height: auto;
    }
    UploadModal Horizontal#buttons {
        align-horizontal: right;
        height: auto;
        margin-top: 1;
    }
    UploadModal Button:focus {
        border: tall $accent;
        background: $surface-lighten-1;
        color: $text;
    }
    UploadModal Button.-disabled {
        opacity: 0.5;
    }
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("y", "confirm", "Yes"),
        ("n", "cancel", "No"),
    ]

    def __init__(
        self,
        bundle: UploadBundle,
        *,
        language: str = "en",
    ) -> None:
        super().__init__()
        self._bundle = bundle
        self._language = language
        self._validation = bundle.validate()

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(t("rag.upload.confirm.title", lang=self._language))
            yield Static(self._render_parts(), id="parts")
            error_text = self._render_reasons() if not self._validation.ok else ""
            yield Static(error_text, id="error")
            with Horizontal(id="buttons"):
                yield Button(
                    t("rag.upload.confirm.no_button", lang=self._language),
                    id="cancel",
                )
                # The disabled flag governs the click path; the focus
                # cycling is allowed either way so screen-reader users
                # can still see the disabled state on the focused row.
                yield Button(
                    t("rag.upload.confirm.yes_button", lang=self._language),
                    id="confirm",
                    variant="primary",
                    disabled=not self._validation.ok,
                )

    def on_mount(self) -> None:
        # Default focus mirrors the validation state: on the Yes button
        # when the user can proceed, on Cancel when they can't (so the
        # most obvious action is one Enter away).
        target_id = "confirm" if self._validation.ok else "cancel"
        self.query_one(f"#{target_id}", Button).focus()

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
            idx = 0
        delta = 1 if event.key == "right" else -1
        buttons[(idx + delta) % len(buttons)].focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_confirm(self) -> None:
        # ``y`` keybind respects the same disabled gate as the button.
        if not self._validation.ok:
            return
        self.dismiss(True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        if event.button.id == "confirm" and self._validation.ok:
            self.dismiss(True)

    # ----- render helpers -----------------------------------------------

    def _render_parts(self) -> str:
        """One ``rag.upload.part.row`` per Part, joined with newlines.

        Parts with no extracted content yet (OCR pending / failed) use
        a slightly different template so we don't print
        "(image, 0 chars)" — that misreads as "the file is empty"
        rather than "we don't know yet".
        """
        if not self._bundle.parts:
            return ""
        lines: list[str] = []
        for part in self._bundle.parts:
            lines.append(self._format_part(part))
        return "\n".join(lines)

    def _format_part(self, part: UploadPart) -> str:
        icon = _STATUS_ICON.get(part.status, "?")
        kind_label = t(f"rag.upload.kind.{part.kind}", lang=self._language)
        key = "rag.upload.part.row_empty" if part.chars == 0 else "rag.upload.part.row"
        return t(
            key,
            lang=self._language,
            icon=icon,
            source=part.source,
            kind=kind_label,
            chars=part.chars,
        )

    def _render_reasons(self) -> str:
        """Translate each validation reason via ``rag.upload.gate.*``.

        Reasons come in three shapes:

        * ``"empty"`` — bare key.
        * ``"<code>:<source>"`` — ocr_failed / ocr_pending / low_content.
        * ``"total_too_short:<actual>/<floor>"`` — special-case format.
        """
        out: list[str] = []
        for reason in self._validation.reasons:
            out.append(self._translate_reason(reason))
        return "\n".join(out)

    def _translate_reason(self, reason: str) -> str:
        if reason == "empty":
            return t("rag.upload.gate.empty", lang=self._language)
        if ":" not in reason:
            return reason
        code, arg = reason.split(":", 1)
        if code == "total_too_short" and "/" in arg:
            actual, floor = arg.split("/", 1)
            return t(
                "rag.upload.gate.total_too_short",
                lang=self._language,
                actual=actual,
                floor=floor,
            )
        return t(
            f"rag.upload.gate.{code}",
            lang=self._language,
            source=arg,
        )

    # ----- accessors used by tests --------------------------------------

    @property
    def bundle(self) -> UploadBundle:
        return self._bundle

    @property
    def validation_ok(self) -> bool:
        return self._validation.ok
