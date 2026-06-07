"""One-shot toast notification overlay (used for background-job completions)."""

from __future__ import annotations

from textual.containers import Container
from textual.widgets import Static


class Toast(Container):
    """Bottom-right ephemeral notification.

    The app posts a toast with ``Toast.show(parent, text, kind)``; the toast
    auto-dismisses after ``ttl`` seconds.
    """

    DEFAULT_CSS = """
    Toast {
        dock: bottom;
        layer: notification;
        align-horizontal: right;
        width: auto;
        height: auto;
        padding: 0 1;
        margin: 0 2 4 0;
    }
    Toast > Static {
        background: $accent;
        color: $text;
        padding: 0 1;
    }
    Toast.error > Static {
        background: $error;
    }
    Toast.success > Static {
        background: $success;
    }
    """

    def __init__(self, text: str, kind: str = "info", ttl: float = 4.0) -> None:
        super().__init__()
        self._text = text
        self._ttl = ttl
        if kind in {"error", "success"}:
            self.add_class(kind)

    def compose(self):
        yield Static(self._text)

    def on_mount(self) -> None:
        self.set_timer(self._ttl, self.remove)
