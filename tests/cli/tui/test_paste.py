"""Tests for ``cli.tui.paste`` clipboard classifier.

Platform-specific reading is hard to unit-test without mocking each
OS's clipboard surface; tests focus on the ``classify_text`` decision
tree and exercise the platform readers' empty/missing-tool paths.
"""

from __future__ import annotations

import builtins
import logging
import subprocess
from pathlib import Path

import claritymed.cli.tui.paste as paste_mod
from claritymed.cli.tui.paste import (
    Empty,
    FilePath,
    ImageBytes,
    LargeText,
    SmallText,
    classify_text,
    read_clipboard,
)


def test_classify_empty_text():
    assert isinstance(classify_text(""), Empty)


def test_classify_small_text():
    out = classify_text("hello world")
    assert isinstance(out, SmallText)
    assert out.text == "hello world"


def test_classify_large_text():
    blob = "x" * 1000
    out = classify_text(blob)
    assert isinstance(out, LargeText)
    assert len(out.text) == 1000


def test_classify_file_path_when_exists(tmp_path: Path):
    f = tmp_path / "report.pdf"
    f.write_bytes(b"%PDF")
    out = classify_text(str(f))
    assert isinstance(out, FilePath)
    assert out.path == f


def test_classify_file_path_missing_falls_back_to_text():
    """A path-shaped string that doesn't exist isn't a FilePath."""
    out = classify_text("/no/such/file.pdf")
    assert isinstance(out, SmallText)


def test_dataclasses_are_frozen():
    assert ImageBytes.__dataclass_fields__["bytes"].name == "bytes"
    # Pass: ensures the import worked and the type is usable.
    img = ImageBytes(bytes=b"x", ext="png")
    assert img.ext == "png"


def test_read_clipboard_returns_a_clipboard_content():
    """End-to-end smoke — whatever the system returns is a valid type."""
    out = read_clipboard()
    assert isinstance(out, (ImageBytes, FilePath, LargeText, SmallText, Empty))


def test_macos_warns_once_when_appkit_missing(monkeypatch, caplog):
    """Without pyobjc the image leg is silently broken; a one-time WARNING
    in app.log is the only signal a user has that screenshot paste will
    never work. Regression guard against the dep dropping off pyproject.
    """
    # Force the AppKit import inside _read_macos to fail.
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "AppKit":
            raise ImportError("forced for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    # Reset the module-level guard so the warning fires this run.
    monkeypatch.setattr(paste_mod, "_warned_no_appkit", False)
    # Make osascript fallback deterministic — clipboard is empty.
    monkeypatch.setattr(
        paste_mod.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="", stderr=""),
    )

    with caplog.at_level(logging.WARNING, logger="claritymed.cli.tui.paste"):
        out = paste_mod._read_macos()

    assert isinstance(out, Empty)
    assert any(
        "pyobjc-framework-Cocoa not importable" in r.message for r in caplog.records
    ), [r.message for r in caplog.records]


def test_macos_osascript_stderr_is_captured(monkeypatch):
    """osascript's stderr (where CoreGraphics writes the jp2 noise) must
    not bleed into the parent terminal. Verify subprocess.run is invoked
    with capture_output=True so stderr stays inside the subprocess.
    """
    # Force AppKit unavailable so we hit the osascript branch.
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "AppKit":
            raise ImportError("forced for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    monkeypatch.setattr(paste_mod, "_warned_no_appkit", True)  # silence the warning

    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout="some text", stderr="")

    monkeypatch.setattr(paste_mod.subprocess, "run", _fake_run)

    paste_mod._read_macos()

    assert captured["cmd"][0] == "osascript"
    assert captured["kwargs"].get("capture_output") is True, captured["kwargs"]


# --- read_clipboard dispatch -------------------------------------------


def test_read_clipboard_dispatches_to_linux(monkeypatch):
    """On linux, read_clipboard must route to ``_read_linux`` (not the macos branch)."""
    monkeypatch.setattr(paste_mod.sys, "platform", "linux")
    called: dict = {}

    def _fake_linux():
        called["yes"] = True
        return SmallText("from-linux")

    monkeypatch.setattr(paste_mod, "_read_linux", _fake_linux)
    out = paste_mod.read_clipboard()
    assert called == {"yes": True}
    assert isinstance(out, SmallText)
    assert out.text == "from-linux"


def test_read_clipboard_dispatches_to_windows(monkeypatch):
    monkeypatch.setattr(paste_mod.sys, "platform", "win32")
    called: dict = {}

    def _fake_windows():
        called["yes"] = True
        return Empty()

    monkeypatch.setattr(paste_mod, "_read_windows", _fake_windows)
    out = paste_mod.read_clipboard()
    assert called == {"yes": True}
    assert isinstance(out, Empty)


def test_read_clipboard_swallows_inner_exception(monkeypatch, caplog):
    """Any exception from a platform reader must degrade to Empty + a WARNING.

    The TUI relies on this — a missing clipboard helper must never crash
    the paste keybinding.
    """
    monkeypatch.setattr(paste_mod.sys, "platform", "darwin")

    def _boom():
        raise RuntimeError("clipboard helper exploded")

    monkeypatch.setattr(paste_mod, "_read_macos", _boom)

    with caplog.at_level(logging.WARNING, logger="claritymed.cli.tui.paste"):
        out = paste_mod.read_clipboard()
    assert isinstance(out, Empty)
    assert any("clipboard read failed" in r.message for r in caplog.records)


def test_read_clipboard_returns_empty_on_unknown_platform(monkeypatch):
    """If we ever ship on, say, freebsd, the function falls through to Empty."""
    monkeypatch.setattr(paste_mod.sys, "platform", "freebsd13")
    out = paste_mod.read_clipboard()
    assert isinstance(out, Empty)


# --- _read_macos branch coverage ---------------------------------------


class _FakePasteboard:
    """Stand-in for ``NSPasteboard.generalPasteboard()``.

    The two type-getter methods are the only API the macOS reader uses.
    """

    def __init__(self, *, png=None, text=None):
        self._png = png
        self._text = text

    def dataForType_(self, kind):  # noqa: N802 — match ObjC selector name
        assert kind == "public.png"
        return self._png

    def stringForType_(self, kind):  # noqa: N802
        assert kind == "public.utf8-plain-text"
        return self._text


def _install_fake_appkit(monkeypatch, pasteboard):
    """Inject a fake ``AppKit`` module so ``from AppKit import NSPasteboard`` succeeds."""
    import sys
    import types

    fake_mod = types.ModuleType("AppKit")

    class _NSPasteboard:
        @staticmethod
        def generalPasteboard():  # noqa: N802
            return pasteboard

    fake_mod.NSPasteboard = _NSPasteboard
    monkeypatch.setitem(sys.modules, "AppKit", fake_mod)


def test_read_macos_returns_imagebytes_when_pasteboard_has_png(monkeypatch):
    _install_fake_appkit(monkeypatch, _FakePasteboard(png=b"PNGbytes"))
    out = paste_mod._read_macos()
    assert isinstance(out, ImageBytes)
    assert out.bytes == b"PNGbytes"
    assert out.ext == "png"


def test_read_macos_returns_classified_text_when_pasteboard_has_text(monkeypatch):
    _install_fake_appkit(monkeypatch, _FakePasteboard(text="hello"))
    out = paste_mod._read_macos()
    assert isinstance(out, SmallText)
    assert out.text == "hello"


def test_read_macos_returns_empty_when_pasteboard_has_neither(monkeypatch):
    _install_fake_appkit(monkeypatch, _FakePasteboard())  # png=None, text=None
    out = paste_mod._read_macos()
    assert isinstance(out, Empty)


def test_read_macos_osascript_called_process_error_returns_empty(monkeypatch):
    """If osascript itself fails (e.g. ScriptingBridge missing), we degrade silently."""
    # Force AppKit unavailable so we fall into osascript.
    real_import = builtins.__import__

    def _no_appkit(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "AppKit":
            raise ImportError("forced for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _no_appkit)
    monkeypatch.setattr(paste_mod, "_warned_no_appkit", True)

    def _fail(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr="boom")

    monkeypatch.setattr(paste_mod.subprocess, "run", _fail)

    out = paste_mod._read_macos()
    assert isinstance(out, Empty)


# --- _read_linux --------------------------------------------------------


def test_read_linux_returns_imagebytes_from_xclip(monkeypatch):
    """xclip image cmd succeeds → ImageBytes(png)."""

    def _fake_run(cmd, **kwargs):
        assert cmd[0] == "xclip"
        return subprocess.CompletedProcess(cmd, 0, stdout=b"PNGdata", stderr=b"")

    monkeypatch.setattr(paste_mod.subprocess, "run", _fake_run)
    out = paste_mod._read_linux()
    assert isinstance(out, ImageBytes)
    assert out.bytes == b"PNGdata"
    assert out.ext == "png"


def test_read_linux_falls_back_to_text_when_image_paths_fail(monkeypatch):
    """xclip + wl-paste image legs both fail → text leg runs and returns SmallText."""
    calls: list[str] = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd[0])
        if cmd[0] == "xclip" and "image/png" in cmd:
            raise FileNotFoundError("xclip not installed")
        if cmd[0] == "wl-paste" and "--type" in cmd:
            raise FileNotFoundError("wl-paste not installed")
        if cmd[0] == "xclip":
            return subprocess.CompletedProcess(cmd, 0, stdout="hello", stderr="")
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(paste_mod.subprocess, "run", _fake_run)
    out = paste_mod._read_linux()
    assert isinstance(out, SmallText)
    assert out.text == "hello"
    # Image leg attempted before text leg — order matters because the
    # image branch returns early on success.
    assert calls[:2] == ["xclip", "wl-paste"]


def test_read_linux_returns_empty_when_all_tools_missing(monkeypatch):
    def _missing(cmd, **kwargs):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(paste_mod.subprocess, "run", _missing)
    assert isinstance(paste_mod._read_linux(), Empty)


def test_read_linux_swallows_called_process_error_on_text_leg(monkeypatch):
    """A tool that exits non-zero on the text leg → continue to the next tool."""
    seq = iter(
        [
            FileNotFoundError("xclip image"),
            FileNotFoundError("wl-paste image"),
            subprocess.CalledProcessError(1, ["xclip"], stderr="exit 1"),
            FileNotFoundError("wl-paste text"),
        ]
    )

    def _fake_run(cmd, **kwargs):
        exc = next(seq)
        raise exc

    monkeypatch.setattr(paste_mod.subprocess, "run", _fake_run)
    assert isinstance(paste_mod._read_linux(), Empty)


# --- _read_windows ------------------------------------------------------


def test_read_windows_returns_empty_when_win32clipboard_missing(monkeypatch):
    """No pywin32 → Empty; the rest of the function is short-circuited."""
    real_import = builtins.__import__

    def _no_win32(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "win32clipboard":
            raise ImportError("forced for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _no_win32)
    assert isinstance(paste_mod._read_windows(), Empty)


def _install_fake_win32(monkeypatch, *, has_dib=False, has_text=False, text=""):
    """Inject a fake ``win32clipboard`` module with selectable formats."""
    import sys
    import types

    mod = types.ModuleType("win32clipboard")
    calls: dict = {"opened": False, "closed": False}

    def _open():
        calls["opened"] = True

    def _close():
        calls["closed"] = True

    def _available(fmt):
        if fmt == 8:
            return has_dib
        if fmt == 13:
            return has_text
        return False

    def _get(fmt):
        if fmt == 8:
            return b"DIBdata"
        if fmt == 13:
            return text
        raise KeyError(fmt)

    mod.OpenClipboard = _open
    mod.CloseClipboard = _close
    mod.IsClipboardFormatAvailable = _available
    mod.GetClipboardData = _get
    monkeypatch.setitem(sys.modules, "win32clipboard", mod)
    return calls


def test_read_windows_text_path_returns_classified_text(monkeypatch):
    """Text format available + no image → returns the classified text."""
    calls = _install_fake_win32(monkeypatch, has_text=True, text="hello win")
    out = paste_mod._read_windows()
    assert isinstance(out, SmallText)
    assert out.text == "hello win"
    # The clipboard handle is always closed via the ``finally`` block.
    assert calls["opened"] and calls["closed"]


def test_read_windows_returns_empty_when_neither_format_available(monkeypatch):
    calls = _install_fake_win32(monkeypatch)  # no DIB, no text
    assert isinstance(paste_mod._read_windows(), Empty)
    assert calls["opened"] and calls["closed"]


def test_read_windows_image_path_skipped_without_pil(monkeypatch):
    """DIB available but PIL not importable → fall through to next format check."""
    _install_fake_win32(monkeypatch, has_dib=True, has_text=True, text="fallback")

    real_import = builtins.__import__

    def _no_pil(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "PIL":
            raise ImportError("forced for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _no_pil)

    out = paste_mod._read_windows()
    # PIL missing → image branch is silently skipped; text branch wins.
    assert isinstance(out, SmallText)
    assert out.text == "fallback"
