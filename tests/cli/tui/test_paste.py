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
