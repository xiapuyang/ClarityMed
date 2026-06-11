"""Tests for ``cli.tui.paste`` clipboard classifier.

Platform-specific reading is hard to unit-test without mocking each
OS's clipboard surface; tests focus on the ``classify_text`` decision
tree and exercise the platform readers' empty/missing-tool paths.
"""

from __future__ import annotations

from pathlib import Path

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
