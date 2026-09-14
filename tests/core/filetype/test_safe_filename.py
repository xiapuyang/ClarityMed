"""Tests for the filename sanitizer feeding ``ocr.json``."""

from __future__ import annotations

import unicodedata

from claritymed.core.filetype.safe_filename import MAX_FILENAME_LEN, safe_filename


def test_none_and_empty_return_none():
    assert safe_filename(None) is None
    assert safe_filename("") is None
    assert safe_filename("   ") is None
    assert safe_filename("\t\n") is None


def test_chinese_filename_preserved_verbatim():
    """Chinese names round-trip without any escape / fold — that's the
    whole point of ensure_ascii=False on the JSON write."""
    name = "化验单_2026年6月.pdf"
    assert safe_filename(name) == name


def test_emoji_filename_preserved():
    """User-facing names sometimes carry emoji (Slack downloads, etc.)."""
    assert safe_filename("🏥 lab.pdf") == "🏥 lab.pdf"


def test_nfd_input_normalized_to_nfc():
    """macOS HFS+ returns NFD; we store NFC so the same uploaded file
    has the same ``original_filename`` regardless of which OS first
    wrote the sentinel."""
    nfd = unicodedata.normalize("NFD", "café.pdf")
    nfc = unicodedata.normalize("NFC", "café.pdf")
    # Sanity: the two strings really differ at the byte level.
    assert nfd != nfc
    assert safe_filename(nfd) == nfc


def test_control_chars_become_underscore():
    """Control bytes (NUL, bell, ESC, etc.) become ``_`` — they never
    belong in real filenames and would break JSON-pretty inspection."""
    assert safe_filename("a\x00b\x07c\x1bd.pdf") == "a_b_c_d.pdf"


def test_del_character_replaced():
    """DEL (0x7F) is also a control char per ASCII; sanitize it too."""
    assert safe_filename("foo\x7fbar.pdf") == "foo_bar.pdf"


def test_path_separators_preserved():
    """``/`` and ``\\`` are legal JSON characters; we store them
    verbatim because ``original_filename`` is for display, not for
    rebuilding a filesystem path. If a caller later misuses the field
    that's the caller's bug, not ours to defend against here."""
    assert safe_filename("dir/sub/file.pdf") == "dir/sub/file.pdf"
    assert safe_filename(r"C:\Users\me\file.pdf") == r"C:\Users\me\file.pdf"


def test_max_length_unchanged_at_boundary():
    name = "a" * (MAX_FILENAME_LEN - 4) + ".pdf"
    assert len(name) == MAX_FILENAME_LEN
    assert safe_filename(name) == name


def test_long_name_truncates_preserving_extension():
    """``"a"*1000 + ".pdf"`` keeps the ``.pdf`` suffix."""
    name = "a" * 1000 + ".pdf"
    out = safe_filename(name)
    assert out is not None
    assert len(out) == MAX_FILENAME_LEN
    assert out.endswith(".pdf")
    assert out.startswith("a")


def test_long_name_without_real_extension_truncates_blind():
    """No ``.`` in the trailing 16 chars → just chop at 255 with no
    extension preservation."""
    name = "a" * 1000
    out = safe_filename(name)
    assert out == "a" * MAX_FILENAME_LEN


def test_long_extension_treated_as_no_extension():
    """A ``.aaaaaaaaaaaaaaaaa`` (17+ chars) is unlikely to be a real
    extension; truncate blind rather than special-casing it."""
    name = "filename." + ("a" * 30)
    out = safe_filename(name)
    assert out is not None
    # 30-char "extension" exceeds _MAX_SUFFIX_LEN — falls into blind
    # truncation path; output stays under cap and is the head of the
    # input.
    assert len(out) <= MAX_FILENAME_LEN
    assert out == name[:MAX_FILENAME_LEN]


def test_whitespace_trimmed():
    """Leading/trailing whitespace is trimmed (Ctrl+V drops sometimes
    carry a stray newline from terminal bracketed-paste)."""
    assert safe_filename("  report.pdf  \n") == "report.pdf"
