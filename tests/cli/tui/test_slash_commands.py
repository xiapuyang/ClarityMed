"""Slash command parser tests — pure, no Textual required."""

from __future__ import annotations

import pytest

from claritymed.cli.tui.slash_commands import HELP_TEXT, parse


def test_help_command():
    p = parse("/help")
    assert p.name == "help"
    assert p.is_command
    assert p.arg == ""


def test_upload_command_with_arg():
    p = parse("/upload /tmp/report.pdf")
    assert p.name == "upload"
    assert p.arg == "/tmp/report.pdf"


def test_mode_command_arg_lowercased_in_app():
    # Parser preserves arg case; the app lowercases it before validation.
    p = parse("/mode INGEST")
    assert p.name == "mode"
    assert p.arg == "INGEST"


def test_unknown_command_keeps_head_in_arg():
    p = parse("/sproingify foo")
    assert p.name == "unknown"
    assert p.arg == "sproingify"
    assert not p.is_command


def test_plain_text_is_not_command():
    p = parse("what is hypertension?")
    assert p.name == "not_command"
    assert not p.is_command


def test_leading_whitespace_tolerated():
    p = parse("   /quit")
    assert p.name == "quit"


def test_slash_alone_is_unknown():
    p = parse("/")
    assert p.name == "unknown"


@pytest.mark.parametrize(
    "cmd", ["/upload", "/library", "/mode ask", "/user alice", "/help", "/quit"]
)
def test_help_text_documents_each_command(cmd):
    name = cmd[1:].split()[0]
    assert name in HELP_TEXT
