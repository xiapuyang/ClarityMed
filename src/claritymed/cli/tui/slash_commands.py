"""Slash command parser for the TUI input bar.

The parser is pure (no I/O, no Textual imports) so it can be tested directly.
The app turns the parsed command into a UI action — open a modal, switch user,
trigger a service call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CommandName = Literal[
    "upload",
    "library",
    "user",
    "provider",
    "lang",
    "clear",
    "help",
    "quit",
    "unknown",
    "not_command",
]

KNOWN_COMMANDS: tuple[str, ...] = (
    "upload",
    "library",
    "user",
    "provider",
    "lang",
    "clear",
    "help",
    "quit",
)


@dataclass(frozen=True)
class ParsedCommand:
    """Result of parsing a single line of user input."""

    name: CommandName
    arg: str = ""
    raw: str = ""

    @property
    def is_command(self) -> bool:
        return self.name not in ("not_command", "unknown")


def parse(line: str) -> ParsedCommand:
    """Parse one user input line.

    Returns:
        ``ParsedCommand(name="not_command")`` when the line does not start with
        ``/``; ``unknown`` when the head is unrecognised; otherwise a known
        command name plus the remainder as ``arg``.
    """
    stripped = line.lstrip()
    if not stripped.startswith("/"):
        return ParsedCommand(name="not_command", raw=line)
    parts = stripped[1:].split(maxsplit=1)
    if not parts:
        return ParsedCommand(name="unknown", raw=line)
    head = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""
    if head not in KNOWN_COMMANDS:
        return ParsedCommand(name="unknown", arg=head, raw=line)
    return ParsedCommand(name=head, arg=rest.strip(), raw=line)  # type: ignore[arg-type]


HELP_TEXT: str = (
    "Slash commands:\n"
    "  /upload <path>    Upload a file into your personal RAG library\n"
    "  /library [query]  Inspect collections (empty = list every collection,\n"
    "                    query = run full RAG retrieval incl. translation)\n"
    "  /user [id]        Switch active user (admin only; no arg opens picker)\n"
    "  /provider [id]    Switch LLM provider; no arg opens picker (F3)\n"
    "  /lang [code]      Switch reply language (en/zh); no arg opens picker\n"
    "  /clear            Start a new chat session (keeps history on disk)\n"
    "  /help             Show this help\n"
    "  /quit             Exit the TUI\n"
    "Mode is cycled with Shift+Tab (ingest / ask / rag). "
    "Anything not starting with / is routed automatically (default: ask)."
)
