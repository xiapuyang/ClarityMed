"""Rule registry + factory.

Adding a new rule is one entry in ``_RULES`` plus a class implementing
the ``AuditRule`` protocol. ``list_rules`` powers the CLI ``--rule
<name>`` help text and the ``audit list-rules`` command.
"""

from __future__ import annotations

from claritymed.core.audit.rules.base import AuditRule
from claritymed.core.audit.rules.tool_announced import ToolAnnouncedButSkippedRule

_RULES: dict[str, type[AuditRule]] = {
    "tool_announced_but_skipped": ToolAnnouncedButSkippedRule,
}


def list_rules() -> list[tuple[str, str]]:
    """Return ``(name, description)`` pairs for every registered rule."""
    return [(name, cls.description) for name, cls in _RULES.items()]


def build_rules(only: list[str] | None = None) -> list[AuditRule]:
    """Instantiate the requested rules.

    Args:
        only: Subset of rule names to build. ``None`` builds every
            registered rule.

    Raises:
        KeyError: A name in ``only`` is not registered. Fail loud — a
            typo on the CLI must not silently no-op.
    """
    if only is None:
        return [cls() for cls in _RULES.values()]
    rules: list[AuditRule] = []
    for name in only:
        cls = _RULES.get(name)
        if cls is None:
            raise KeyError(f"unknown audit rule {name!r}; known: {sorted(_RULES)!r}")
        rules.append(cls())
    return rules
