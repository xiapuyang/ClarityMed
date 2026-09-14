"""Audit-log analysis (post-hoc; never on the request path).

``reader.read_audit_events`` walks rotated ``audit.log*`` files under
``CLARITYMED_LOG_DIR`` and yields one parsed event dict per line, after
optional time / user filters. Rules consume that stream via
``AuditRule.accept`` and emit a ``RuleReport`` summarising what they
found.

CLI entry: ``claritymed audit scan [--rule NAME] [--since ...]``.
"""

from claritymed.core.audit.reader import read_audit_events
from claritymed.core.audit.rules import AuditRule, RuleReport, build_rules, list_rules

__all__ = [
    "AuditRule",
    "RuleReport",
    "build_rules",
    "list_rules",
    "read_audit_events",
]
