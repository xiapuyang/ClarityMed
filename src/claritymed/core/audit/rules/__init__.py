"""Pluggable audit-log rules.

A rule is a small streaming aggregator: ``accept(event)`` is called once
per audit event during a scan, ``report()`` returns a structured
``RuleReport`` at the end. Adding a new rule is one class + one
factory entry — no CLI changes needed.
"""

from claritymed.core.audit.rules.base import AuditRule, RuleReport
from claritymed.core.audit.rules.factory import build_rules, list_rules

__all__ = [
    "AuditRule",
    "RuleReport",
    "build_rules",
    "list_rules",
]
