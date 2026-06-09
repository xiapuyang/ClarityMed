"""``AuditRule`` protocol + ``RuleReport`` dataclass.

Streaming aggregator pattern: callers iterate the audit stream once and
hand each event to every registered rule, then collect reports at the
end. Memory cost is bounded by each rule's own accumulator (counters
+ a few samples), not by the log size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class RuleReport:
    """One rule's findings over a stream of audit events.

    * ``total_relevant``: how many events the rule deemed relevant
      (denominator for any rates the rule computes).
    * ``counts``: per-group tallies the operator wants to see; keys are
      rule-defined (e.g. ``"qwen3:14b"`` for a per-model breakdown).
    * ``findings``: short human-readable observations with recommended
      actions ("switch to deterministic mode", "tune prompt v4", etc.).
    * ``samples``: a few representative raw payloads so an operator can
      eyeball matches without re-greping the log.
    """

    name: str
    description: str
    total_relevant: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    samples: list[dict] = field(default_factory=list)


class AuditRule(Protocol):
    """One streaming audit-log rule.

    Stateful: ``accept`` is called once per audit event in chronological
    order; ``report`` is called once at the end. Implementations
    accumulate counters / samples internally.
    """

    name: str
    description: str

    def accept(self, event: dict) -> None:
        """Consider one event. May ignore."""
        ...

    def report(self) -> RuleReport:
        """Return the accumulated findings."""
        ...
