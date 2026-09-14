"""Approval-rule persistence layer for the tool dispatcher.

Lives under ``approvals.rules`` in the existing per-user
``settings.yaml`` so admins (and the user themselves) can see/revoke
without grokking a separate format. Each rule binds:

* ``tool`` — the tool name (e.g. ``save_allergy``).
* ``args_pattern`` — a strict-subset-equality match against the args
  dict; ``sha256`` / ``record_path`` / ``attachments`` are never part
  of a pattern (always re-prompt regardless of rule).
* ``ttl_hours`` — TTL window in hours. ``granted_at`` + ``ttl_hours``
  derive ``expires_at`` at write time.

Key invariants:

* **Replace-on-duplicate**: a second ``add_rule`` for the same
  ``(tool, args_pattern)`` replaces the existing entry (extends TTL
  from now). Never stacks.
* **20 rules per tool cap** with oldest-evicted; eviction audits
  ``tool.rule_evicted`` so operators can spot a misbehaving caller
  that keeps minting rules.
* **Strict pydantic validation**: any single malformed rule in the
  YAML → treat the full list as empty (fail closed) + audit
  ``settings.rules.load_failed``. Same for total-corrupt YAML.
* **Evaluation order**: explicit ``deny`` → explicit ``allow`` →
  fall through (prompt). First match wins.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from claritymed.config import tool_approval_rule_ttl_hours
from claritymed.core.locks import file_lock
from claritymed.core.observability.audit import audit_event
from claritymed.stores.paths import user_settings_path, validate_user_id

logger = logging.getLogger(__name__)

# Fields that never participate in pattern matching — the brainstorm
# decision. A rule grant covers a *shape* of call, not a specific blob.
_OPAQUE_PATTERN_KEYS = frozenset({"sha256", "record_path", "attachments"})
_RULES_PER_TOOL_CAP = 20
_DEFAULT_TTL_HOURS = 24  # overridden at call time by tool_approval_rule_ttl_hours()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ApprovalRule(BaseModel):
    """One persisted rule entry."""

    model_config = ConfigDict(extra="forbid")

    id: str
    tool: str = Field(min_length=1, max_length=64)
    action: Literal["allow", "deny"] = "allow"
    args_pattern: dict[str, Any] = Field(default_factory=dict)
    granted_at: datetime
    ttl_hours: int = Field(ge=1, le=720)
    expires_at: datetime

    def matches(self, tool: str, args: dict[str, Any]) -> bool:
        """Strict-subset-equality on the pattern keys; opaque keys ignored."""
        if tool != self.tool:
            return False
        for key, value in self.args_pattern.items():
            if key in _OPAQUE_PATTERN_KEYS:
                continue
            if args.get(key) != value:
                return False
        return True


class SettingsStore:
    """Per-user rules accessor backed by ``settings.yaml``.

    ``approvals.rules`` is the only key this store touches; other
    callers (``AccountStore``) keep using their own keys unchanged.
    """

    def __init__(self, user_id: str) -> None:
        self.user_id = validate_user_id(user_id)
        self.path: Path = user_settings_path(self.user_id)

    # --- read --------------------------------------------------------

    def list_rules(self, *, now: datetime | None = None) -> list[ApprovalRule]:
        """Return all rules, pruning expired ones from the on-disk copy.

        ``now`` is overridable for tests; production callers pass nothing
        and get ``datetime.now(UTC)``.

        Fast path (no expired rows): no lock — the read is a single YAML
        load and the caller does not need write semantics.

        Slow path (prune triggers a write): take the same ``file_lock``
        ``add_rule`` / ``revoke_rule`` use and **re-read inside the lock**
        before persisting. Without the re-read, a concurrent ``add_rule``
        that lands between our pre-lock load and our write would be
        silently overwritten by our stale snapshot — that was the
        race adversarial reviewer ADV-006 surfaced (a freshly persisted
        ``always_tool`` grant disappearing because a sibling
        ``list_rules`` call was mid-prune).
        """
        now = now or _utcnow()
        raw = self._load_raw()
        rules = self._parse_rules(raw)
        live = [r for r in rules if r.expires_at > now]
        if len(live) == len(rules):
            return live
        with file_lock(self._lock_path()):
            raw = self._load_raw()
            rules = self._parse_rules(raw)
            live = [r for r in rules if r.expires_at > now]
            if len(live) != len(rules):
                self._persist_rules(raw, live)
                for r in rules:
                    if r.expires_at <= now:
                        audit_event(
                            "tool.rule_expired",
                            {"rule_id": r.id, "tool": r.tool},
                        )
        return live

    def match_rule(
        self,
        tool: str,
        args: dict[str, Any],
        *,
        now: datetime | None = None,
    ) -> ApprovalRule | None:
        """Return the first matching non-expired rule, or None.

        Evaluation order is deny → allow so a deny rule beats an allow
        rule on the same shape (one user override stops a stale grant).
        """
        rules = self.list_rules(now=now)
        for action in ("deny", "allow"):
            for rule in rules:
                if rule.action == action and rule.matches(tool, args):
                    return rule
        return None

    # --- write -------------------------------------------------------

    def add_rule(
        self,
        tool: str,
        args_pattern: dict[str, Any],
        *,
        ttl_hours: int | None = None,
        action: Literal["allow", "deny"] = "allow",
    ) -> ApprovalRule:
        """Persist a rule. Replace-on-duplicate; evict oldest above cap."""
        if ttl_hours is None:
            ttl_hours = tool_approval_rule_ttl_hours()
        now = _utcnow()
        rule = ApprovalRule(
            id=str(uuid.uuid4()),
            tool=tool,
            action=action,
            args_pattern={
                k: v for k, v in args_pattern.items() if k not in _OPAQUE_PATTERN_KEYS
            },
            granted_at=now,
            ttl_hours=ttl_hours,
            expires_at=now + timedelta(hours=ttl_hours),
        )

        with file_lock(self._lock_path()):
            raw = self._load_raw()
            existing = self._parse_rules(raw)
            # Replace any duplicate (same tool + same pattern shape).
            kept = [
                r
                for r in existing
                if not (r.tool == rule.tool and r.args_pattern == rule.args_pattern)
            ]
            kept.append(rule)
            # Per-tool cap with oldest-evicted.
            per_tool: dict[str, list[ApprovalRule]] = {}
            for r in kept:
                per_tool.setdefault(r.tool, []).append(r)
            for t, rs in list(per_tool.items()):
                if len(rs) > _RULES_PER_TOOL_CAP:
                    rs.sort(key=lambda r: r.granted_at)
                    while len(rs) > _RULES_PER_TOOL_CAP:
                        evicted = rs.pop(0)
                        audit_event(
                            "tool.rule_evicted",
                            {"rule_id": evicted.id, "tool": t},
                        )
                    per_tool[t] = rs
            final = [r for rs in per_tool.values() for r in rs]
            self._persist_rules(raw, final)
        return rule

    def revoke_rule(self, rule_id: str) -> bool:
        """Remove a rule by id. Returns True iff it existed."""
        with file_lock(self._lock_path()):
            raw = self._load_raw()
            existing = self._parse_rules(raw)
            before = len(existing)
            kept = [r for r in existing if r.id != rule_id]
            if len(kept) == before:
                return False
            self._persist_rules(raw, kept)
            return True

    # --- internals ---------------------------------------------------

    def _lock_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".lock")

    def _load_raw(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        except yaml.YAMLError:
            audit_event(
                "settings.rules.load_failed",
                {"reason": "yaml parse"},
            )
            logger.warning("settings.yaml parse failed; treating as empty")
            return {}

    def _parse_rules(self, raw: dict[str, Any]) -> list[ApprovalRule]:
        approvals = raw.get("approvals", {}) if isinstance(raw, dict) else {}
        items = approvals.get("rules") if isinstance(approvals, dict) else None
        if not isinstance(items, list):
            return []
        out: list[ApprovalRule] = []
        for entry in items:
            try:
                out.append(ApprovalRule.model_validate(entry))
            except ValidationError:
                audit_event(
                    "settings.rules.load_failed",
                    {"reason": "rule schema"},
                )
                logger.warning("approvals.rules entry rejected; treating list as empty")
                # Fail closed: any single bad rule discards the whole list.
                return []
        return out

    def _persist_rules(self, raw: dict[str, Any], rules: list[ApprovalRule]) -> None:
        raw = dict(raw)
        approvals = dict(raw.get("approvals") or {}) if isinstance(raw, dict) else {}
        approvals["rules"] = [r.model_dump(mode="json") for r in rules]
        raw["approvals"] = approvals
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(raw, fh, sort_keys=False, allow_unicode=True)
            tmp.replace(self.path)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise
