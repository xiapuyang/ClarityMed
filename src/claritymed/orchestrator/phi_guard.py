"""Outbound PHI guard (architecture §6.4 — the cloud-PHI red line).

``PhiGuard.check_outbound`` walks a payload along configured dot-paths and
redacts (or blocks) PHI before the orchestrator hands a prompt to a cloud
LLM. Rules are declared in ``configs/safety.yaml`` under the ``phi`` key
and reloaded via ``PhiGuard.from_config()`` whenever an admin updates them.

The path syntax is intentionally simple: dot-separated keys with ``*`` as a
list wildcard, e.g. ``patient.allergies.*.substance``. We resist a real
JSONPath dependency until the rule set demonstrates real-world complexity
the simple matcher cannot handle.

Per-user opt-in: when ``Account.cloud_provider_opt_in`` is True, ``deny``
becomes ``warn`` (the field still gets redacted, but the call proceeds).
This is the patient's "I know what I'm doing" knob.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from claritymed import config as _cfg

logger = logging.getLogger(__name__)

REDACTED = "[REDACTED]"

Provider = Literal["local", "cloud"]
Action = Literal["allowed", "redacted", "blocked"]


class PhiRules(BaseModel):
    """Parsed rules from ``configs/safety.yaml`` ``phi`` section."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fields: list[str] = Field(default_factory=list)
    providers: dict[Provider, Literal["allow", "deny"]] = Field(default_factory=dict)
    on_deny: Literal["redact", "raise"] = "redact"


class PhiHit(BaseModel):
    """One detection from ``check_outbound``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field_path: str
    original_type: str
    action: Action


class PhiGuard:
    """PHI policy. Construct from rules; call ``check_outbound`` per request."""

    def __init__(self, rules: PhiRules) -> None:
        self.rules = rules

    @classmethod
    def from_config(cls) -> "PhiGuard":
        _cfg.reload_configs()
        phi_raw = _cfg.load_yaml("safety.yaml").get("phi") or {}
        return cls(PhiRules.model_validate(phi_raw))

    def check_outbound(
        self,
        payload: dict[str, Any],
        provider_kind: Provider,
        cloud_opt_in: bool = False,
    ) -> tuple[dict[str, Any], list[PhiHit]]:
        """Walk the payload and either redact or block PHI fields.

        Returns (possibly-modified payload, list of hits). When the provider
        is allowed for PHI, the payload comes back untouched.
        """
        decision = self.rules.providers.get(provider_kind, "deny")
        if decision == "allow":
            return payload, []

        redacted = copy.deepcopy(payload)
        hits: list[PhiHit] = []
        for path in self.rules.fields:
            for resolved_path, original in self._iter_field(redacted, path):
                action: Action = "redacted"
                if self.rules.on_deny == "raise" and not cloud_opt_in:
                    action = "blocked"
                hits.append(
                    PhiHit(
                        field_path=resolved_path,
                        original_type=type(original).__name__,
                        action=action,
                    )
                )
                if action == "redacted":
                    self._set_in(redacted, resolved_path, REDACTED)
                else:
                    # raise mode: leave payload as-is; caller decides
                    pass
        return redacted, hits

    @staticmethod
    def _iter_field(payload: dict, path: str):
        """Yield (resolved_path, value) for every payload location matching path."""
        parts = path.split(".")
        yield from PhiGuard._walk(payload, parts, [])

    @staticmethod
    def _walk(node, parts, trail):
        if not parts:
            yield ".".join(str(p) for p in trail), node
            return
        head, *rest = parts
        if head == "*":
            if isinstance(node, list):
                for idx, item in enumerate(node):
                    yield from PhiGuard._walk(item, rest, [*trail, idx])
            return
        if isinstance(node, dict) and head in node:
            yield from PhiGuard._walk(node[head], rest, [*trail, head])

    @staticmethod
    def _set_in(payload: dict, path: str, value: Any) -> None:
        parts: list = []
        for p in path.split("."):
            parts.append(int(p) if p.isdigit() else p)
        node = payload
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = value
