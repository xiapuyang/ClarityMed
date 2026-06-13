"""Outbound PHI guard (architecture §6.4 — the cloud-PHI red line).

``PhiGuard.check_outbound`` walks a payload along configured dot-paths and
redacts (or blocks) PHI before the orchestrator hands a prompt to a cloud
LLM. Rules are declared in ``configs/safety.yaml`` under the ``phi`` key
and reloaded via ``PhiGuard.from_config()`` whenever an admin updates them.

The path syntax is intentionally simple: dot-separated keys with ``*`` as a
list wildcard, e.g. ``patient.allergies.*.substance``. We resist a real
JSONPath dependency until the rule set demonstrates real-world complexity
the simple matcher cannot handle.

Free-text PII scrubbing is handled by ``core.scrub.ScrubService`` —
``PhiGuard.scrub_free_text`` delegates there.
"""

from __future__ import annotations

import copy
import logging
import threading
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from claritymed import config as _cfg
from claritymed.core.scrub.service import ScrubConfig, ScrubReport, ScrubService

if TYPE_CHECKING:
    from claritymed.core.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

REDACTED = "[REDACTED]"

Provider = Literal["local", "cloud"]
Action = Literal["allowed", "redacted", "blocked"]


class ChunkFilterReport(BaseModel):
    """Per-call summary of ``filter_chunks_for_provider``. Counts only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total: int
    kept: int
    filtered_phi: int
    provider_kind: Provider


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


_GUARD_CACHE: "PhiGuard | None" = None
_GUARD_LOCK = threading.Lock()


def get_default_guard() -> "PhiGuard":
    """Return the process-wide cached ``PhiGuard``. Delegates to ``from_config``."""
    return PhiGuard.from_config()


def invalidate_guard_cache() -> None:
    """Force the next ``from_config`` / ``get_default_guard`` call to rebuild from disk."""
    global _GUARD_CACHE
    with _GUARD_LOCK:
        _GUARD_CACHE = None


class PhiGuard:
    """PHI policy. Construct from rules; call ``check_outbound`` per request."""

    def __init__(
        self,
        rules: PhiRules,
        scrub_service: ScrubService | None = None,
    ) -> None:
        self.rules = rules
        self._scrub = scrub_service or ScrubService(ScrubConfig())

    @classmethod
    def from_config(cls) -> "PhiGuard":
        """Return the process-wide cached ``PhiGuard``, building it on first call.

        Caching lives here so that any caller — ``get_default_guard()``,
        ``PhiAssertionModel.__init__``, or direct ``PhiGuard.from_config()``
        calls — all get the same instance.  Without this, a new
        ``ScrubService`` (and its 809 MB ONNX pipeline) would be constructed
        per ``build_model()`` call, accumulating in memory across benchmark
        trials or evals.

        Hot-reload: call ``invalidate_guard_cache()`` to force a rebuild on
        the next call (e.g. after editing ``safety.yaml`` at runtime).
        """
        global _GUARD_CACHE
        if _GUARD_CACHE is not None:  # fast path — no lock needed after init
            return _GUARD_CACHE
        with _GUARD_LOCK:
            if _GUARD_CACHE is not None:  # re-check inside lock
                return _GUARD_CACHE
            phi_raw = _cfg.load_yaml("safety.yaml").get("phi") or {}
            rules = PhiRules.model_validate(
                {
                    k: v
                    for k, v in phi_raw.items()
                    if k in {"fields", "providers", "on_deny"}
                }
            )
            _GUARD_CACHE = cls(rules, ScrubService.from_config())
            return _GUARD_CACHE

    def check_outbound(
        self,
        payload: dict[str, Any],
        provider_kind: Provider,
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
                if self.rules.on_deny == "raise":
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
        return redacted, hits

    def filter_chunks_for_provider(
        self,
        chunks: list["RetrievedChunk"],
        provider_kind: Provider,
    ) -> tuple[list["RetrievedChunk"], "ChunkFilterReport"]:
        """Filter retrieved chunks by the target provider's PHI policy.

        Local provider keeps everything. Cloud provider drops any chunk that
        is marked ``is_phi=True`` AND not opt-in as ``can_cloud=True``. The
        filter is the retrieval-layer companion to ``check_outbound`` — it
        operates on already-retrieved chunks just before they would be
        injected into the prompt.

        Returns the surviving chunks plus a counts report (safe for audit
        logging — no chunk content recorded).
        """
        from claritymed.core.schemas.retrieval import RetrievedChunk  # noqa: F401

        total = len(chunks)
        if provider_kind == "local":
            report = ChunkFilterReport(
                total=total,
                kept=total,
                filtered_phi=0,
                provider_kind=provider_kind,
            )
            return list(chunks), report

        kept: list[RetrievedChunk] = []
        filtered = 0
        for chunk in chunks:
            if chunk.is_phi and not chunk.can_cloud:
                filtered += 1
                continue
            kept.append(chunk)

        report = ChunkFilterReport(
            total=total,
            kept=len(kept),
            filtered_phi=filtered,
            provider_kind=provider_kind,
        )
        return kept, report

    def scrub_free_text(self, text: str) -> tuple[str, ScrubReport]:
        """Scrub PII from free text. Delegates to ``ScrubService``."""
        return self._scrub.scrub(text)

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
