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
import re
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from claritymed import config as _cfg

if TYPE_CHECKING:
    from claritymed.core.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

REDACTED = "[REDACTED]"

Provider = Literal["local", "cloud"]
Action = Literal["allowed", "redacted", "blocked"]


class FreeTextRule(BaseModel):
    """One regex rule for free-text PII scrubbing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    regex: str
    replacement: str


class NerConfig(BaseModel):
    """Optional NER pass over free text. Off by default."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    entities: list[str] = Field(default_factory=list)


class ScrubReport(BaseModel):
    """Per-call summary of what ``scrub_free_text`` did. Counts only — no
    original spans are recorded, so the report is safe to emit as audit log.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_hits: dict[str, int] = Field(default_factory=dict)
    ner_hits: int = 0
    text_len_before: int = 0
    text_len_after: int = 0


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
    free_text_patterns: list[FreeTextRule] = Field(default_factory=list)
    ner: NerConfig = Field(default_factory=NerConfig)


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
        """Redact PII spans (phone, email, ID, MRN, etc.) in free text.

        This is the second PHI defense layer (M10) alongside the structured
        ``check_outbound`` (M6). It runs at two enforcement points: before
        chunks land in ``user_rag``, and before any LLM prompt is assembled —
        regardless of provider kind, because user privacy is not solely a
        cloud-egress concern.

        Returns the scrubbed text plus a counts-only report (safe for audit
        logging — no original spans included).
        """
        if not text:
            return text, ScrubReport(text_len_before=0, text_len_after=0)

        original_len = len(text)
        rule_hits: dict[str, int] = {}
        scrubbed = text

        for rule in self.rules.free_text_patterns:
            pattern = re.compile(rule.regex)
            new_text, count = pattern.subn(rule.replacement, scrubbed)
            if count > 0:
                rule_hits[rule.name] = count
                scrubbed = new_text

        ner_hits = 0
        if self.rules.ner.enabled:
            # NER pass is intentionally left as a hook — v1 ships with
            # regex-only. Wiring a local NER model (GLiNER, spaCy, etc.)
            # is a follow-up task gated on real free-text traffic.
            logger.debug("NER enabled but no model wired yet — skipping.")

        return scrubbed, ScrubReport(
            rule_hits=rule_hits,
            ner_hits=ner_hits,
            text_len_before=original_len,
            text_len_after=len(scrubbed),
        )

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
