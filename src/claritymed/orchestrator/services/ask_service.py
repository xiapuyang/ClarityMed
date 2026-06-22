"""Ask service: PHI scrub on input + drive the ask turn.

Owns the turn shape: PHI scrub, plugin composition (deterministic
pre-invoke text + tool registration), agent build, streaming, audit,
chat-session persistence. Per-feature behaviour (RAG, future vision /
symptoms) is delegated to ``FeaturePlugin`` instances built by
``core.features.build_features``; the service never branches on
"which feature" or "which mode" beyond that composition step.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Callable

from claritymed.core.events import (
    Done,
    Error,
    Event,
    LlmCallStarted,
    LlmFirstToken,
    TokenChunk,
    TokensUsed,
    ToolCompleted,
    ToolStarted,
)
from claritymed.core.features import TurnContext, build_features
from claritymed.core.interaction import InteractiveChannelUnavailable
from claritymed.core.observability.audit import audit_event
from claritymed.core.observability.latency import LatencyTrace, build_step_records
from claritymed.core.observability.latency import usage_dict as _usage_dict
from claritymed.core.observability.logging import get_access_logger
from claritymed.core.observability.tool_announce import detect_announcement
from claritymed.core.phi.guard import PhiGuard, get_default_guard
from claritymed.orchestrator.agents import make_ask_agent
from claritymed.orchestrator.agents.ask_deps import AskDeps

if TYPE_CHECKING:
    from pydantic_ai.models import Model

    from claritymed.core.features import FeaturePlugin
    from claritymed.core.interaction.prompt_channel import PromptChannel
    from claritymed.core.interaction.tool_approval_channel import ToolApprovalChannel
    from claritymed.core.rag.strategies.base import RagStrategy
    from claritymed.core.schemas import ProviderConfig
    from claritymed.core.schemas.retrieval import RetrievedChunk
    from claritymed.core.translation import TranslationProvider
    from claritymed.orchestrator.services.chat_session import ChatSession

logger = logging.getLogger(__name__)

_UNKNOWN = "?"

# Sliding-window history budget (bytes of serialized pydantic-ai messages).
# ~80 KB is roughly 20–25 K tokens — generous for chat history while
# leaving room for system prompt + tool schemas + completion in a 128 K
# context model. When exceeded, oldest request/response pairs are dropped
# from the front and a ``mode.ask.history_trimmed`` audit event records
# how many messages went.
HISTORY_BUDGET_BYTES = 80_000
# Floor on how many messages we always keep, regardless of size. Two
# pairs (4 messages) preserves enough recent context for the model to
# stay coherent even with a pathologically long single turn.
HISTORY_MIN_KEEP = 4

# How long to wait for in-flight OCR jobs to finish at turn start before
# proceeding with whatever status the attachment has. Caps the worst-case
# user wait at ~1 minute — local LLM OCR on a 30B+ model can take 20-40s
# per image; cloud paths are sub-10s. Past the cap the inline placeholder
# expansion renders ``ocr_status="pending"`` and the turn continues so a
# single stuck provider can't dead-end the chat.
OCR_AWAIT_TIMEOUT_S = 60.0
OCR_AWAIT_POLL_S = 0.25


# Matches the exact splice ``_compose_prompt`` emits when an Evidence
# block is present. Anchoring on both the ``Evidence (cite by [n]):``
# header AND the ``\n\nQuestion:`` separator keeps this conservative —
# we only strip blocks the framework itself produced, never a user
# message that happens to mention the phrase.
_EVIDENCE_BLOCK_RE = re.compile(
    r"\n*Evidence \(cite by \[n\]\):.*?\n\nQuestion:\s*",
    re.DOTALL,
)
# Citation marker like ``[1]`` / ``[12]``. We only consider it
# out-of-range — never strip legitimate bracketed text — by matching
# digits only.
_CITATION_RE = re.compile(r"\[(\d+)\]")


def _strip_evidence_block(text: str) -> str:
    """Remove one inline ``Evidence (cite by [n]):`` block from a stored prompt.

    pydantic-ai's message history captures the *full* request body we
    sent — including any retrieval evidence ``RagFeature.pre_invoke``
    spliced in. Carrying that into the next turn teaches the LLM that
    prior turns' citation indices (``[1]``..``[N]``) are still valid,
    which is how ``[11]`` showed up in an answer whose current turn only
    surfaced 2 sources. Strip the splice; the assistant's reply (which
    still references the indices) stays — that's fine, the indices are
    now opaque to the model.
    """
    return _EVIDENCE_BLOCK_RE.sub("", text, count=1)


class HistoryScrubFailed(Exception):
    """Privacy-filter model failed during cross-turn history scrub.

    Raised from the ``scrub`` callable passed to
    :func:`_sanitize_history_for_llm` when the ONNX privacy-filter model
    layer reports ``model_failed=True``. The caller is expected to
    catch this and refuse the cloud-bound request — mirrors the
    fail-loud contract of the user-input scrub at the top of
    ``_run_scoped`` (see ``"scrub_unavailable"`` Error path).
    """


def _sanitize_history_for_llm(messages: list, *, scrub=None) -> list:
    """Return a copy of message history with Evidence blocks removed.

    Touches ``UserPromptPart`` content in both shapes pydantic-ai uses:

    * Plain string content — Evidence block stripped, then (when
      ``scrub`` is supplied) the cleaned text scrubbed.
    * List-form multimodal content — each ``str`` element is scrubbed
      individually so PHI smuggled in via an OCR-text part during a
      prior local turn does not bypass the cloud-bound check. Non-string
      elements (``BinaryContent``, ``ImageUrl``, structured parts) pass
      through unchanged: they carry no free text to scrub. Evidence
      blocks are *not* stripped from list-form content because the
      retrieval pipeline never splices via list-form today.

    ``ModelRequest`` / ``UserPromptPart`` are dataclasses in pydantic-ai,
    so we ``dataclasses.replace`` rather than ``model_copy``.

    When ``scrub`` is provided (callable ``str -> str``), every text
    surface above is scrubbed. This is the cloud-turn defense against
    cross-turn PHI replay: local-turn prompts persisted into
    ``messages_json`` carry raw user input, and switching providers
    mid-session would otherwise leak that history to the cloud LLM
    unscrubbed. The scrub callable is supplied by the caller (typically
    a ``PhiGuard.scrub_free_text``-backed closure) so this helper stays
    decoupled from the guard. The callable may raise
    :class:`HistoryScrubFailed` to fail loud — this helper does not
    swallow it.
    """
    import dataclasses

    from pydantic_ai.messages import ModelRequest, UserPromptPart

    out: list = []
    for m in messages:
        if not isinstance(m, ModelRequest):
            out.append(m)
            continue
        new_parts: list = []
        changed = False
        for p in m.parts:
            if isinstance(p, UserPromptPart):
                if isinstance(p.content, str):
                    cleaned = _strip_evidence_block(p.content)
                    if scrub is not None:
                        cleaned = scrub(cleaned)
                    if cleaned != p.content:
                        new_parts.append(dataclasses.replace(p, content=cleaned))
                        changed = True
                        continue
                elif isinstance(p.content, list) and scrub is not None:
                    new_content: list = []
                    part_changed = False
                    for el in p.content:
                        if isinstance(el, str):
                            cleaned_el = scrub(el)
                            if cleaned_el != el:
                                part_changed = True
                            new_content.append(cleaned_el)
                        else:
                            new_content.append(el)
                    if part_changed:
                        new_parts.append(dataclasses.replace(p, content=new_content))
                        changed = True
                        continue
            new_parts.append(p)
        out.append(dataclasses.replace(m, parts=new_parts) if changed else m)
    return out


def _trim_message_history(messages, budget: int):
    """Drop oldest pairs until the serialized history fits the budget.

    Returns ``(trimmed, dropped_count)``. Trim works on request/response
    pairs (2 messages at a time) because dropping a lone request leaves
    the next response unanchored, which pydantic-ai rejects.

    Cost: at most O(log N) full serializations because we use bisection
    on the drop count when the initial estimate overshoots, and a single
    final serialization to confirm. The previous implementation
    re-encoded the entire kept list after every pair drop, which was
    quadratic in message count for big overshoots.
    """
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    if not messages:
        return messages, 0
    encoded_full = ModelMessagesTypeAdapter.dump_json(messages)
    if len(encoded_full) <= budget:
        return messages, 0
    max_drop_pairs = (len(messages) - HISTORY_MIN_KEEP) // 2
    if max_drop_pairs <= 0:
        return messages, 0
    # Bisect on the number of pairs to drop. `lo` is the largest known
    # drop count whose kept slice does NOT fit budget (starts at 0:
    # the full slice was already shown over-budget above). `hi` is the
    # smallest drop count whose kept slice fits, or `max_drop_pairs + 1`
    # as the open upper sentinel until we find one. We want the smallest
    # drop_pairs that fits, which is `hi` when the loop terminates.
    lo, hi = 0, max_drop_pairs + 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        candidate = messages[2 * mid :]
        if len(ModelMessagesTypeAdapter.dump_json(candidate)) <= budget:
            hi = mid
        else:
            lo = mid
    # Clamp when even the maximum allowed drop doesn't fit — we still
    # respect the HISTORY_MIN_KEEP floor and let the caller decide what
    # to do with an over-budget kept slice.
    drop_pairs = min(hi, max_drop_pairs)
    return messages[2 * drop_pairs :], drop_pairs * 2


class AskService:
    """Drive ask mode: scrub, compose features into a turn, stream.

    Boundaries the plugins do not own:

    * PHI scrubbing before any LLM call (cloud / local invariant).
    * Plugin composition — gather deterministic pre-invoke text +
      register tool-mode callables on the agent.
    * Streaming loop — merge LLM token stream with tool-emitted events.
    * Chat-session persistence (user turn before, assistant after).
    * Sources block injection just before ``Done``.
    * ``mode.ask`` audit row including per-feature modes.
    """

    def __init__(
        self,
        model: "Model",
        guard: PhiGuard | None = None,
        language: str = "en",
        chat_session: "ChatSession | None" = None,
        provider_id: str = _UNKNOWN,
        model_name: str = _UNKNOWN,
        *,
        strategy: "RagStrategy | None" = None,
        provider_config: "ProviderConfig | None" = None,
        user_whitelist: list[str] | None = None,
        translation_service: "TranslationProvider | None" = None,
        rag_mode: str = "tool",
        profile_context_mode: str = "deterministic",
        features: "list[FeaturePlugin] | None" = None,
        prompt_channel: "PromptChannel | None" = None,
        tool_approval_channel: "ToolApprovalChannel | None" = None,
        symptoms_factory: "Callable[[], FeaturePlugin] | None" = None,
        vision_factory: "Callable[[], FeaturePlugin] | None" = None,
    ) -> None:
        self._model = model
        self._guard = guard or get_default_guard()
        self._language = language
        self._chat_session = chat_session
        self._provider_id = provider_id
        self._model_name = model_name
        self._strategy = strategy
        self._provider_config = provider_config
        self._user_whitelist = user_whitelist
        self._translation_service = translation_service
        # Host-supplied interaction channel for the ``ask_user_question``
        # tool. None means the host is non-interactive — the tool body
        # falls back to a plain-text hint to the LLM rather than
        # blocking on a UI that does not exist.
        self._prompt_channel = prompt_channel
        # Host-supplied channel for the per-tool PHI write approval modal.
        # None ⇒ the ingest tools are NOT wired this turn (no UI to host
        # the prompt). Mirrors the ``prompt_channel`` gate: rather than
        # registering a toolset that will always be denied, we omit it
        # entirely from the LLM's view so the model picks a different
        # path (chat reply, ask_user_question) on its own.
        self._tool_approval_channel = tool_approval_channel
        # PromptRegistry walks every YAML in the store on construction.
        # When the channel is wired up we build the tool every turn, so
        # cache the registry once instead of paying disk + Pydantic
        # validation cost on each ``run()``.
        self._prompt_registry = None

        # Plugins are stateless w.r.t. turn data — built once at startup.
        # Test paths can inject a pre-built list to assert dispatch
        # without going through the factory.
        # AttachmentsFeature needs the session_id at turn time; bind a
        # closure over the (potentially-late-bound) ``_chat_session`` ref
        # so a session swapped in after construction is picked up.
        def _current_session_id() -> str | None:
            return (
                self._chat_session.session_id
                if self._chat_session is not None
                else None
            )

        if profile_context_mode != "off":
            from claritymed.core.features.profile_context_plugin import (
                ProfileContextFeature,
            )

            _pcm = profile_context_mode

            def _profile_context_factory() -> FeaturePlugin:
                return ProfileContextFeature(mode=_pcm)

        else:
            _profile_context_factory = None

        self._features = (
            features
            if features is not None
            else build_features(
                rag_mode=rag_mode,
                rag_strategy=strategy,
                get_session_id=_current_session_id,
                ingest_factory=(
                    self._build_ingest_factory(_current_session_id)
                    if tool_approval_channel is not None and chat_session is not None
                    else None
                ),
                symptoms_factory=symptoms_factory,
                vision_factory=vision_factory,
                profile_context_factory=_profile_context_factory,
            )
        )
        # Snapshot per-feature modes for the audit row; the LLM-facing
        # tool list is computed per-turn from ``as_tool``.
        self._feature_modes = {f.name: f.mode for f in self._features}
        self._last_chunks: list = []
        # Vision registry's catalog cross-check is deferred to first turn:
        # ``make_vision_factory`` tries it in a sync ``asyncio.run`` which
        # raises inside an already-running loop (TUI path). Bench/e2e
        # bootstrap explicitly before constructing AskService, so the
        # call here is idempotent in those paths.
        self._vision_bootstrap_attempted = False

    @property
    def last_chunks(self) -> list:
        """Retrieved chunks from the most recent run() call (for testing)."""
        return self._last_chunks

    async def _run_post_process_hooks(self, deps, result: dict) -> None:
        """Invoke ``post_process`` on every plugin that implements
        :class:`~claritymed.core.features.base.PostProcessHook`.

        The hook itself short-circuits when its tool was not called
        this turn — implementers track that via their own per-request
        state (e.g. the symptoms plugin's request-id stash). Keeping
        the dispatch unconditional means the orchestrator does not
        need to know which tool name belongs to which plugin.

        Audit-only by contract — implementers should not mutate the
        user-visible reply (KTD-2 / KTD-3). Failures are caught and
        logged at warning level; a buggy hook must not break the reply.
        """
        from claritymed.core.features.base import PostProcessHook

        text = result.get("final_text") or ""
        for plugin in self._features:
            if not isinstance(plugin, PostProcessHook):
                continue
            try:
                new_text = await plugin.post_process(text, result)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "post_process hook on plugin %s raised; reply unchanged",
                    plugin.name,
                    exc_info=True,
                )
                continue
            if isinstance(new_text, str):
                result["final_text"] = new_text
                text = new_text

    def _build_ingest_factory(
        self,
        get_session_id: "Callable[[], str | None]",
    ) -> "Callable[[], FeaturePlugin]":
        """Return a closure that constructs ``IngestToolsFeature`` on demand.

        Bound at construction time but evaluated lazily by ``build_features``
        so the heavy stores (SettingsStore, ToolDispatcher) only materialize
        when ingest tools are actually wired. Closures over ``self`` are
        intentional — the dispatcher's callables need access to the current
        user_id + session_id via ContextVars at tool-call time, not at
        AskService construction.
        """
        from typing import Any as _Any

        from claritymed.context import user_id_ctx
        from claritymed.orchestrator.features.ingest_tools_plugin import (
            IngestToolsFeature,
        )
        from claritymed.orchestrator.services.tool_dispatcher import ToolDispatcher
        from claritymed.stores.session_attachments import (
            SessionAttachments,
        )
        from claritymed.stores.settings_store import SettingsStore

        def _session_shas() -> set[str]:
            uid = user_id_ctx.get()
            sid = get_session_id()
            if uid is None or sid is None:
                return set()
            try:
                rows = SessionAttachments(uid, sid).list()
            except Exception:  # noqa: BLE001
                logger.exception("ingest: failed to list session attachments")
                return set()
            return {row.sha256 for row in rows}

        def _rule_match(tool_name: str, args: dict) -> str | None:
            uid = user_id_ctx.get()
            if uid is None:
                return None
            try:
                rule = SettingsStore(uid).match_rule(tool_name, args)
            except Exception:  # noqa: BLE001
                logger.exception("ingest: settings store match_rule failed")
                return None
            if rule is None or rule.action != "allow":
                return None
            return rule.id

        # Holder mutated by ``_factory`` so ``_approval_required`` (which
        # ``ApprovalRequiredToolset`` captures at construction time) can
        # reach the live dispatcher. Declared before the closure so
        # static-analysis tools see it bound.
        dispatcher_slot: dict = {}

        def _approval_required(_ctx: _Any, tool_def: _Any, args: dict) -> bool:
            """Decide if a tool call needs the modal.

            Runs the full ``ToolDispatcher.gate`` pipeline: schema
            validation, sha256 set check, record_path containment, then
            allow-rule lookup. A validation failure raises a typed
            exception that pydantic-ai surfaces to the LLM as a tool
            error - the modal never opens for a malformed call. A rule
            hit returns ``False`` (no approval needed); otherwise
            ``True`` and the framework raises ``ApprovalRequired`` so
            AskService can drive the modal.

            Deny rules are matched by ``_resolve_approvals`` after the
            framework emits ``DeferredToolRequests`` so the audit row
            carries the rule id and the user sees an explicit
            ``Tool denied`` event rather than a silent skip.
            """
            dispatcher = dispatcher_slot.get("dispatcher")
            if dispatcher is None:
                return True
            # ``RunContext.retry`` is the per-tool-name retry counter
            # pydantic-ai bumps via ``ToolManager.for_run_step`` after a
            # failed step. attempt=0 means first try; attempt=N means
            # this is the (N+1)-th time the model has emitted this tool
            # name in the current ``agent.run``. Logging it here pairs
            # with ``tool_args_invalid`` (logged from the dispatcher when
            # validation fails) so a maintainer can grep app.log and see
            # the full ``(tool_name, args, attempt, outcome)`` trail.
            attempt = getattr(_ctx, "retry", 0)
            logger.info(
                "tool_call tool=%s attempt=%d args=%s",
                tool_def.name,
                attempt,
                args,
            )
            result = dispatcher.gate(tool_def.name, args)
            return not result.allowed

        def _resolve_tool_prompt_language() -> str:
            """Pick the language for the seven tool descriptions.

            Order: ``CLARITYMED_TOOL_PROMPT_LANG`` env (en / zh) wins,
            then the per-turn user language. The env override decouples
            tool-prompt language from chat language, letting operators
            A/B which language the model handles tool calls better in —
            an EN tool description with a ZH conversation is a valid
            configuration, since the model only reads the description
            once at agent construction.
            """
            override = os.environ.get("CLARITYMED_TOOL_PROMPT_LANG", "").strip().lower()
            if override in ("en", "zh"):
                return override
            return self._language

        def _factory() -> "FeaturePlugin":
            dispatcher = ToolDispatcher(
                session_attachments=_session_shas,
                rule_match=_rule_match,
            )
            dispatcher_slot["dispatcher"] = dispatcher
            return IngestToolsFeature(
                dispatcher=dispatcher,
                approval_required_func=_approval_required,
                language=_resolve_tool_prompt_language(),
            )

        return _factory

    async def _resolve_approvals(
        self,
        deferred,
        user_id: str,
        out: "asyncio.Queue[Event | None]",
    ):
        """Drive the approval channel for each pending tool call.

        Always returns a ``DeferredToolResults`` (never ``None``).
        Cancellation / channel-unavailable is signalled by inserting
        ``ToolDenied`` entries into the ``approvals`` map for the
        affected calls — the caller does not need a separate None branch
        for that case. The defensive ``if resolved is None`` guard at
        the call site is kept as belt-and-braces against future
        refactors that introduce a real None return.

        Each iteration emits ``ToolStarted`` / ``ToolCompleted`` events
        so the UI shows progress through the modals.

        Decision mapping:

        * ``once``         → ``ToolApproved``.
        * ``always_tool``  → persist ``allow`` rule (empty pattern,
          default TTL) + ``ToolApproved``.
        * ``deny``         → ``ToolDenied``.

        Deny rules are evaluated here (not in ``approval_required_func``)
        so the audit row carries the matched rule id, and so the user
        gets a clean ``Tool denied`` event in the stream instead of a
        silent skip.
        """
        from pydantic_ai.tools import (
            DeferredToolResults,
            ToolApproved,
            ToolDenied,
        )

        from claritymed.core.interaction import ApprovalDecision
        from claritymed.stores.settings_store import SettingsStore

        channel = self._tool_approval_channel
        approvals: dict = {}
        calls = list(deferred.approvals)
        total = len(calls)
        store = SettingsStore(user_id)
        cancelled = False
        for idx, call in enumerate(calls):
            tool_name = call.tool_name
            args = (
                call.args_as_dict()
                if hasattr(call, "args_as_dict")
                else (call.args if isinstance(call.args, dict) else {})
            )
            await out.put(
                ToolStarted(tool_name=tool_name, args_preview=f"{idx + 1}/{total}")
            )
            t_start = time.perf_counter()
            # Deny rule short-circuit: framework should have allowed
            # nothing through ``approval_required_func`` for these
            # calls, but a deny rule can still match and pre-empt the
            # modal entirely.
            try:
                deny_rule = store.match_rule(tool_name, args)
            except Exception:  # noqa: BLE001
                logger.exception("resolve_approvals: rule lookup failed")
                deny_rule = None
            if deny_rule is not None and deny_rule.action == "deny":
                approvals[call.tool_call_id] = ToolDenied(
                    message=f"Denied by rule {deny_rule.id[:8]}."
                )
                audit_event(
                    "tool.approval.denied",
                    {
                        "user_id": user_id,
                        "tool_name": tool_name,
                        "rule_id": deny_rule.id,
                        "reason": "deny_rule",
                    },
                )
                await out.put(
                    ToolCompleted(
                        tool_name=tool_name,
                        duration_ms=int((time.perf_counter() - t_start) * 1000),
                        summary="denied by rule",
                    )
                )
                continue

            if cancelled or channel is None:
                approvals[call.tool_call_id] = ToolDenied(
                    message="No approval channel available."
                )
                audit_event(
                    "tool.approval.denied",
                    {
                        "user_id": user_id,
                        "tool_name": tool_name,
                        "reason": "channel_unavailable",
                    },
                )
                await out.put(
                    ToolCompleted(
                        tool_name=tool_name,
                        duration_ms=int((time.perf_counter() - t_start) * 1000),
                        summary="channel unavailable",
                    )
                )
                continue

            try:
                decision: ApprovalDecision = await channel.request(
                    tool_name,
                    args,
                    breadcrumb=f"Tool {idx + 1}/{total}",
                )
            except asyncio.CancelledError:
                audit_event(
                    "tool.cancelled_by_shutdown",
                    {"user_id": user_id, "tool_name": tool_name},
                )
                approvals[call.tool_call_id] = ToolDenied(
                    message="Cancelled by shutdown."
                )
                # Subsequent calls in this batch also denied without
                # opening another modal — the host UI is going away.
                cancelled = True
                await out.put(
                    ToolCompleted(
                        tool_name=tool_name,
                        duration_ms=int((time.perf_counter() - t_start) * 1000),
                        summary="cancelled",
                    )
                )
                continue
            except InteractiveChannelUnavailable as exc:
                logger.warning(
                    "resolve_approvals: channel unavailable for %s: %s",
                    tool_name,
                    exc,
                )
                approvals[call.tool_call_id] = ToolDenied(
                    message="Approval UI unavailable."
                )
                audit_event(
                    "tool.approval.denied",
                    {
                        "user_id": user_id,
                        "tool_name": tool_name,
                        "reason": "channel_unavailable",
                    },
                )
                await out.put(
                    ToolCompleted(
                        tool_name=tool_name,
                        duration_ms=int((time.perf_counter() - t_start) * 1000),
                        summary="channel unavailable",
                    )
                )
                continue

            kind = decision.decision
            if kind == "deny":
                approvals[call.tool_call_id] = ToolDenied(
                    message="User denied this tool call."
                )
                audit_event(
                    "tool.approval.denied",
                    {"user_id": user_id, "tool_name": tool_name},
                )
                summary = "denied"
            elif kind == "once":
                approvals[call.tool_call_id] = ToolApproved()
                audit_event(
                    "tool.approval.granted",
                    {"user_id": user_id, "tool_name": tool_name, "scope": "once"},
                )
                summary = "approved (once)"
            elif kind == "always_tool":
                try:
                    rule = store.add_rule(tool_name, {}, action="allow")
                    audit_event(
                        "tool.approval.granted",
                        {
                            "user_id": user_id,
                            "tool_name": tool_name,
                            "scope": "always_tool",
                            "rule_id": rule.id,
                        },
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("failed to persist always_tool rule")
                approvals[call.tool_call_id] = ToolApproved()
                summary = "approved (always_tool)"
            else:  # pragma: no cover — Decision Literal covers all branches
                approvals[call.tool_call_id] = ToolDenied(
                    message=f"Unknown decision: {kind!r}"
                )
                summary = "denied (unknown decision)"

            await out.put(
                ToolCompleted(
                    tool_name=tool_name,
                    duration_ms=int((time.perf_counter() - t_start) * 1000),
                    summary=summary,
                )
            )

        # ``DeferredToolResults`` carries approvals only; deferred.calls
        # (non-approval deferred tools) are passed through unchanged
        # because v1 has none of those.
        return DeferredToolResults(approvals=approvals, calls={}, metadata={})

    async def run(self, user_input: str, user_id: str) -> AsyncIterator[Event]:
        from claritymed.context import (
            apply_context,
            new_request_id,
            request_id_ctx,
            reset_context,
        )

        rid = request_id_ctx.get() or new_request_id()
        tokens = apply_context(rid, user_id, self._language)
        try:
            await self._bootstrap_vision_once()
            async for ev in self._run_inner(user_input, user_id):
                yield ev
        finally:
            reset_context(tokens)

    async def _bootstrap_vision_once(self) -> None:
        """Run the vision registry's catalog cross-check on the first turn.

        ``make_vision_factory`` cannot bootstrap synchronously when it
        is constructed from inside a running asyncio loop (the TUI
        path), so it logs ``vision: bootstrap deferred`` and hands back
        an unverified registry. This method honors that deferral by
        calling :meth:`VisionFeature.ensure_bootstrapped`, which writes
        the failure into ``vision.disabled_reason`` so other surfaces
        (TUI toast, ``<image vision_disabled="…">`` attribute) can react.
        On failure we also drop the feature from the active session so
        the LLM never sees a broken tool description.

        TUI mount may have already called ``ensure_bootstrapped`` —
        that's fine; the underlying method is idempotent.
        """
        if self._vision_bootstrap_attempted:
            return
        self._vision_bootstrap_attempted = True
        from claritymed.orchestrator.features.vision_plugin import VisionFeature

        vision = next((f for f in self._features if isinstance(f, VisionFeature)), None)
        if vision is None:
            return
        reason = await vision.ensure_bootstrapped()
        if reason is not None:
            self._features = [f for f in self._features if f is not vision]
            self._feature_modes.pop(vision.name, None)

    async def _run_inner(self, user_input: str, user_id: str) -> AsyncIterator[Event]:
        from opentelemetry import trace as otel_trace

        tracer = otel_trace.get_tracer("claritymed.ask")
        with tracer.start_as_current_span("ask.request"):
            async for ev in self._run_scoped(user_input, user_id):
                yield ev

    async def _run_scoped(self, user_input: str, user_id: str) -> AsyncIterator[Event]:
        from claritymed.context import (
            apply_context,
            attach_session_baggage,
            detach_session_baggage,
            language_ctx,
            request_id_ctx,
        )

        # Capture for the finally block. When the TUI cancels mid-stream, the
        # consumer's finally runs reset_context() before this generator is
        # aclose()'d by asyncio's finalizer — which fires in a different task
        # context where our ContextVars are unset. We rehydrate from these
        # captures so audit_event() in the cancellation path still works.
        captured_rid = request_id_ctx.get()
        captured_lang = language_ctx.get() or self._language

        # PHI scrub only for cloud-bound turns.
        # scrub_free_text loads the ONNX session on first call (can take
        # 10-60 s); running it in a thread keeps the event loop free.
        is_cloud = getattr(self._provider_config, "kind", None) == "cloud"
        if is_cloud:
            scrubbed, report = await asyncio.to_thread(
                self._guard.scrub_free_text, user_input
            )
            audit_event(
                "mode.ask.scrub",
                payload={
                    "user_id": user_id,
                    "rule_hits": report.rule_hits,
                    "model_hits": report.model_hits,
                    "model_failed": report.model_failed,
                    "text_len_before": report.text_len_before,
                    "text_len_after": report.text_len_after,
                },
            )
            if report.model_failed:
                # privacy_filter.enabled=true is the safety contract.
                # If the model layer fails for a cloud-bound turn, fail
                # loud instead of silently leaking PHI the regex layer
                # missed. The user sees an Error event; nothing reaches
                # the LLM.
                logger.error(
                    "privacy-filter model failed on cloud turn; refusing to "
                    "send unscrubbed text to %s",
                    self._provider_id,
                )
                yield Error(
                    error_type="scrub_unavailable",
                    message=(
                        "Privacy filter is configured but unavailable; "
                        "refusing to send unscrubbed text to the cloud "
                        "provider. Switch to a local provider or fix the "
                        "filter setup, then retry."
                    ),
                    retryable=False,
                )
                return
        else:
            scrubbed = user_input

        output_lang = self._language

        deps = AskDeps(
            strategy=self._strategy,
            user_id=user_id,
            user_whitelist=self._user_whitelist,
            provider_config=self._provider_config,
            language=output_lang,
            translation_service=self._translation_service,
            mode=self._feature_modes.get("rag", "tool"),
            prompt_channel=self._prompt_channel,
        )

        # User turn first so the JSONL timeline reflects send order.
        if self._chat_session is not None:
            try:
                self._chat_session.append_user(scrubbed)
            except Exception:  # noqa: BLE001
                logger.exception("failed to append user turn to chat session")
        message_history = (
            self._chat_session.message_history() if self._chat_session else None
        )
        if message_history:
            message_history, dropped = _trim_message_history(
                message_history, HISTORY_BUDGET_BYTES
            )
            if dropped:
                audit_event(
                    "mode.ask.history_trimmed",
                    payload={
                        "user_id": user_id,
                        "dropped_messages": dropped,
                        "kept_messages": len(message_history),
                        "budget_bytes": HISTORY_BUDGET_BYTES,
                    },
                )

        session_token = (
            attach_session_baggage(self._chat_session.session_id)
            if self._chat_session is not None
            else None
        )
        result: dict = {
            "final_text": "",
            "messages_json": None,
            "usage": None,
            "steps": [],
            "latency": None,
            "had_error": False,
        }
        finalized = False
        try:
            async for event in self._stream_turn(
                scrubbed, deps, message_history, user_id, result
            ):
                if isinstance(event, Done):
                    # Finalize while still inside the generator so _finalize_turn
                    # runs before the consumer closes us on Done.  Moving this
                    # after `yield event` would make it dead code because the TUI
                    # returns immediately when it receives Done.
                    if not result["had_error"]:
                        # Clamp before persist so future turns' history
                        # never carries out-of-range markers forward.
                        from claritymed.core.rag.retrieval_pipeline import (
                            deduplicate_chunks,
                        )

                        valid_max = len(deduplicate_chunks(deps.retrieved_chunks))
                        cleaned, offending = self._clamp_citations(
                            result["final_text"], valid_max
                        )
                        if offending:
                            audit_event(
                                "ask.citation.out_of_range",
                                payload={
                                    "user_id": user_id,
                                    "valid_max": valid_max,
                                    "offending": offending,
                                },
                            )
                            result["final_text"] = cleaned
                            # The user already saw the bad markers in the
                            # streamed text. Emit a TokenChunk that lists
                            # the dropped indices so the UI can render a
                            # short correction line under the answer,
                            # before the Sources block. Keep the message
                            # ASCII so it round-trips in any locale.
                            offending_str = ", ".join(f"[{n}]" for n in offending)
                            yield TokenChunk(
                                text=(
                                    "\n\n*Note: the markers "
                                    f"{offending_str} above point to sources "
                                    f"beyond the {valid_max} listed below "
                                    "and have been removed from the saved "
                                    "transcript.*"
                                )
                            )
                        self._finalize_turn(user_id, result, deps)
                        finalized = True
                    if deps.retrieved_chunks:
                        self._last_chunks = list(deps.retrieved_chunks)
                        yield TokenChunk(
                            text=self._format_sources(
                                deps.retrieved_chunks, lang=self._language
                            )
                        )
                        if os.environ.get("CLARITYMED_DEBUG"):
                            yield TokenChunk(
                                text=self._format_debug_collections(
                                    deps.retrieved_chunks
                                )
                            )
                yield event
        finally:
            # When the consumer cancels mid-stream (TUI Esc) or the
            # generator is GC'd without ever seeing Done, persist a
            # short cancelled-turn record so resume sees the question
            # paired with an empty assistant reply rather than a
            # dangling user message. Best-effort: the chat session may
            # already be closed.
            #
            # Rehydrate ContextVars from captures if the consumer's reset
            # already fired. We don't bother resetting after — the alien
            # context this runs in (asyncio finalizer) is discarded anyway.
            if request_id_ctx.get() is None and captured_rid:
                apply_context(captured_rid, user_id, captured_lang)
            if not finalized and self._chat_session is not None:
                try:
                    self._chat_session.append_assistant(
                        text=result["final_text"],
                        messages_json=result["messages_json"] or b"",
                        model=self._model_name,
                        provider_id=self._provider_id,
                        usage=result["usage"],
                        latency=result["latency"],
                        steps=result["steps"],
                        cancelled=True,
                    )
                    audit_event(
                        "mode.cancelled",
                        payload={
                            "user_id": user_id,
                            "session_id": self._chat_session.session_id,
                            "had_error": bool(result["had_error"]),
                        },
                    )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "failed to persist cancelled turn for user %s", user_id
                    )
            detach_session_baggage(session_token)

    async def _wait_for_ocr_and_expand_placeholders(
        self, scrubbed: str, deps: AskDeps, user_id: str
    ) -> tuple[str, TurnContext]:
        """Block on in-flight OCR then expand attachment placeholders.

        Without the OCR wait the user can paste an image, hit Enter
        immediately, and the inline placeholder expansion renders
        ``ocr_status="pending"`` to the LLM — which then answers without
        the extracted text. Capped at ``OCR_AWAIT_TIMEOUT_S`` so a stuck
        provider can't block the turn forever.

        Placeholder expansion runs BEFORE any feature sees the prompt so
        ``RagFeature.pre_invoke``'s query-rewrite call sees real OCR'd
        text, not the raw ``[Image sha:abcd1234]`` placeholder (which
        would otherwise hallucinate a query from a placeholder string).
        Only placeholders the user kept in their text get expanded —
        session attachments without a matching placeholder this turn
        stay out of the prompt so "delete the placeholder" remains a
        meaningful UI gesture.

        Returns the (possibly rewritten) scrubbed text and a fresh
        :class:`TurnContext` reflecting it.
        """
        from claritymed.core.attachments_feature import AttachmentsFeature

        turn_ctx = TurnContext(scrubbed=scrubbed, deps=deps)
        if self._chat_session is not None:
            await self._await_pending_ocr(user_id, self._chat_session.session_id)
        attachments_feature = next(
            (f for f in self._features if isinstance(f, AttachmentsFeature)),
            None,
        )
        if attachments_feature is not None:
            scrubbed = await attachments_feature.expand_placeholders(scrubbed, turn_ctx)
            turn_ctx = TurnContext(scrubbed=scrubbed, deps=deps)
        return scrubbed, turn_ctx

    async def _run_deterministic_pre_invoke(
        self, turn_ctx: TurnContext, deps: AskDeps
    ) -> tuple[str, "Event | None", list["Event"]]:
        """Run every deterministic feature's ``pre_invoke``; join the blocks.

        Order matters: features run in registration order so future
        plugins can rely on stable layout (e.g. vision findings always
        above RAG evidence). A failing ``pre_invoke`` short-circuits the
        loop and returns an ``Error`` event in the second slot — the
        caller is responsible for yielding it.

        Returns ``(joined_text, error_event_or_None, drained_events)``.
        ``drained_events`` are events the deterministic features queued
        on ``deps.event_queue`` (e.g. ``RetrievalPending``) that the
        caller should yield before ``LlmCallStarted``.
        """
        pre_blocks: list[str] = []
        error: "Event | None" = None
        for feature in self._features:
            if feature.mode != "deterministic":
                continue
            try:
                text = await feature.pre_invoke(turn_ctx)
            except Exception as exc:  # noqa: BLE001
                logger.exception("pre_invoke failed for %s", feature.name)
                error = Error(
                    error_type="config_error",
                    message=f"{feature.name}.pre_invoke: {exc}",
                    retryable=False,
                )
                break
            if text:
                pre_blocks.append(text)
        drained: list["Event"] = []
        while not deps.event_queue.empty():
            try:
                drained.append(deps.event_queue.get_nowait())
            except Exception:  # noqa: BLE001
                break
        return "\n\n".join(pre_blocks), error, drained

    async def _scrub_assembled_prompt_for_cloud(
        self, prompt: str, user_id: str
    ) -> tuple[str, "Event | None"]:
        """Second-pass PHI scrub on the assembled cloud-bound prompt.

        Attachment OCR expansion and feature ``pre_invoke`` blocks join
        the prompt AFTER the user-input scrub. For cloud-bound turns
        the final prompt must pass through the guard once more so OCR'd
        PHI and profile-context fields don't ride past the gate. Local
        turns skip this pass — the user already consented to send raw
        text to their own machine.

        Returns ``(prompt, error_event_or_None)``. The error is set
        when the privacy-filter model layer fails — caller yields it
        and aborts the turn.
        """
        if getattr(self._provider_config, "kind", None) != "cloud":
            return prompt, None
        scrubbed, report = await asyncio.to_thread(self._guard.scrub_free_text, prompt)
        audit_event(
            "mode.ask.scrub_assembled",
            payload={
                "user_id": user_id,
                "rule_hits": report.rule_hits,
                "model_hits": report.model_hits,
                "model_failed": report.model_failed,
                "text_len_before": report.text_len_before,
                "text_len_after": report.text_len_after,
            },
        )
        if report.model_failed:
            logger.error(
                "privacy-filter model failed on assembled cloud prompt; "
                "refusing to send unscrubbed pre_blocks/attachments to %s",
                self._provider_id,
            )
            return scrubbed, Error(
                error_type="scrub_unavailable",
                message=(
                    "Privacy filter is configured but unavailable for "
                    "the assembled prompt; refusing to send unscrubbed "
                    "context to the cloud provider. Switch to a local "
                    "provider or fix the filter setup, then retry."
                ),
                retryable=False,
            )
        return scrubbed, None

    def _build_agent_for_turn(self) -> tuple[Any, bool]:
        """Construct the per-turn pydantic-ai agent.

        Collects tools and toolsets across every feature, conditionally
        registers ``ask_user_question`` when a prompt channel is wired,
        builds the union output type (``str | DeferredToolRequests``
        when any toolset is present so approval-required tool calls
        bubble back as deferred requests instead of looping forever),
        and resolves dynamic system prompts from features that expose
        ``system_prompt_fn``.

        Returns ``(agent, any_tool)`` — the boolean lets the caller
        decide whether to wire ``UsageLimits`` (only meaningful when
        the LLM can loop via tool calls).
        """
        tools: list = [t for f in self._features if (t := f.as_tool()) is not None]
        toolsets: list = [
            ts for f in self._features if (ts := f.as_toolset()) is not None
        ]
        if self._prompt_channel is not None:
            from claritymed.config import ask_user_question_max_retries
            from claritymed.core.interaction import build_ask_user_question_tool
            from claritymed.core.prompts.registry import PromptRegistry

            if self._prompt_registry is None:
                self._prompt_registry = PromptRegistry()
            tools.append(
                build_ask_user_question_tool(
                    self._prompt_registry,
                    language=self._language,
                    max_retries=ask_user_question_max_retries(),
                )
            )
        any_tool = bool(tools) or bool(toolsets)
        if toolsets:
            from pydantic_ai.tools import DeferredToolRequests

            agent_output_type: Any = str | DeferredToolRequests
        else:
            agent_output_type = str
        # ``tool_proposal`` carries the LLM-facing meta-rules for the
        # seven write tools (when to propose save_record vs save_allergy,
        # how to reference attachments by sha256, etc.). Only relevant
        # when at least one approval-gated toolset is wired.
        extra_prompts = ["tool_proposal"] if toolsets else None
        # ``symptoms_final_reply`` is injected dynamically: SymptomsFeature
        # sets deps.symptoms_reply_guide only when the tool returns a real
        # differential. The dynamic system_prompt fn is a no-op on
        # user_declined / eligible:false / server_error turns, so no
        # extra tokens are burned when the sub-session produces no result.
        dynamic_sys_prompts = [
            f.system_prompt_fn()
            for f in self._features
            if hasattr(f, "system_prompt_fn")
        ]
        agent = make_ask_agent(
            self._model,
            language=self._language,
            tools=tools,
            toolsets=toolsets,
            output_type=agent_output_type,
            extra_prompt_names=extra_prompts,
            dynamic_system_prompts=dynamic_sys_prompts or None,
        )
        return agent, any_tool

    async def _stream_turn(
        self,
        scrubbed: str,
        deps: AskDeps,
        message_history,
        user_id: str,
        result: dict,
    ) -> AsyncIterator[Event]:
        """Compose feature plugins into one turn and stream events.

        Deterministic features run pre-LLM and contribute prompt text;
        tool features register callables on the agent. The same loop
        handles both shapes — the only branch is whether the prompt
        gets a prepended evidence block.

        The body delegates each named phase to a helper so this method
        stays readable: OCR + placeholder expansion, deterministic
        pre-invoke, cloud-side assembled-prompt scrub, agent construction,
        and the producer / drainer event loop. Helpers return
        ``(payload, error_event_or_None)`` so the generator only yields
        from one place per phase.
        """
        from pydantic_ai import UsageLimits
        from pydantic_ai.exceptions import UsageLimitExceeded

        scrubbed, turn_ctx = await self._wait_for_ocr_and_expand_placeholders(
            scrubbed, deps, user_id
        )

        pre_text, pre_error, drained_events = await self._run_deterministic_pre_invoke(
            turn_ctx, deps
        )
        for ev in drained_events:
            yield ev
        if pre_error is not None:
            result["had_error"] = True
            yield pre_error
            return

        prompt = f"{pre_text}\n\nQuestion: {scrubbed}" if pre_text else scrubbed

        prompt, scrub_error = await self._scrub_assembled_prompt_for_cloud(
            prompt, user_id
        )
        if scrub_error is not None:
            result["had_error"] = True
            yield scrub_error
            return

        agent, any_tool = self._build_agent_for_turn()

        out: asyncio.Queue[Event | None] = asyncio.Queue()
        st: dict = {
            "t_start": time.perf_counter(),
            "t_first_token": None,
            "t_end": 0.0,
        }

        async def _producer() -> None:
            logger.debug("_producer: START")
            await out.put(
                LlmCallStarted(
                    model_name=self._model_name,
                    provider_id=self._provider_id,
                )
            )
            audit_event(
                "llm.call.start",
                payload={
                    "user_id": user_id,
                    "provider_id": self._provider_id,
                    "model": self._model_name,
                },
            )
            get_access_logger().info(
                "llm.call.start model=%s provider=%s",
                self._model_name,
                self._provider_id,
            )
            # ``UsageLimits`` only matters when at least one tool is
            # registered; without tools the LLM cannot loop. Keep the
            # backstop on tool turns so a misbehaving model cannot run
            # the retrieval pipeline 50× per turn.
            # When the current turn is cloud, re-scrub every prior
            # ``UserPromptPart`` in the carried history. Local turns
            # persist raw user text into ``messages_json``; without this
            # the next cloud turn replays unscrubbed PHI from earlier
            # turns. Recomputed here so the producer is self-contained
            # rather than closing over a flag from the outer scope.
            producer_is_cloud = getattr(self._provider_config, "kind", None) == "cloud"
            history_scrub = None
            if producer_is_cloud and message_history:

                def _history_scrub(s: str) -> str:
                    # Capture the report so we can fail loud when the
                    # privacy-filter model layer drops out. Matches the
                    # ``"scrub_unavailable"`` posture of the user-input
                    # scrub: regex-only output is never enough to send
                    # to a cloud provider — the regex layer is documented
                    # as the floor, the model layer is the actual
                    # contract.
                    scrubbed, report = self._guard.scrub_free_text(s)
                    if report.model_failed:
                        raise HistoryScrubFailed(s[:40])
                    return scrubbed

                history_scrub = _history_scrub
            # ``run_stream`` would stop the agent graph at the first model
            # output matching ``output_type`` (``str``) — meaning when the
            # model emits intermediate text *together with* a tool call in
            # one response, the text is treated as final and the tool call
            # is silently dropped. That broke ``ask_user_question``: the
            # modal never opened on Ollama-style local models that emit
            # "为了更准确地评估…" + ``ask_user_question(…)`` in the same
            # message. Use ``run`` with an ``event_stream_handler`` so the
            # graph runs to completion (tools execute) while text deltas
            # still stream live to the UI.
            from pydantic_ai.messages import (
                PartDeltaEvent,
                PartStartEvent,
                TextPart,
                TextPartDelta,
            )

            async def _handle_events(_ctx, events) -> None:  # noqa: ANN001
                async for event in events:
                    text: str | None = None
                    if isinstance(event, PartStartEvent) and isinstance(
                        event.part, TextPart
                    ):
                        text = event.part.content
                    elif isinstance(event, PartDeltaEvent) and isinstance(
                        event.delta, TextPartDelta
                    ):
                        text = event.delta.content_delta
                    if not text:
                        continue
                    if st["t_first_token"] is None:
                        st["t_first_token"] = time.perf_counter()
                        ttft_ms = int((st["t_first_token"] - st["t_start"]) * 1000)
                        logger.debug("_producer: FIRST TOKEN ttft=%dms", ttft_ms)
                        await out.put(LlmFirstToken(ttft_ms=ttft_ms))
                    await out.put(TokenChunk(text=text))

            base_run_kwargs: dict = {
                "deps": deps,
                "event_stream_handler": _handle_events,
            }
            if any_tool:
                from claritymed.config import load_yaml

                _request_limit: int = (
                    load_yaml("app.yaml").get("agent", {}).get("request_limit", 10)
                )
                base_run_kwargs["usage_limits"] = UsageLimits(
                    request_limit=_request_limit
                )
            try:
                # Offload to thread because ``history_scrub`` runs the
                # ONNX privacy-filter pipeline per ``UserPromptPart`` —
                # on long sessions that would block the event loop for
                # seconds. The user-input scrub a few lines up uses the
                # same pattern.
                initial_history = (
                    await asyncio.to_thread(
                        _sanitize_history_for_llm,
                        message_history,
                        scrub=history_scrub,
                    )
                    if message_history
                    else None
                )
            except HistoryScrubFailed:
                logger.error(
                    "privacy-filter model failed during history scrub on "
                    "cloud turn; refusing to replay prior turns to %s",
                    self._provider_id,
                )
                audit_event(
                    "mode.ask.scrub",
                    payload={
                        "user_id": user_id,
                        "stage": "history",
                        "model_failed": True,
                    },
                )
                result["had_error"] = True
                await out.put(
                    Error(
                        error_type="scrub_unavailable",
                        message=(
                            "Privacy filter is configured but unavailable; "
                            "refusing to replay prior turns to the cloud "
                            "provider. Switch to a local provider or fix "
                            "the filter setup, then retry."
                        ),
                        retryable=False,
                    )
                )
                await out.put(None)
                return
            try:
                # Deferred-tool resume loop. The first iteration sends the
                # user prompt; subsequent iterations replay the previous
                # turn's full message history plus the approval results so
                # the framework executes the now-approved tools and runs
                # the model's follow-up. Cap iterations as a defense in
                # depth — a sane gate plus a sane model produces at most
                # 1 deferred round per turn.
                from pydantic_ai.tools import DeferredToolRequests

                run_result = None
                deferred_results = None
                max_resume_rounds = 4
                for round_idx in range(max_resume_rounds + 1):
                    run_kwargs = dict(base_run_kwargs)
                    if deferred_results is None:
                        run_kwargs["message_history"] = initial_history
                        logger.debug("_producer: ENTER agent.run (initial)")
                        run_result = await agent.run(prompt, **run_kwargs)
                    else:
                        run_kwargs["message_history"] = run_result.all_messages()  # type: ignore[union-attr]
                        run_kwargs["deferred_tool_results"] = deferred_results
                        logger.debug(
                            "_producer: ENTER agent.run (resume round=%d)",
                            round_idx,
                        )
                        run_result = await agent.run(**run_kwargs)
                    logger.debug("_producer: agent.run returned")

                    if isinstance(run_result.output, DeferredToolRequests):
                        if round_idx == max_resume_rounds:
                            # The previous design force-denied here and then
                            # `continue`'d — but `continue` on the last loop
                            # iteration exits the loop, so the force-denials
                            # never reached agent.run and the user got an
                            # empty answer. Surface a visible failure
                            # instead: after N denied resume rounds the
                            # model has shown it can't terminate; another
                            # force-deny round is wishful thinking. A clean
                            # Error is more honest than an empty bubble.
                            logger.warning(
                                "_producer: exceeded %d deferred rounds; "
                                "ending turn with scrub_unavailable-style "
                                "Error",
                                max_resume_rounds,
                            )
                            audit_event(
                                "tool.approval.denied",
                                {
                                    "user_id": user_id,
                                    "reason": "round_limit_exceeded",
                                    "rounds": max_resume_rounds,
                                },
                            )
                            await out.put(
                                Error(
                                    error_type="llm_error",
                                    message=(
                                        f"Model exceeded {max_resume_rounds} "
                                        "tool-approval rounds without "
                                        "producing a final answer. Try "
                                        "rephrasing the request or denying "
                                        "the tools manually."
                                    ),
                                    retryable=True,
                                )
                            )
                            result["had_error"] = True
                            break
                        resolved = await self._resolve_approvals(
                            run_result.output, user_id, out
                        )
                        if resolved is None:
                            # cancelled / channel unavailable across all
                            # calls — exit loop with current state.
                            result["had_error"] = True
                            break
                        deferred_results = resolved
                        continue

                    # str output → terminal.
                    result["final_text"] = run_result.output
                    # PostProcessHook fan-out (KTD-2 audit-only):
                    # the symptoms plugin uses this to scan the reply
                    # for tier-appropriate safety keywords. Hook
                    # returns the (possibly-rewritten) text, but for
                    # v1 every implementer is audit-only — text rides
                    # through unchanged. Failures are swallowed so a
                    # buggy hook never breaks the reply.
                    await self._run_post_process_hooks(deps, result)
                    break

                if run_result is not None and not result["had_error"]:
                    try:
                        result["messages_json"] = run_result.all_messages_json()
                    except Exception:  # noqa: BLE001
                        logger.exception("failed to capture pydantic-ai messages")
                    try:
                        result["usage"] = run_result.usage
                    except Exception:  # noqa: BLE001
                        logger.exception("failed to capture pydantic-ai usage")
                    try:
                        result["steps"] = build_step_records(
                            list(run_result.new_messages())
                        )
                    except Exception:  # noqa: BLE001
                        logger.exception("failed to build per-step records")
            except UsageLimitExceeded:
                logger.warning("_producer: agent.run hit request_limit cap")
                result["had_error"] = True
                from claritymed.core.i18n import t

                await out.put(
                    Error(
                        error_type="usage_limit",
                        message=t("errors.usage_limit", lang=self._language),
                        retryable=True,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "_producer: agent.run raised %s: %s",
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                result["had_error"] = True
                await out.put(
                    Error(error_type="llm_error", message=str(exc), retryable=True)
                )
            finally:
                logger.debug("_producer: FINALLY (sentinel → out)")
                st["t_end"] = time.perf_counter()
                await out.put(None)

        async def _drain_tools() -> None:
            # Forward tool-emitted events from the deps queue concurrently
            # with the text stream so RetrievalPending / ToolStarted / etc
            # surface immediately rather than piling up behind the next
            # text chunk. The earlier version polled at 10Hz with
            # ``wait_for(get(), timeout=0.1)``, burning ~300 spurious
            # scheduler entries per 30 s call. A plain ``await get()``
            # plus the consumer's ``cancel()`` in finally is sufficient,
            # and any unexpected exception is re-raised so the consumer
            # loop sees it instead of silently dying.
            while True:
                try:
                    ev = await deps.event_queue.get()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("event-queue drain raised; re-raising")
                    raise
                await out.put(ev)

        producer_task = asyncio.create_task(_producer())
        drain_task = asyncio.create_task(_drain_tools())

        try:
            while True:
                ev = await out.get()
                if ev is None:
                    break
                yield ev
            # Normal completion: flush any events still in the queues
            # so trailing telemetry (ToolCompleted, Steps, etc.) is not
            # lost. Done OUTSIDE the finally — yielding from a finally
            # block raises ``RuntimeError: async generator ignored
            # GeneratorExit`` when the consumer cancels (TUI Esc),
            # which previously crashed every cancelled turn.
            while not out.empty():
                try:
                    ev = out.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if ev is not None:
                    yield ev
            while not deps.event_queue.empty():
                try:
                    yield deps.event_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        finally:
            # Cancellation cleanup only — never yield here. On cancel
            # we drop pending events: the consumer is gone, and these
            # events are non-critical telemetry (the audit row was
            # written in _finalize_turn before Done).
            producer_task.cancel()
            drain_task.cancel()
            await asyncio.gather(producer_task, drain_task, return_exceptions=True)

        t_end = st["t_end"]
        t_start = st["t_start"]
        t_first_token = st["t_first_token"]
        result["latency"] = LatencyTrace(
            total_ms=int((t_end - t_start) * 1000),
            ttft_ms=(
                int((t_first_token - t_start) * 1000)
                if t_first_token is not None
                else None
            ),
            completion_ms=(
                int((t_end - t_first_token) * 1000)
                if t_first_token is not None
                else None
            ),
        )
        if result["had_error"]:
            return
        # TokensUsed before Done so consumers that close on Done still
        # see the usage row. ``usage`` may be None when the run failed
        # before recording a RunUsage (kept defensive even though the
        # had_error branch above already returned).
        usage = result.get("usage")
        if usage is not None:
            from claritymed.core.llm.context_window import estimate_context_window

            input_tokens = (
                getattr(usage, "input_tokens", None)
                or getattr(usage, "request_tokens", None)
                or 0
            )
            output_tokens = (
                getattr(usage, "output_tokens", None)
                or getattr(usage, "response_tokens", None)
                or 0
            )
            total_tokens = getattr(usage, "total_tokens", None) or (
                input_tokens + output_tokens
            )
            yield TokensUsed(
                model_name=self._model_name,
                provider_id=self._provider_id,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                total_tokens=int(total_tokens),
                context_window=estimate_context_window(self._model_name),
            )
        yield Done(final=result["final_text"])

    async def _await_pending_ocr(self, user_id: str, session_id: str) -> None:
        """Block until every ``pending`` attachment in the session reaches a
        terminal OCR status (``done`` / ``empty`` / ``failed``).

        Polls the on-disk sentinel via ``BlobStore.ocr_done`` rather than
        subscribing to ``OcrCompleted`` events because (a) the worker is
        owned by the TUI app, not AskService — wiring an event channel
        across that boundary would mean threading a queue through three
        layers; (b) the sentinel is the same source of truth
        ``SessionAttachments.mark_ocr_status`` writes to, so polling
        observes exactly what the next
        ``AttachmentsFeature.expand_placeholders`` call would see anyway.

        Capped at ``OCR_AWAIT_TIMEOUT_S``. On timeout we just return —
        the inline placeholder expansion still renders
        ``ocr_status="pending"`` so the LLM sees the state explicitly.
        """
        from claritymed.stores.session_attachments import (
            SessionAttachments,
        )
        from claritymed.stores.blob_store import BlobStore

        try:
            rows = SessionAttachments(user_id, session_id).list()
        except Exception:  # noqa: BLE001
            logger.exception("ocr-await: session attachments unreadable")
            return

        pending_shas = [r.sha256 for r in rows if r.ocr_status == "pending"]
        if not pending_shas:
            return

        blob_store = BlobStore(user_id)
        deadline = time.monotonic() + OCR_AWAIT_TIMEOUT_S
        logger.info(
            "ocr-await: waiting on %d pending blob(s) (timeout=%.0fs)",
            len(pending_shas),
            OCR_AWAIT_TIMEOUT_S,
        )
        while pending_shas:
            pending_shas = [s for s in pending_shas if not blob_store.ocr_done(s)]
            if not pending_shas:
                logger.info("ocr-await: all pending blobs settled")
                return
            if time.monotonic() >= deadline:
                logger.warning(
                    "ocr-await: timed out with %d still pending: %s",
                    len(pending_shas),
                    [s[:8] for s in pending_shas],
                )
                return
            await asyncio.sleep(OCR_AWAIT_POLL_S)

    def _finalize_turn(self, user_id: str, result: dict, deps: AskDeps) -> None:
        """Audit + chat-session persistence after the stream closes."""
        latency = result["latency"]
        if self._chat_session is not None and result["messages_json"] is not None:
            try:
                self._chat_session.append_assistant(
                    text=result["final_text"],
                    messages_json=result["messages_json"],
                    model=self._model_name,
                    provider_id=self._provider_id,
                    usage=result["usage"],
                    latency=latency,
                    steps=result["steps"],
                )
            except Exception:  # noqa: BLE001
                logger.exception("failed to append assistant turn to chat session")

        self._maybe_audit_announced_but_skipped(user_id, result, deps)

        payload: dict[str, object] = {
            "user_id": user_id,
            "answer_len": len(result["final_text"]),
            "model": self._model_name,
            "provider_id": self._provider_id,
            "latency_ms": latency.total_ms if latency else 0,
            "features": dict(self._feature_modes),
        }
        if latency and latency.ttft_ms is not None:
            payload["ttft_ms"] = latency.ttft_ms
        if latency and latency.completion_ms is not None:
            payload["completion_ms"] = latency.completion_ms
        if result["usage"] is not None:
            payload.update(_usage_dict(result["usage"]))
        if result["steps"]:
            payload["steps"] = result["steps"]
        if self._chat_session is not None:
            payload["session_id"] = self._chat_session.session_id
        if deps.tool_calls:
            payload["tool_calls"] = dict(deps.tool_calls)
        audit_event("mode.ask", payload=payload)
        ttft_str = (
            f" ttft={latency.ttft_ms}ms"
            if latency and latency.ttft_ms is not None
            else ""
        )
        tok_str = (
            f" tokens={result['usage'].total_tokens}"
            if result["usage"] is not None
            and getattr(result["usage"], "total_tokens", None)
            else ""
        )
        get_access_logger().info(
            "llm.call.done total=%dms%s%s model=%s",
            latency.total_ms if latency else 0,
            ttft_str,
            tok_str,
            self._model_name,
        )

    def _maybe_audit_announced_but_skipped(
        self, user_id: str, result: dict, deps: AskDeps
    ) -> None:
        """Emit ``mode.ask.tool_announced_but_skipped`` when warranted.

        Skipped if RAG is not in tool mode (deterministic always
        retrieves; agentic is disabled). Otherwise: regex the final
        text; if a match is present and the retrieve tool call count
        is still zero, emit one audit row. The matched snippet is NOT
        recorded — the LLM may have paraphrased user PHI back, and the
        signal we actually need (a count by provider/model) does not
        require the text. The pattern itself is captured for triage.
        """
        if self._feature_modes.get("rag") != "tool":
            return
        if deps.tool_calls.get("retrieve_medical_literature", 0) > 0:
            return
        snippet = detect_announcement(result["final_text"])
        if not snippet:
            return
        payload: dict[str, object] = {
            "user_id": user_id,
            "tool": "retrieve_medical_literature",
            "model": self._model_name,
            "provider_id": self._provider_id,
            "snippet_len": len(snippet),
        }
        if self._chat_session is not None:
            payload["session_id"] = self._chat_session.session_id
        audit_event("mode.ask.tool_announced_but_skipped", payload=payload)

    @staticmethod
    def _clamp_citations(text: str, max_n: int) -> tuple[str, list[int]]:
        """Strip ``[N]`` markers where ``N`` exceeds ``max_n``.

        Defense-in-depth against citation hallucination. ``max_n`` is the
        number of deduplicated Sources entries this turn — that's what
        the user sees, so any ``[N]`` above that is unmoored. Returns
        the cleaned text plus the sorted, deduplicated list of offending
        indices so the caller can audit them.

        We strip rather than substitute ``[?]`` because the stripped
        form reads cleanly when the user re-opens the conversation; an
        in-line ``[?]`` is just visual noise once the streaming UI has
        already rendered the original (incorrect) bracket.
        """
        bad: list[int] = []

        def _sub(m: "re.Match[str]") -> str:
            n = int(m.group(1))
            if n > max_n:
                bad.append(n)
                return ""
            return m.group(0)

        cleaned = _CITATION_RE.sub(_sub, text)
        return cleaned, sorted(set(bad))

    @staticmethod
    def _format_evidence(chunks: "list[RetrievedChunk]") -> str:
        from claritymed.core.rag.retrieval_pipeline import format_evidence

        return format_evidence(chunks)

    @staticmethod
    def _collection_label(chunk: "RetrievedChunk", lang: str | None) -> str:
        """Look up the user-facing label for a chunk's corpus.

        ``user_rag`` collapses to one shared label (e.g. ``My Library``)
        across every per-user store — ``user_rag_alice`` etc. is an
        internal id we never surface. ``system_rag`` chunks look up
        ``rag.collection.<collection_name>`` and fall back to the raw
        ``collection_name`` when no translation exists, so adding a new
        corpus needs only an i18n entry, never a code change.
        """
        from claritymed.core.i18n import t

        if chunk.source == "user_rag":
            key = "rag.collection.user_rag"
            label = t(key, lang=lang)
            return label if label != key else "My Library"
        name = chunk.collection_name or "source"
        key = f"rag.collection.{name}"
        label = t(key, lang=lang)
        return label if label != key else name

    @staticmethod
    def _format_sources(chunks: "list[RetrievedChunk]", lang: str | None = None) -> str:
        """Build the user-facing Sources section.

        Layout: ``[N] <title-or-uri> · <corpus-label>``. The corpus
        label comes from i18n (``configs/i18n/*.yaml`` under
        ``rag.collection.*``) so ``my library`` translates to ``我的
        资料库`` automatically and new corpora can be added without
        editing Python.

        This block is appended *after* ``_finalize_turn`` has persisted
        ``result["final_text"]``, so the corpus label never enters chat
        history and the LLM cannot mimic it on the next turn.
        """
        if not chunks:
            return ""
        from claritymed.core.rag.retrieval_pipeline import deduplicate_chunks

        unique = deduplicate_chunks(chunks)
        lines = ["\n\n**Sources:**"]
        for i, c in enumerate(unique, start=1):
            doc_title = c.doc_title.replace("_", " ") if c.doc_title else None
            title = c.source_uri or doc_title
            corpus = AskService._collection_label(c, lang)
            display = f"{title} · {corpus}" if title else corpus
            lines.append(f"- [{i}] {display}")
        return "\n".join(lines)

    @staticmethod
    def _format_debug_collections(chunks: "list[RetrievedChunk]") -> str:
        """Markdown block listing the RAG-collection backing each Sources entry.

        Only emitted when ``CLARITYMED_DEBUG=1``. Indices match the
        Sources block (same dedup-by-doc_id, same first-seen order) so a
        reader can map ``[N]`` in the answer ↔ Sources ``[N]`` ↔ Debug
        ``[N]``. When more than one chunk of the same doc passed
        retrieval, the line reports the best score and a chunk count
        rather than spawning a second row — that information lives in
        the ``rag.retrieval`` audit row already.
        """
        lines = ["\n\n---\n**Debug — RAG Collections:**"]
        groups: dict[str, list] = {}
        order: list[str] = []
        for c in chunks:
            if c.doc_id not in groups:
                groups[c.doc_id] = []
                order.append(c.doc_id)
            groups[c.doc_id].append(c)
        for i, doc_id in enumerate(order, start=1):
            members = groups[doc_id]
            first = members[0]
            col = first.collection_name or "unknown"
            scores = [
                c.rerank_score if c.rerank_score is not None else c.score
                for c in members
            ]
            scores = [s for s in scores if s is not None]
            best = max(scores) if scores else None
            score_str = f"{best:.3f}" if best is not None else "—"
            n = len(members)
            chunk_suffix = f", {n} chunks" if n > 1 else ""
            doc = first.doc_id or "—"
            lines.append(f"- [{i}] `{col}` · `{doc}` (score {score_str}{chunk_suffix})")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _compose_prompt(scrubbed: str, evidence_block: str) -> str:
        if not evidence_block:
            return scrubbed
        return f"{evidence_block}\n\nQuestion: {scrubbed}"
