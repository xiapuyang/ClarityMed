"""Vision feature — disease detection from medical images.

Tool-mode plugin per KTD-V1: the tool body holds the confirm modal +
the fallback flow + the result transform inside a single pydantic-ai
tool call. The LLM is not invoked between turns; it sees one structured
result and composes the final reply guided by ``vision_final_reply.yaml``.

Defense-in-depth gates run *inside* the tool body, in order, before any
HTTP call hits the vision-server:

1. ``ocr_has_report=true`` → :class:`OcrOverrideResult` (KTD-V6).
   Skip inference; the LLM reads the clinician's reading from the
   ``<image>`` tag's OCR text.
2. ``is_medical=false`` → :class:`NotMedicalResult` (R6). The image
   isn't a clinical scan; refuse without reaching the server.
3. ``modality != model.accepted_modality`` → :class:`ModalityMismatchResult`
   (KTD-V3 hard gate). Zero traffic to the server even though the
   server's own ``modality_mismatch`` envelope would also catch this.

Only after all three gates do we ask the user for confirm, route via
:class:`~claritymed.core.vision.registry.VisionRegistry`, and run the
fallback flow within ``tool.total_budget_ms``.

``post_process`` implements
:class:`~claritymed.core.features.base.PostProcessHook` as an audit-only
specialist-keyword check (KTD-V1): when ``clinical_action`` is in
``_AUDIT_ACTIONS``, missing a specialist phrase from
``configs/i18n/<lang>/vision.yaml::vision.specialist_keywords.<action>``
fires ``vision.specialist_keywords.missing``. Audit only — no mutation,
no retry.

This file mirrors ``symptoms_plugin.py`` structurally on purpose;
reviewers comparing the two see the same lifecycle hooks, the same
audit shape, and the same dynamic ``system_prompt_fn`` for reply-
composition guidance.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import TYPE_CHECKING, Any, Callable

import httpx

from claritymed.context import get_context_or_raise
from claritymed.core.features.base import FeatureMode, TurnContext
from claritymed.core.i18n.loader import t, t_list
from claritymed.core.interaction.prompt_channel import (
    InteractiveChannelUnavailable,
    UserDeclinedAnswer,
)
from claritymed.core.interaction.schemas import (
    AskUserQuestionInput,
    Question,
    QuestionOption,
)
from claritymed.core.observability.audit import audit_event
from claritymed.core.observability.audit_payloads import write_payload
from claritymed.core.prompts.registry import PromptRegistry
from claritymed.core.vision.registry import VisionRegistry
from claritymed.core.vision.result import to_llm_payload
from claritymed.core.vision.schemas import (
    ClinicalAction,
    DiseaseSpec,
    ModalityMismatchResult,
    ModelSpec,
    NotMedicalResult,
    OcrOverrideResult,
    RawDetection,
    ServerSpec,
    UserDeclinedResult,
    VisionConfig,
)
from claritymed.core.vision.wire import DetectOptions
from claritymed.errors import (
    ImageHashMismatchError,
    UnknownDiseaseError,
    VisionServerUnreachableError,
)
from claritymed.stores.blob_store import BlobStore
from claritymed.stores.session_attachments import SessionAttachments

# Forward-ref-friendly imports for pydantic-ai's type inspection
# (mirrors symptoms_plugin's pattern — these names must exist in
# globals so ``get_type_hints`` resolves the string annotation).
from pydantic_ai import RunContext

from claritymed.core.turn_state import TurnState

if TYPE_CHECKING:
    from pydantic_ai.toolsets import AbstractToolset

logger = logging.getLogger(__name__)

TOOL_NAME = "detect_disease_from_image"
PROMPT_NAMES = (
    "detect_disease_from_image_tool",
    "vision_final_reply",
    "parse_radiology_report",
)

# Audit-tier allow list — only clinical actions whose reply text MUST
# carry specialist-referral language. ``routine_followup`` /
# ``no_action`` / ``inconclusive_review`` have softer reply playbooks
# (KTD-V1) so a missing "see a specialist" phrase there is not a
# safety signal worth firing.
_AUDIT_ACTIONS: tuple[ClinicalAction, ...] = ("urgent_specialist", "soon_specialist")

# Window of the LLM reply scanned for specialist keywords. Mirrors the
# symptoms safety_keywords audit shape: the reply must lead with the
# referral, so checking past 600 chars would let "rest" later in the
# message satisfy an urgent_specialist audit.
_AUDIT_LEADING_CHARS = 600


# --- validators ------------------------------------------------------------


def _validate_vision_prompts(registry: PromptRegistry) -> None:
    """Fail-loud check: every Unit 7/8 prompt loads bilingually.

    Mirrors ``_validate_symptoms_prompts`` — a missing YAML at first
    call would silently degrade the LLM to no description. The plugin
    constructor is the right place to catch it; the registry is already
    in memory.
    """
    missing: list[str] = []
    for prompt_name in PROMPT_NAMES:
        for lang in ("en", "zh"):
            try:
                registry.get(prompt_name, language=lang)  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001
                missing.append(f"{prompt_name}.{lang}")
    if missing:
        raise RuntimeError(
            "Missing vision prompts: "
            + ", ".join(missing)
            + ". Add the YAML(s) under core/prompts/store/ before "
            "wiring the vision feature."
        )


def _validate_specialist_keywords() -> None:
    """Fail-loud check: every audited clinical_action has both EN + ZH
    specialist phrases.

    A bilingual omission silently disables the
    ``vision.specialist_keywords.missing`` audit signal — we'd be
    blind to the LLM dropping the referral copy. Mirrors the symptoms
    safety-keywords validator.
    """
    missing: list[str] = []
    for action in _AUDIT_ACTIONS:
        key = f"vision.specialist_keywords.{action}"
        for lang in ("en", "zh"):
            entries = t_list(key, lang=lang)
            if not entries:
                missing.append(f"{key}.{lang}")
                continue
            if any(not item.strip() for item in entries):
                raise RuntimeError(
                    f"i18n key {key!r} ({lang}) contains a blank entry; "
                    f"fill or delete it in configs/i18n/{lang}/vision.yaml"
                )
    if missing:
        raise RuntimeError(
            "Missing vision specialist_keywords i18n entries: "
            + ", ".join(missing)
            + ". Define them in configs/i18n/<lang>/vision.yaml under "
            "vision.specialist_keywords.<action>."
        )


# --- modal builders --------------------------------------------------------


def _confirm_question(disease_id: str, language: str) -> Question:
    """Build the confirm modal from i18n keys.

    Copy lives in ``configs/i18n/<lang>/vision.yaml`` under
    ``vision.confirm.<disease_id>.*``. The plugin renders the
    question text and yes/no labels in the user's language so a
    Chinese-locale user is asked in Chinese.
    """
    prefix = f"vision.confirm.{disease_id}"
    return Question(
        question=t(f"{prefix}.body", lang=language),
        header=t(f"{prefix}.title", lang=language),
        options=[
            QuestionOption(
                label=t(f"{prefix}.yes_label", lang=language),
                description=t(f"{prefix}.yes_description", lang=language),
            ),
            QuestionOption(
                label=t(f"{prefix}.no_label", lang=language),
                description=t(f"{prefix}.no_description", lang=language),
            ),
        ],
    )


# --- tracing baggage (KTD-V2) ----------------------------------------------


def _attach_vision_baggage(
    disease_id: str, model_id: str, server_id: str
) -> object | None:
    """Attach the three vision baggage keys onto the active OTel context.

    Returns the attach token; ``None`` if OpenTelemetry is unavailable
    (mirrors ``_attach_symptoms_baggage``).
    """
    try:
        from opentelemetry import baggage
        from opentelemetry import context as otel_context
        from opentelemetry.trace import get_current_span

        ctx = otel_context.get_current()
        ctx = baggage.set_baggage(
            "claritymed.vision.disease_id", disease_id, context=ctx
        )
        ctx = baggage.set_baggage("claritymed.vision.model_id", model_id, context=ctx)
        ctx = baggage.set_baggage("claritymed.vision.server_id", server_id, context=ctx)
        span = get_current_span()
        if span.get_span_context() is not None and span.get_span_context().is_valid:
            span.set_attribute("claritymed.vision.disease_id", disease_id)
            span.set_attribute("claritymed.vision.model_id", model_id)
            span.set_attribute("claritymed.vision.server_id", server_id)
        return otel_context.attach(ctx)
    except ImportError:
        return None
    except Exception:  # noqa: BLE001
        logger.warning("vision baggage attach failed", exc_info=True)
        return None


def _detach_baggage(token: object | None) -> None:
    if token is None:
        return
    try:
        from opentelemetry import context as otel_context

        otel_context.detach(token)
    except Exception:  # noqa: BLE001
        pass


# --- plugin ----------------------------------------------------------------


class VisionFeature:
    """Tool-mode plugin exposing ``detect_disease_from_image``.

    Constructor stores collaborators only. Validators run eagerly so a
    missing YAML or i18n key fails the plugin build rather than the
    first request.
    """

    name = "vision"
    mode: FeatureMode = "tool"

    def __init__(
        self,
        *,
        config: VisionConfig,
        registry: VisionRegistry,
        get_session_id: Callable[[], str | None],
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        self._config = config
        self._registry = registry
        self._get_session_id = get_session_id
        self._prompt_registry = prompt_registry or PromptRegistry()
        # Cache the model_id → ModelSpec lookup. Built once at construction
        # time so per-tool-call paths (``_run_fallback_flow``,
        # ``_covered_diseases_block``) don't rebuild a dict every invocation.
        self._models_by_id: dict[str, Any] = {m.id: m for m in self._config.models}
        _validate_vision_prompts(self._prompt_registry)
        _validate_specialist_keywords()
        # Per-request post_process state — keyed by request_id. Cleared
        # after post_process consumes it.
        self._stash: dict[str, dict[str, Any]] = {}
        # Public health state — populated by ``ensure_bootstrapped`` (or
        # the legacy ``AskService._bootstrap_vision_once`` path). Stays
        # ``None`` while everything is healthy; on bootstrap failure
        # carries a short human-readable reason that the TUI surfaces as
        # a startup toast + status indicator and the AttachmentsFeature
        # bakes into the ``<image>`` tag so the LLM stops trying to call
        # the missing tool. Read by external surfaces, written here only.
        self.disabled_reason: str | None = None

    async def pre_invoke(self, ctx: TurnContext) -> str:
        """No-op pre-invoke; vision feature does not preamble-inject."""
        return ""

    async def ensure_bootstrapped(self) -> str | None:
        """Run the registry's catalog cross-check, capturing failure as state.

        Idempotent — second call is a no-op when bootstrap already
        succeeded. On failure populates :attr:`disabled_reason` with a
        short string the TUI can surface verbatim, and returns the same
        string so the caller can branch without a second attribute read.
        Returns ``None`` on success.

        Two callers funnel through here:

        * TUI ``on_mount`` — wants to surface the failure as a startup
          toast + status indicator so the user knows before pasting a
          medical image that the vision tool is offline.
        * :meth:`AskService._bootstrap_vision_once` — legacy first-turn
          deferred path, kept for code that does not run a TUI mount.

        Both pre-existing failure modes (network unreachable, manifest
        sha drift) collapse into the same ``disabled_reason`` string so
        downstream surfaces (toast, ``<image vision_disabled="…">`` tag)
        do not need to branch on exception type.
        """
        if self.disabled_reason is not None:
            return self.disabled_reason
        try:
            await self._registry.bootstrap()
        except Exception as exc:  # noqa: BLE001 — registry raises many shapes
            reason = f"{type(exc).__name__}: {exc!s}"
            self.disabled_reason = reason
            logger.error(
                "vision: registry bootstrap failed; disabling vision tool "
                "for this session (%s)",
                reason,
                exc_info=True,
            )
            return reason
        return None

    def system_prompt_fn(self) -> "Callable":
        """Return a dynamic system-prompt function for pydantic-ai.

        Returns the ``vision_final_reply`` prompt only after the tool
        has set ``deps.vision_reply_guide`` (i.e. on the reply-
        composition call, not the tool-selection call). Empty string
        when the tool was not called or returned a short-circuit kind
        for which the reply prompt does not apply.
        """

        def _fn(ctx: "RunContext[Any]") -> str:
            return getattr(ctx.deps, "vision_reply_guide", None) or ""

        return _fn

    def as_toolset(self) -> "AbstractToolset[Any] | None":
        """Return None; this feature exposes a single tool, not a toolset."""
        return None

    def as_tool(self) -> Callable | None:
        """Return a pydantic-ai Tool wrapping ``_detect``."""
        from pydantic_ai import Tool

        return Tool(
            self._detect,
            name=TOOL_NAME,
            description=self._build_tool_description(),
        )

    def _build_tool_description(self) -> str | None:
        """Build the tool description, injecting ``{covered_diseases}``.

        Each enabled disease contributes one line:
        ``- <disease_id> (<modality>): <intent one-liner>`` so the LLM
        has enough signal to pick the right ``disease_id`` without an
        unwieldy enumeration. Falls back gracefully when the prompt
        YAML is absent (constructor would have raised, but defending
        in depth keeps the tool registration tolerant of missing copy
        in dev).
        """
        try:
            template = self._prompt_registry.get(
                "detect_disease_from_image_tool", language="en"
            )
        except Exception:  # noqa: BLE001
            return None
        if "{covered_diseases}" not in template:
            return template
        lines: list[str] = []
        for disease in self._registry.diseases.values():
            if not disease.enabled:
                continue
            modality = self._models_by_id[disease.primary_model_id].accepted_modality
            intent = t(disease.intent_hints_i18n_key, lang="en")
            # ``t`` returns the bare key on miss; we want the intent
            # surface to either be the localized one-liner or an empty
            # placeholder rather than echoing a dotted key into the LLM
            # prompt.
            if intent == disease.intent_hints_i18n_key:
                intent = "(intent description missing)"
            lines.append(f"- {disease.id} ({modality}): {intent.strip()}")
        covered = (
            "\n".join(lines)
            if lines
            else "  (no diseases enabled — vision tool is a no-op this run)"
        )
        return template.replace("{covered_diseases}", covered)

    # --- tool body --------------------------------------------------------

    async def _detect(
        self,
        ctx: "RunContext[TurnState]",
        disease_id: str,
        image_sha: str,
        model_id_hint: str | None = None,
    ) -> dict[str, Any]:
        """Run the gate cascade → confirm → fallback flow.

        Always returns a dict (never raises into the LLM). Error
        translation happens here so the LLM sees a structured result
        for every branch.
        """
        deps = ctx.deps
        language = getattr(deps, "language", "en") or "en"
        request_id, user_id, _ = get_context_or_raise()
        audit_event(
            "tool.detect_disease_from_image",
            {
                "tool_name": TOOL_NAME,
                "disease_id": disease_id,
                "model_id_hint": model_id_hint,
                "image_sha_prefix": image_sha[:8] if image_sha else None,
            },
        )

        # Step 1: resolve the attachment from the active session. A
        # hallucinated sha (no matching attachment) fails loudly so the
        # LLM gets a structured error rather than the body silently
        # proceeding with an empty payload.
        attachment = self._resolve_attachment(user_id, image_sha)
        if attachment is None:
            return {
                "kind": "missing_attachment",
                "image_sha": image_sha,
                "message": (
                    "no session attachment matches that sha — ask the user to "
                    "re-paste the image, then retry"
                ),
            }
        meta = self._read_vision_meta(user_id, attachment.sha256)

        # Step 2: OCR override (KTD-V6). Tool body short-circuits even
        # when the LLM ignored the tool description's rule 2; the body
        # is the load-bearing enforcement. Shadow inference (KTD-V9) is
        # opt-in via config; default OFF.
        if meta.get("ocr_has_report") is True:
            audit_event(
                "vision_skipped_ocr_override",
                {
                    "disease_id": disease_id,
                    "image_sha_prefix": image_sha[:8],
                },
            )
            if self._config.tool.shadow_inference_on_report_override:
                # Opt-in path — write the marker into the audit payload so
                # an offline pipeline can pair the OCR text with the model
                # output post hoc. We do NOT fire inference in v1 (the
                # offline pipeline is a separate Unit X follow-up) but
                # the payload entry is the contract.
                write_payload(
                    user_id,
                    request_id,
                    {
                        "kind": "vision.shadow_inference",
                        "disease_id": disease_id,
                        "image_sha": attachment.sha256,
                        "reason": "ocr_override",
                        "note": (
                            "shadow inference flagged; offline analysis pipeline "
                            "is a separate task — payload is the breadcrumb only"
                        ),
                    },
                )
            return OcrOverrideResult(
                message=(
                    "image already carries a clinician report; the LLM should "
                    "answer from the OCR text instead of running the model"
                )
            ).model_dump()

        # Step 3: not-medical refuse (R6). is_medical is a tri-state —
        # only False blocks the tool. None (no medical-clip server when
        # the image was ingested) falls through so a missing classifier
        # doesn't permanently shut the feature off.
        if meta.get("is_medical") is False:
            return NotMedicalResult(
                message=(
                    "image is not a medical scan; answer the user's question "
                    "without this tool"
                )
            ).model_dump()

        # Step 4: route via the registry. ``UnknownDiseaseError``
        # surfaces back to the LLM as a structured dict so the model can
        # re-emit the disambig modal with the correct list.
        try:
            server, model = self._registry.route(disease_id, model_id_hint)
        except UnknownDiseaseError as exc:
            return {
                "kind": "unknown_disease",
                "disease_id": disease_id,
                "available": list(exc.available),
                "message": (
                    f"disease {disease_id!r} is not enabled. Re-emit the "
                    f"disambig modal with the available list."
                ),
            }

        # Step 5: modality hard gate (KTD-V3). Defense-in-depth: the
        # server's 422 envelope would also catch this, but bouncing the
        # request before the HTTP round-trip saves a second of
        # wall-clock when the catalog grows.
        image_modality = meta.get("modality")
        if (
            image_modality
            and image_modality != "unknown"
            and image_modality != model.accepted_modality
        ):
            return ModalityMismatchResult(
                model_accepts=model.accepted_modality,
                image_modality=image_modality,
                message=(
                    f"model {model.id!r} accepts modality "
                    f"{model.accepted_modality!r} but the image is tagged "
                    f"{image_modality!r}"
                ),
            ).model_dump()

        # Step 6: confirm modal. Skip when the config flips off the
        # confirm gate (headless e2e flips this) so the bench / e2e
        # paths don't need a fake answer channel for every test.
        if self._config.tool.confirm_before_run:
            channel = getattr(deps, "prompt_channel", None)
            if channel is None:
                audit_event(
                    "vision_detection_event",
                    {
                        "disease_id": disease_id,
                        "phase": "confirm",
                        "reason": "no_interactive_channel",
                    },
                )
                return UserDeclinedResult(
                    message=(
                        "no interactive channel available to confirm running the "
                        "model; tool body refused without confirmation"
                    )
                ).model_dump()
            confirmed = await self._confirm(channel, disease_id, language)
            if not confirmed:
                audit_event(
                    "vision_detection_event",
                    {
                        "disease_id": disease_id,
                        "phase": "confirm",
                        "user_declined": True,
                    },
                )
                return UserDeclinedResult(
                    message="user declined to run the model on the confirm modal"
                ).model_dump()

        # Step 7: fallback flow within total_budget_ms.
        disease = self._registry.diseases[disease_id]
        try:
            raw, attempted = await self._run_fallback_flow(
                disease=disease,
                primary_model=model,
                primary_server=server,
                attachment_user_id=user_id,
                attachment_sha=attachment.sha256,
                language=language,
                request_id=request_id,
            )
        except _NoUsableResult as exc:
            audit_event(
                "vision_detection_event",
                {
                    "disease_id": disease_id,
                    "phase": "fallback",
                    "outcome": "no_usable_result",
                    "fallback_count": exc.attempted,
                },
            )
            return {
                "kind": "no_usable_result",
                "disease_id": disease_id,
                "fallback_count": exc.attempted,
                "warnings": list(exc.warnings),
                "message": (
                    "every model in the fallback flow returned low confidence "
                    "or was unreachable within the budget; recommend clinician "
                    "review"
                ),
            }
        except VisionServerUnreachableError:
            audit_event(
                "vision_detection_event",
                {
                    "disease_id": disease_id,
                    "phase": "fallback",
                    "outcome": "server_unreachable",
                },
            )
            return {
                "kind": "no_usable_result",
                "disease_id": disease_id,
                "fallback_count": 0,
                "warnings": ["vision_server_unreachable"],
                "message": (
                    "the vision server is unreachable; tell the user the model "
                    "could not run and recommend clinician review"
                ),
            }

        # Step 8: build the LLM-facing payload + stash for post_process.
        # Wrap in try/except so a payload-shaping bug never raises into the
        # LLM tool loop — the tool docstring guarantees a structured dict.
        try:
            payload = to_llm_payload(
                raw,
                top_k=self._config.tool.top_k,
                language=language,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "to_llm_payload failed for disease_id=%s model_id=%s",
                disease_id,
                raw.model_id,
            )
            return {
                "disease_id": disease_id,
                "fallback_count": attempted,
                "warnings": ["payload_build_failed"],
                "message": (
                    "vision model returned a result but the payload could "
                    "not be shaped; tell the user the analysis is "
                    f"inconclusive ({type(exc).__name__})"
                ),
            }
        baggage_token = _attach_vision_baggage(disease_id, raw.model_id, server.id)
        try:
            self._emit_detection_event_and_payload(
                user_id=user_id,
                request_id=request_id,
                server=server,
                raw=raw,
                fallback_count=attempted,
            )
        finally:
            _detach_baggage(baggage_token)
        self._stash[request_id] = {
            "clinical_action": raw.clinical_action,
            "disease_id": disease_id,
        }
        # Wire the reply-composition guide for the dynamic system prompt.
        try:
            guide = self._prompt_registry.get("vision_final_reply", language=language)
            ctx.deps.vision_reply_guide = guide  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass
        return payload.model_dump()

    # --- helpers ----------------------------------------------------------

    def _resolve_attachment(self, user_id: str, image_sha: str):
        """Look up the session attachment by sha; return None on miss.

        The LLM is expected to copy the ``sha=...`` attribute verbatim
        from the ``<image>`` tag. Truncated prefixes are rejected here
        (we require an exact match) so a sloppy LLM gets a structured
        error rather than a fuzzy lookup.
        """
        session_id = self._get_session_id()
        if session_id is None or not image_sha:
            return None
        try:
            attachments = SessionAttachments(user_id, session_id)
            return attachments.get(image_sha)
        except Exception:  # noqa: BLE001
            logger.warning(
                "vision: failed to read session attachments for sha=%s",
                image_sha[:8],
                exc_info=True,
            )
            return None

    def _read_vision_meta(self, user_id: str, sha: str) -> dict[str, Any]:
        """Read ``ocr.json`` for ``sha`` and return the vision-tag fields.

        Returns an empty dict when the sentinel is absent (pre-OCR) or
        when the worker omitted the vision fields (server unavailable
        at ingest time). Absence flows downstream as "no tag" — the
        modality hard gate skips, ``is_medical`` is treated as None,
        ``ocr_has_report`` defaults to False.
        """
        try:
            meta = BlobStore(user_id).read_ocr_metadata(sha)
        except Exception:  # noqa: BLE001
            logger.warning(
                "vision: failed to read ocr.json for sha=%s", sha[:8], exc_info=True
            )
            return {}
        return meta or {}

    async def _confirm(self, channel: Any, disease_id: str, language: str) -> bool:
        payload = AskUserQuestionInput(
            questions=[_confirm_question(disease_id, language)]
        )
        try:
            result = await channel.ask(payload)
        except UserDeclinedAnswer:
            return False
        except InteractiveChannelUnavailable:
            return False
        text = next(iter(result.answers.values()), "")
        if isinstance(text, list):
            text = text[0] if text else ""
        s = str(text).strip().lower()
        # Match against the localized yes-label as well as the
        # canonical "yes" / "是" so a translator can change the label
        # without breaking the comparison.
        yes_label = t(f"vision.confirm.{disease_id}.yes_label", lang=language).lower()
        return s in ("yes", "是", "y") or s == yes_label

    async def _run_fallback_flow(
        self,
        *,
        disease: DiseaseSpec,
        primary_model: ModelSpec,
        primary_server: ServerSpec,
        attachment_user_id: str,
        attachment_sha: str,
        language: str,
        request_id: str,
    ) -> tuple[RawDetection, int]:
        """Walk ``disease.effective_flow`` until a usable result.

        ``effective_flow`` is ``[primary_model_id, *flow]`` — the
        primary always runs first, then any declared fallbacks. Accepts
        the result when ``confidence_tier`` is ``"medium"`` or
        ``"high"``. ``"low"`` falls through to the next model; the
        server has already overridden ``clinical_action`` to
        ``inconclusive_review`` per KTD-V10, so even when we exhaust
        the chain the last attempt carries safe LLM-facing copy.

        Raises:
            _NoUsableResult: every model in the chain returned low-conf
                or couldn't be reached within ``total_budget_ms``.
            VisionServerUnreachableError: the *very first* model is
                unreachable and there are no fallbacks to attempt.
                Bubble up so the tool body's outer ``except`` can branch
                cleanly.
        """
        try:
            image_bytes, wire_sha = _read_blob_bytes(attachment_user_id, attachment_sha)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "vision: failed to read blob bytes for sha=%s: %s",
                attachment_sha[:8],
                exc,
            )
            raise _NoUsableResult(0, [f"blob_read_failed: {exc!s}"]) from exc

        budget_ms = self._config.tool.total_budget_ms
        safety = self._config.tool.fallback_safety_factor
        deadline = time.monotonic() + budget_ms / 1000.0
        warnings: list[str] = []
        attempted = 0
        last_low: RawDetection | None = None

        chain = disease.effective_flow
        for model_id in chain:
            spec = self._models_by_id.get(model_id)
            if spec is None:
                continue
            remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
            # Skip flow steps the per-model expected_ms can't fit in the
            # remaining budget (with a safety factor) — better to return
            # whatever we have than start a call we'll abort halfway.
            if attempted > 0 and remaining_ms < spec.expected_ms * safety:
                warnings.append(
                    f"skipping {spec.id}: remaining {remaining_ms}ms < "
                    f"expected {spec.expected_ms}ms * safety {safety}"
                )
                continue
            server = self._registry.servers[spec.server_id]
            client = self._registry.client_for(server)
            attempted += 1
            try:
                response = await client.detect(
                    request_id=request_id,
                    disease_id=disease.id,
                    model_id=spec.id,
                    image_bytes=image_bytes,
                    language=language,
                    options=DetectOptions(),
                    sha256=wire_sha,
                )
            except VisionServerUnreachableError as exc:
                warnings.append(f"{spec.id}: server_unreachable: {exc!s}")
                # Primary-and-only fallthrough — no fallbacks declared.
                # Re-raise so the tool body emits a dedicated message
                # ("server unreachable") instead of "every model came
                # back low".
                if attempted == 1 and spec is primary_model and len(chain) == 1:
                    raise
                continue
            except httpx.HTTPStatusError as exc:
                code = "http_error"
                try:
                    code = exc.response.json().get("error", {}).get("code", code)
                except Exception:  # noqa: BLE001
                    pass
                warnings.append(f"{spec.id}: {code} ({exc.response.status_code})")
                continue
            except ImageHashMismatchError as exc:
                warnings.append(f"{spec.id}: image_hash_mismatch ({exc!s})")
                continue

            raw = _wire_to_raw(response)
            if raw.classification.confidence_tier in ("medium", "high"):
                return raw, attempted
            warnings.append(
                f"{spec.id}: confidence_tier=low — falling through to next model"
            )
            last_low = raw

        if last_low is not None:
            # Budget exhausted with only low-conf results in hand. The
            # server already overrode clinical_action to
            # inconclusive_review so returning the last one is safe; the
            # LLM-facing payload carries the warnings list so the reply
            # can quote them.
            last_low.warnings.extend(warnings)
            return last_low, attempted
        raise _NoUsableResult(attempted, warnings)

    def _emit_detection_event_and_payload(
        self,
        *,
        user_id: str,
        request_id: str,
        server: ServerSpec,
        raw: RawDetection,
        fallback_count: int,
    ) -> None:
        """Emit the vision_detection_event audit + write the PHI payload."""
        audit_event(
            "vision_detection_event",
            {
                "disease_id": raw.disease_id,
                "model_id": raw.model_id,
                "server_id": server.id,
                "elapsed_ms": raw.elapsed_ms,
                "top1": raw.classification.top1,
                "top1_prob": round(float(raw.classification.top1_prob), 3),
                "confidence_tier": raw.classification.confidence_tier,
                "cancer_status": raw.cancer_status,
                "clinical_action": raw.clinical_action,
                "quality_gate_passed": raw.input_quality.passed,
                "fallback_count": fallback_count,
                "ocr_override_fired": False,
            },
        )
        write_payload(
            user_id,
            request_id,
            {
                "kind": "vision.detection",
                "disease_id": raw.disease_id,
                "model_id": raw.model_id,
                "model_version": raw.model_version,
                "server_id": server.id,
                "top1": raw.classification.top1,
                "top1_prob": float(raw.classification.top1_prob),
                "probabilities": [float(p) for p in raw.classification.probabilities],
                "labels": list(raw.classification.labels),
                "confidence_tier": raw.classification.confidence_tier,
                "cancer_status": raw.cancer_status,
                "clinical_action": raw.clinical_action,
                "quality_gate_passed": raw.input_quality.passed,
                "warnings": list(raw.warnings),
            },
        )

    # --- post_process (PostProcessHook) -----------------------------------

    async def post_process(self, text: str, tool_result: dict[str, Any]) -> str:
        """Audit-only specialist-keyword check (KTD-V1).

        Always returns ``text`` unchanged. Emits
        ``vision.specialist_keywords.missing`` when an
        ``urgent_specialist`` / ``soon_specialist`` reply does not
        contain an action-appropriate phrase. Keyword lists live in
        ``configs/i18n/<lang>/vision.yaml::vision.specialist_keywords.
        <action>`` and are read via :func:`t_list` so translators own
        them.
        """
        try:
            request_id, _, language = get_context_or_raise()
        except Exception:  # noqa: BLE001
            return text
        stash = self._stash.pop(request_id, None)
        if stash is None:
            return text
        action = stash.get("clinical_action")
        if action not in _AUDIT_ACTIONS:
            return text
        keywords = t_list(f"vision.specialist_keywords.{action}", lang=language)
        leading = text[:_AUDIT_LEADING_CHARS].lower()
        if any(kw.lower() in leading for kw in keywords):
            audit_event(
                "vision_detection_event",
                {
                    "disease_id": stash.get("disease_id"),
                    "phase": "post_process",
                    "clinical_action": action,
                    "specialist_keyword_match": True,
                },
            )
            return text
        audit_event(
            "vision.specialist_keywords.missing",
            {
                "disease_id": stash.get("disease_id"),
                "clinical_action": action,
                "observed_leading_chars": min(_AUDIT_LEADING_CHARS, len(text)),
            },
        )
        return text


# --- module helpers --------------------------------------------------------


def _read_blob_bytes(user_id: str, sha256: str) -> tuple[bytes, str]:
    """Resolve vision-decodable bytes + their wire sha for one blob.

    Returns ``(bytes, sha)`` where ``sha`` is the digest the vision
    server should cross-check on the wire. Two cases:

    * ``vision.png`` sidecar exists — written by the OCR worker's
      1-page image-PDF rasterizer. Returns the PNG bytes and their
      sha (the source PDF's sha, which is what the caller has, would
      mismatch and the server would reject as ``image_hash_mismatch``).
    * No sidecar — falls back to the original ``content.<ext>`` and
      reuses the caller's ``sha256`` since it equals the bytes' digest
      by construction.

    The blob dir contains exactly one ``content.*`` file (other entries
    are sidecars: ``ocr.md``, ``ocr.json``, ``vision.png``). We pick
    the first content file that isn't a ``.tmp`` partial — mirrors the
    TUI's blob-loading shape.
    """
    blob_dir = BlobStore(user_id).dir(sha256)
    if not blob_dir.exists():
        raise FileNotFoundError(f"blob directory missing: {blob_dir}")
    sidecar = blob_dir / "vision.png"
    if sidecar.exists():
        png_bytes = sidecar.read_bytes()
        return png_bytes, hashlib.sha256(png_bytes).hexdigest()
    candidates = [
        p
        for p in blob_dir.iterdir()
        if p.name.startswith("content.") and not p.name.endswith(".tmp")
    ]
    if not candidates:
        raise FileNotFoundError(f"no content.* file in blob dir: {blob_dir}")
    return candidates[0].read_bytes(), sha256


def _wire_to_raw(response) -> RawDetection:
    """Project ``servers/vision/wire.py::DetectResponse`` into the core
    ``RawDetection`` model.

    Wire schemas and core schemas have separate import lineages so
    server-internal fields can evolve without churning the client
    surface. ``RawDetection`` is a structural subset; the dump-and-
    validate path is the simplest round-trip.
    """
    return RawDetection.model_validate(response.model_dump())


class _NoUsableResult(RuntimeError):
    """Raised inside :meth:`VisionFeature._run_fallback_flow` when every
    model in flow either failed or returned low-confidence.

    Captures the attempt count + the accumulated warnings so the tool
    body can hand both back to the LLM in one structured payload.
    """

    def __init__(self, attempted: int, warnings: list[str]) -> None:
        super().__init__("no usable result in fallback flow")
        self.attempted = attempted
        self.warnings = warnings


# --- production factory ----------------------------------------------------


def make_vision_factory(
    *,
    get_session_id: Callable[[], str | None],
) -> "Callable[[], VisionFeature] | None":
    """Build the :class:`VisionFeature` factory for AskService.

    Returns ``None`` when:

    * ``configs/vision.yaml`` is absent or malformed.
    * All diseases have ``enabled: false`` (kill switch).
    * The registry bootstrap raises (vision-server unreachable, catalog
      drift, etc.). The orchestrator prefers a silently-disabled vision
      tool over a hard boot failure so the rest of the app stays usable
      while an operator restarts the server.
    """
    import asyncio

    from claritymed.config import load_vision_config

    try:
        config = load_vision_config()
    except Exception:  # noqa: BLE001
        logger.warning("vision: config load failed; feature disabled", exc_info=True)
        return None

    enabled = [d for d in config.diseases if d.enabled]
    if not enabled:
        logger.info("vision: no enabled diseases; feature disabled")
        return None

    registry = VisionRegistry(config)
    try:
        asyncio.run(registry.bootstrap())
    except RuntimeError as exc:
        # asyncio.run raises when a loop is already running — fall back
        # to deferring bootstrap to AskService startup. The factory
        # contract is "build the plugin if possible"; a running loop is
        # the caller's job to handle.
        if "asyncio.run" in str(exc):
            logger.info(
                "vision: bootstrap deferred to AskService first turn "
                "(caller has a running loop)"
            )
        else:
            logger.warning("vision: registry bootstrap failed; feature disabled")
            return None
    except Exception:  # noqa: BLE001
        logger.warning(
            "vision: registry bootstrap failed; feature disabled", exc_info=True
        )
        return None

    def _factory() -> VisionFeature:
        return VisionFeature(
            config=config,
            registry=registry,
            get_session_id=get_session_id,
        )

    return _factory


__all__ = [
    "TOOL_NAME",
    "VisionFeature",
    "make_vision_factory",
]
