"""Extractor LLM — pulls :class:`ExtractedSymptoms` from chat history.

The extractor is the only LLM in the gate pipeline that touches the
raw user history (the composer sees structured findings, never the
patient's free text). Per KTD-E1 + the project's PHI guard contract,
this LLM must therefore be local (``ProviderConfig.kind == "local"``)
— callers wire it with a local model and ``build_default_extractor``
fails loud if no local provider is available.

Local-only is the floor, not a recommendation. Sending the raw
conversation history to a cloud provider for symptom extraction would
defeat the entire ``phi_guard`` posture the project is built around;
the safety net's *own* implementation cannot be the privacy leak.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

from claritymed.core.emergency.schemas import ExtractedSymptoms
from claritymed.core.prompts.registry import PromptRegistry

if TYPE_CHECKING:
    from pydantic_ai.models import Model

logger = logging.getLogger(__name__)

_EXTRACTOR_PROMPT_NAME = "emergency_extractor"


class Extractor(Protocol):
    """Extractor interface — implementations may be LLM or stubs.

    The Protocol shape lets tests inject a deterministic fake while
    production wires the LLM-backed implementation.
    """

    async def extract(
        self,
        query: str,
        history: list[Any] | None,
    ) -> ExtractedSymptoms:
        """Return a structured symptom snapshot for the latest turn."""
        ...


class LLMExtractor:
    """pydantic-ai-backed extractor reading ``emergency_extractor.yaml``.

    Construction is cheap (registry + model handle); the LLM call only
    fires inside :meth:`extract`. Operators who want a different local
    model pass it via ``model``; tests can either inject this class
    with a stub model or implement :class:`Extractor` directly.

    pydantic-ai's structured-output path (``output_type=ExtractedSymptoms``)
    handles the JSON schema, validation, and one-shot retry for us —
    we do not hand-roll parse/validate logic. See the project rule
    "Use pydantic-ai's built-ins before writing your own".
    """

    def __init__(
        self,
        model: "Model",
        registry: PromptRegistry | None = None,
        *,
        language: str = "en",
    ) -> None:
        self._model = model
        self._registry = registry or PromptRegistry()
        # The extractor's *system prompt* language is independent of
        # the user-facing reply language — see CLAUDE.md's
        # ``CLARITYMED_TOOL_PROMPT_LANG`` note. Default English here
        # because the canonical-token vocabulary is English regardless
        # of the user's chat language; the prompt itself instructs the
        # LLM to read mixed-language history and emit English tokens.
        self._language = language

    async def extract(
        self,
        query: str,
        history: list[Any] | None,
    ) -> ExtractedSymptoms:
        """Run the extractor over ``query`` + ``history``.

        Failures are surfaced to the caller; :class:`EmergencyTriage`
        catches and falls open to ``routine_noop`` so a single
        extractor downtime cannot deny the user their answer (plan
        §"Open Questions" → fail-open default).
        """
        from pydantic_ai import Agent

        system_prompt = self._registry.get(
            _EXTRACTOR_PROMPT_NAME,
            language=self._language,  # type: ignore[arg-type]
        )
        agent: Agent[None, ExtractedSymptoms] = Agent(
            self._model,
            output_type=ExtractedSymptoms,
            system_prompt=system_prompt,
        )
        user_text = _format_history_for_extractor(query, history)
        result = await agent.run(user_text)
        return result.output


def _format_history_for_extractor(
    query: str,
    history: list[Any] | None,
) -> str:
    """Render ``history`` + ``query`` into the extractor's user message.

    pydantic-ai's ``message_history`` parameter would also let the
    model see prior turns, but it carries the full ModelMessage shape
    including tool calls / structured outputs — noise for the
    extractor's narrow task. Flatten to plain text instead so the
    prompt stays small and the same shape works whether the host has
    a chat session wired or not.
    """
    lines: list[str] = []
    if history:
        for msg in history:
            text = _extract_message_text(msg)
            if not text:
                continue
            role = _extract_message_role(msg)
            lines.append(f"{role}: {text}")
    lines.append(f"user: {query}")
    return "\n".join(lines)


def _extract_message_text(msg: Any) -> str:
    """Best-effort extraction of plain text from a pydantic-ai message.

    The shape of ``msg`` varies (``ModelRequest`` / ``ModelResponse``
    with ``parts: list[...Part]``). We only need user-visible text;
    tool calls, tool responses, and system parts are skipped.
    """
    parts = getattr(msg, "parts", None)
    if not parts:
        return ""
    chunks: list[str] = []
    for part in parts:
        content = getattr(part, "content", None)
        if isinstance(content, str) and content:
            chunks.append(content)
    return "\n".join(chunks)


def _extract_message_role(msg: Any) -> str:
    """Map a pydantic-ai message to ``"user"`` / ``"assistant"`` / ``"system"``."""
    cls_name = type(msg).__name__
    if cls_name == "ModelRequest":
        return "user"
    if cls_name == "ModelResponse":
        return "assistant"
    return "system"


def build_default_extractor(
    *,
    registry: PromptRegistry | None = None,
    language: str = "en",
) -> LLMExtractor | None:
    """Construct an :class:`LLMExtractor` against the configured local provider.

    Reads ``configs/emergency.yaml::provider_id`` to decide which
    local provider to bind. When that field is unset, falls back to
    "first kind=local entry in models.yaml" (today's default). Returns
    ``None`` when no usable local provider is found — :class:`EmergencyTriage`
    then stays in noop mode and the host can wire a custom extractor
    explicitly.
    """
    from claritymed.core.emergency._provider import build_local_gate_model
    from claritymed.core.emergency.config import load_emergency_config

    cfg = load_emergency_config()
    model = build_local_gate_model("extractor", prefer_id=cfg.provider_id)
    if model is None:
        return None
    return LLMExtractor(model, registry=registry, language=language)
