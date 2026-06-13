"""``PhiAssertionModel`` — layer 3 of the 3-layer PHI defense.

Wraps a pydantic-ai ``Model`` and intercepts every ``request`` /
``request_stream`` call. Before delegating to the inner model it scans
all messages for PHI content and raises :class:`PhiLeakDetected` (a
``NonRetryableLLMError`` subclass) on detection.

Two scan passes:

* **Content pass** — every ``UserPromptPart`` / ``SystemPromptPart`` /
  ``TextPart`` / ``ToolReturnPart`` content goes through
  ``PhiGuard.scrub_free_text``. Any regex- or model-detected PHI hit
  fires :class:`PhiLeakDetected`. Catches assistant echoes ("your HGB
  is 105") and tool-return ``extracted_labs.value`` numbers.
* **Marker pass** — message parts carrying an explicit ``_phi_safe``
  flag are skipped without a content scan (e.g. system prompts loaded
  from PromptRegistry, the original user prompt that ``AskService``
  already scrubbed). This is the false-positive escape hatch.
* **Trusted tool-return auto-marker** — every ``ToolReturnPart`` whose
  ``tool_name`` is in ``_TOOL_RETURN_PHI_SAFE`` gets ``_phi_safe`` set
  on first scan and is then routed through the marker pass on every
  subsequent step. Used for ingest tools whose return content is
  constructed server-side (status literals like ``{"ok": True}`` and
  internal slug paths like ``papers/2026-06-13-abc23xyz``) and is
  guaranteed not to echo user text. The cloud NER reliably misreads
  the random slug suffix as ``secret`` / ``account_number``; this
  list is how the policy says "we wrote these returns, they are safe
  by construction".

Wired in ``core/llm/model.py:build_model`` when ``provider.kind ==
"cloud"``. Wrap order is ``PhiAssertionModel(LoggingModel(Base))`` so
the assertion fires *before* the inner model logs the message — a
leak detected here never reaches the wire and never gets persisted in
``llm.log`` as a side channel.

Why ``NonRetryableLLMError`` matters: pydantic-ai's retry loop would
otherwise re-issue the same offending prompt up to N times, generating
N audit rows for one underlying leak. Subclassing the marker base
short-circuits the retry handler.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from pydantic_ai.models import Model as _PydanticModel

from claritymed.core.observability.audit import audit_event
from claritymed.core.phi.guard import PhiGuard
from claritymed.errors import PhiLeakDetected

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.settings import ModelSettings

logger = logging.getLogger(__name__)

# Attribute the caller can set on a Message instance to opt out of the
# content scan (e.g. for the originally-scrubbed user prompt). Strict
# attribute check, not a payload key, so the LLM can't request its own
# bypass through structured output.
_PHI_SAFE_ATTR = "_phi_safe"

# Tool returns that are PHI-safe by construction — see module docstring.
# Adding a name here is a promise that the tool's return content is
# server-built (status literal or internal slug/path), never echoes
# user-supplied free text. Args (which may carry user text) flow through
# the audit_payloads sidecar at mode 0600 and never appear in the
# ToolReturnPart content.
_TOOL_RETURN_PHI_SAFE: frozenset[str] = frozenset(
    {
        # Slug-bearing returns: {"record_path": "..."} / {"library_path": ...}
        # / {"deleted": ...}. Slug shape is "<YYYY-MM-DD>-<8-char base32>"
        # which the NER tags as ``secret`` / ``account_number``.
        "save_record",
        "save_to_library",
        "delete_record",
        # Status-literal returns: {"ok": True} or {"ok": False, "reason": ...}.
        # Listed for symmetry — they happen not to trip the NER today but
        # they are equally safe by construction.
        "save_medication",
        "save_allergy",
        "save_condition",
        "update_profile_field",
    }
)


def _is_phi_safe(message: Any) -> bool:
    return bool(getattr(message, _PHI_SAFE_ATTR, False))


def _auto_mark_safe_tool_return(part: Any) -> None:
    """Set ``_phi_safe`` on ingest tool returns that are safe by construction.

    No-op for parts that already carry the marker, are not
    ``ToolReturnPart``, or whose ``tool_name`` is not in the allowlist.
    Setting the attribute means future scans short-circuit at
    ``_is_phi_safe`` without re-checking the allowlist.
    """
    from pydantic_ai.messages import ToolReturnPart

    if _is_phi_safe(part):
        return
    if not isinstance(part, ToolReturnPart):
        return
    if getattr(part, "tool_name", None) in _TOOL_RETURN_PHI_SAFE:
        setattr(part, _PHI_SAFE_ATTR, True)


def _scan_text_for_phi(text: str, guard: PhiGuard) -> bool:
    """Return True iff the scrubber reports at least one PHI hit."""
    if not text:
        return False
    _, report = guard.scrub_free_text(text)
    if report.rule_hits:
        return True
    if getattr(report, "model_hits", 0) > 0:
        return True
    return False


def _scan_messages(messages: list[ModelMessage], guard: PhiGuard) -> str | None:
    """Walk every message part; return the first failure-reason string.

    None ⇒ messages are clean. The returned string is short and PHI-free
    (it names the offending part type, not the offending text).
    """
    from pydantic_ai.messages import (
        SystemPromptPart,
        TextPart,
        ToolReturnPart,
        UserPromptPart,
    )

    for msg in messages:
        if _is_phi_safe(msg):
            continue
        parts = getattr(msg, "parts", None)
        if parts is None:
            continue
        for part in parts:
            _auto_mark_safe_tool_return(part)
            if _is_phi_safe(part):
                continue
            if isinstance(
                part, (UserPromptPart, SystemPromptPart, TextPart, ToolReturnPart)
            ):
                content = getattr(part, "content", "")
                if not isinstance(content, str):
                    # ToolReturnPart may carry a dict; flatten to its
                    # JSON string so the regex layer still sees numbers
                    # and names.
                    try:
                        import json

                        content = json.dumps(content, ensure_ascii=False)
                    except Exception:  # noqa: BLE001
                        content = str(content)
                if _scan_text_for_phi(content, guard):
                    return type(part).__name__
        # ModelResponse text parts also flow through TextPart above.
    return None


def _raise_leak(part_type: str) -> None:
    """Emit audit row + raise the typed exception."""
    try:
        audit_event(
            "phi.leak_detected",
            {"layer_triggered": "content", "part_type": part_type},
        )
    except Exception:  # noqa: BLE001
        # Audit failures must never mask the real error.
        logger.warning("phi.leak_detected audit emission failed", exc_info=True)
    raise PhiLeakDetected(
        f"PHI defense engaged: detected PHI in outbound {part_type} content"
    )


class PhiAssertionModel:
    """Cloud-bound ``Model`` decorator that refuses calls carrying PHI.

    Registered as a virtual subclass of pydantic-ai's ``Model`` ABC for
    the same reason ``LoggingModel`` is — ``isinstance(x, Model)``
    passes so callers (and ``infer_model``) treat the wrapper as a
    drop-in. All other ``Model`` attributes pass through to ``_inner``
    via ``__getattr__``.
    """

    def __init__(
        self, inner: "_PydanticModel", *, guard: PhiGuard | None = None
    ) -> None:
        self._inner = inner
        self._guard = guard if guard is not None else PhiGuard.from_config()

    # ---- delegation -----------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)

    async def __aenter__(self) -> "PhiAssertionModel":
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> Any:
        return await self._inner.__aexit__(*args)

    # ---- pydantic-ai Model interface ------------------------------------

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: "ModelSettings | None",
        model_request_parameters: "ModelRequestParameters",
    ) -> Any:
        offending = _scan_messages(messages, self._guard)
        if offending is not None:
            _raise_leak(offending)
        return await self._inner.request(
            messages, model_settings, model_request_parameters
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: "ModelSettings | None",
        model_request_parameters: "ModelRequestParameters",
        run_context: Any = None,
    ) -> AsyncIterator[Any]:
        offending = _scan_messages(messages, self._guard)
        if offending is not None:
            _raise_leak(offending)
        # ``request_stream`` is itself a context manager on the inner
        # model; forward all kwargs the inner signature accepts.
        if run_context is not None:
            ctx = self._inner.request_stream(
                messages,
                model_settings,
                model_request_parameters,
                run_context,
            )
        else:
            ctx = self._inner.request_stream(
                messages, model_settings, model_request_parameters
            )
        async with ctx as stream:
            yield stream


# Register as a virtual Model subclass so isinstance(wrapper, Model) ==
# True. Mirrors LoggingModel's approach; lets ``infer_model`` and other
# pydantic-ai helpers accept the wrapper unchanged.
_PydanticModel.register(PhiAssertionModel)  # type: ignore[arg-type]
