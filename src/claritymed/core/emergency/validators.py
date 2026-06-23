"""Phase 4: output validator + post-stream audit tripwire.

Two layers, both reading the same triage assessment:

* :func:`make_triage_output_validator` — pydantic-ai
  ``@agent.output_validator`` factory. Rejects (via ``ModelRetry``)
  non-critical replies that omit the rule's localized action wording.
  Budget = 1 retry; on second failure the validator emits
  ``redflag.validator_unrecoverable`` and accepts the reply so the
  user is not blocked by gate downtime (plan §"Open Questions" →
  fail-open default).
* :func:`audit_reply_missing_action_if_needed` — KTD R10 defense-in-
  depth tripwire. Runs *after* the validator passed and the reply is
  about to be persisted; scans the final text for the same action
  fingerprint and emits ``redflag.reply_missing_action`` if absent.
  Audit-only; never mutates the reply.

Why both: validators run on the LLM's raw output (pre-finalize), so
prompt drift can defeat them silently if the model learns to
fingerprint-match the validator. The tripwire is a second, separate
read of the persisted final text. Two layers, two independent audit
trails — the same logic that powers ``symptoms.safety_keywords``
in the symptoms_plugin (KTD-E5: parallel safety paths by design).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Callable

from claritymed.core.emergency.schemas import EmergencyAssessment
from claritymed.core.i18n.loader import t as i18n_t
from claritymed.core.observability.audit import audit_event

logger = logging.getLogger(__name__)

# Length of the canonical action substring we look for in the reply.
# Long enough to be distinctive ("call your local emergency number"
# / "请立即拨打 120") but short enough to survive minor paraphrase
# (extra words, punctuation swap). Empirically tuned; if the eval
# baseline shows recall drift, lower this first.
_ACTION_FINGERPRINT_CHARS = 18

_BOLD_BLOCK_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")
# Triage levels for which the validator + tripwire fire. Critical
# never reaches the agent (short-circuit handles it), routine has
# nothing to enforce, off bypasses the gate entirely.
_ENFORCEABLE_LEVELS: tuple[str, ...] = ("urgent", "moderate")


def extract_action_fingerprint(i18n_text: str) -> str:
    """Return the canonical short substring to look for in a reply.

    The i18n action wording leads with a bolded sentence — that is the
    load-bearing safety directive. We extract the first ``**...**``
    block, normalize whitespace + unicode, and keep the first
    ``_ACTION_FINGERPRINT_CHARS`` characters. Both validator + tripwire
    use this same fingerprint so the two layers agree on "what counts
    as honoring the action".

    Returns empty string when the i18n text has no bolded prefix —
    the validator then treats the rule as "no enforceable phrase",
    which is the right behavior for any future rule that ships
    plain-text action wording.
    """
    match = _BOLD_BLOCK_RE.search(i18n_text)
    if not match:
        return ""
    raw = match.group(1)
    # NFKC: collapse fullwidth ASCII (e.g. 120 in the zh action key)
    # so "拨打１２０" matches "拨打 120" both ways.
    normalized = unicodedata.normalize("NFKC", raw)
    collapsed = _WHITESPACE_RE.sub(" ", normalized).strip().lower()
    return collapsed[:_ACTION_FINGERPRINT_CHARS]


def reply_honors_triage(
    reply: str,
    triage: EmergencyAssessment,
    *,
    language: str,
) -> bool:
    """Return True iff the reply mentions the triage's action wording.

    ``True`` when:
      * The triage carries no enforceable action (``routine``, no
        matched_rules, or no ``suggested_action_i18n_key``) — there is
        nothing to enforce.
      * The reply contains the fingerprint substring (case-insensitive,
        NFKC-normalized).

    ``False`` only when the triage *has* an enforceable action and the
    reply omits it. That is the validator's retry / tripwire's audit
    trigger.
    """
    if triage.level not in _ENFORCEABLE_LEVELS:
        return True
    if triage.suggested_action_i18n_key is None:
        return True
    action_text = i18n_t(triage.suggested_action_i18n_key, lang=language)
    if action_text == triage.suggested_action_i18n_key:
        # Missing i18n key — cannot enforce; treat as passing rather
        # than blocking the user on operator misconfiguration.
        return True
    fingerprint = extract_action_fingerprint(action_text)
    if not fingerprint:
        return True
    haystack = unicodedata.normalize("NFKC", reply).lower()
    haystack = _WHITESPACE_RE.sub(" ", haystack)
    return fingerprint in haystack


def make_triage_output_validator(
    *,
    language: str,
    retry_budget: int = 1,
) -> Callable[[Any, str], Any]:
    """Return a pydantic-ai-compatible output_validator callable.

    Closure state holds an attempt counter scoped to ONE agent build
    (call ``make_triage_output_validator`` per turn from
    :meth:`AskService._build_agent_for_turn` so the counter resets).
    Behavior:

    * Reply honors triage → return reply unchanged.
    * Reply omits action AND attempts ≤ retry_budget → raise
      ``ModelRetry`` with an explicit instruction. The LLM re-runs;
      this validator is called again on the new output.
    * Reply omits action AND attempts > retry_budget → emit
      ``redflag.validator_unrecoverable`` audit event and accept the
      reply. Operator reads the audit row to catch prompt drift.
    """
    from pydantic_ai import ModelRetry

    attempts = {"n": 0}

    def _validator(ctx: Any, output: str) -> str:
        deps = getattr(ctx, "deps", None)
        triage = getattr(deps, "triage", None)
        if triage is None:
            return output
        if reply_honors_triage(output, triage, language=language):
            return output

        attempts["n"] += 1
        if attempts["n"] <= retry_budget:
            action_text = i18n_t(
                triage.suggested_action_i18n_key or "",
                lang=language,
            )
            raise ModelRetry(
                "Your reply did not lead with the required safety "
                f"action wording for a {triage.level} assessment. "
                "Prepend this sentence VERBATIM and then continue with "
                f"your answer:\n\n{action_text.strip()}"
            )

        try:
            audit_event(
                "redflag.validator_unrecoverable",
                payload={
                    "level": triage.level,
                    "rule_ids": [r.rule_id for r in triage.matched_rules],
                    "action_i18n_key": triage.suggested_action_i18n_key,
                    "retries": attempts["n"] - 1,
                },
            )
        except Exception:  # noqa: BLE001
            logger.exception("redflag.validator_unrecoverable emit failed")
        # Pass through — fail-open per the plan's open-question default.
        # The post-stream tripwire still fires for this reply, so the
        # operator gets a second audit signal.
        return output

    return _validator


def audit_reply_missing_action_if_needed(
    final_text: str,
    triage: EmergencyAssessment | None,
    *,
    language: str,
) -> None:
    """Emit ``redflag.reply_missing_action`` when the post-stream check fails.

    Called from :meth:`AskService._finalize_turn` after the reply is
    finalized but before chat-session persistence. The post-stream
    check is independent of the validator (a learnt model could pass
    the validator and still drift on the next turn — two checks, two
    audit trails).

    Never mutates the reply. ``triage`` may be ``None`` (off-mode);
    nothing to enforce in that case.
    """
    if triage is None:
        return
    if reply_honors_triage(final_text, triage, language=language):
        return
    try:
        audit_event(
            "redflag.reply_missing_action",
            payload={
                "level": triage.level,
                "rule_ids": [r.rule_id for r in triage.matched_rules],
                "action_i18n_key": triage.suggested_action_i18n_key,
            },
        )
    except Exception:  # noqa: BLE001
        logger.exception("redflag.reply_missing_action emit failed")
