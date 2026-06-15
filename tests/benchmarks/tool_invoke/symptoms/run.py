"""Symptoms tool-trigger benchmark runner.

Measures whether the LLM invokes ``predict_disease_from_symptoms`` for
in-scope symptom complaints and correctly refrains for false-positive inputs.

Detection differs from the ingest benchmark: the symptoms tool never reaches
the approval channel. Instead, it drives patient-question modals via the
prompt channel. A trial is counted as "tool invoked" when
``len(channel.calls) >= MODAL_THRESHOLD`` (3), matching the e2e test signal.

Eligibility is always stubbed (``_AlwaysEligible``). This isolates LLM
tool-selection behavior from the eligibility gate, which has its own unit
suite under ``tests/core/symptoms/eligibility``.

Pre-flight:
* Start the symptoms server: ``scripts/run.sh symptoms``
* ``configs/symptoms.yaml`` datasets[0].enabled = true + manifest sha256 set

Example::

    uv run python -m tests.benchmarks.tool_invoke.symptoms.run \\
        --models omlx,deepseek-v4-pro \\
        --langs en,zh --trials 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from claritymed.config import load_env_file, load_symptoms_config
from claritymed.context import apply_context, new_request_id, reset_context
from claritymed.core.interaction.schemas import (
    AskUserQuestionInput,
    AskUserQuestionResult,
)
from claritymed.core.rag import load_retrieval_config
from claritymed.core.symptoms.client import SymptomsServerClient
from claritymed.core.symptoms.eligibility.base import (
    EligibilityResult,
    EligibilityStrategy,
)
from claritymed.core.symptoms.registry import DatasetRegistry
from claritymed.orchestrator.features.symptoms_plugin import SymptomsFeature
from claritymed.orchestrator.services import AskService
from claritymed.orchestrator.services.chat_session import ChatSession

from tests.benchmarks.tool_invoke import base
from tests.benchmarks.tool_invoke.symptoms.cases import (
    CASES,
    MODAL_THRESHOLD,
    Case,
)

logger = logging.getLogger(__name__)

PER_TURN_TIMEOUT_S = 240.0
_DEFAULT_SYMPTOMS_URL = "http://127.0.0.1:8084"

CORRECT_OUTCOMES: frozenset[str] = frozenset({"correct"})


# --- always-eligible stub -------------------------------------------------


class _AlwaysEligible(EligibilityStrategy):
    """Bypass eligibility so the bench isolates LLM tool-selection only."""

    async def check(self, complaint, language, profile, dataset) -> EligibilityResult:
        return EligibilityResult(eligible=True, reason="in_scope", confidence=1.0)


# --- channels -------------------------------------------------------------


class _AutoAnswerChannel:
    """Answers symptom question modals with plausible values.

    Decision rules match the e2e ``_AutoAnswerChannel``:
    * Age question → 45
    * Other numeric → question's min
    * Sex → Male
    * Yes/No options → Yes
    * Anything else → first option
    """

    def __init__(self) -> None:
        self.calls: list[AskUserQuestionInput] = []

    async def ask(self, payload: AskUserQuestionInput) -> AskUserQuestionResult:
        self.calls.append(payload)
        answers: dict[str, str] = {}
        numeric_values: dict[str, float] = {}
        for q in payload.questions:
            qtext = q.question
            if q.numeric is not None:
                if "old" in qtext.lower() or "年" in qtext or "岁" in qtext:
                    numeric_values[qtext] = 45.0
                else:
                    numeric_values[qtext] = float(q.numeric.min)
                continue
            if not q.options:
                answers[qtext] = "Yes"
                continue
            labels = [opt.label for opt in q.options]
            lower = qtext.lower()
            if "sex" in lower or "性别" in qtext:
                pick = next(
                    (lab for lab in labels if lab.lower() in {"male", "男"}),
                    labels[0],
                )
            else:
                pick = next(
                    (lab for lab in labels if lab.lower() in {"yes", "是"}),
                    labels[0],
                )
            answers[qtext] = pick
        return AskUserQuestionResult(answers=answers, numeric_values=numeric_values)


class _NullApprovalChannel:
    """Approves any ingest tool that fires (fp guard — we only care about
    symptoms modals, but fp cases might legitimately call ingest tools)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def request(
        self,
        tool_name: str,
        args: dict,
        *,
        breadcrumb: str | None = None,
    ) -> Any:
        from claritymed.core.interaction import ApprovalDecision

        self.calls.append({"tool_name": tool_name, "args": dict(args)})
        return ApprovalDecision(decision="once")


# --- factory ------------------------------------------------------------------


def _make_symptoms_factory(client: SymptomsServerClient):
    config = load_symptoms_config()
    registry = DatasetRegistry(config.datasets)
    eligibility = _AlwaysEligible()

    def _factory() -> SymptomsFeature:
        return SymptomsFeature(
            config=config,
            registry=registry,
            client=client,
            eligibility=eligibility,
        )

    return _factory


# --- server pre-flight --------------------------------------------------------


def _symptoms_server_ready(base_url: str) -> bool:
    try:
        resp = httpx.get(f"{base_url}/health", timeout=3.0)
    except httpx.HTTPError:
        return False
    if resp.status_code != 200:
        return False
    body = resp.json()
    return bool(body.get("datasets_loaded"))


# --- trial dataclass ----------------------------------------------------------


@dataclass
class TrialRecord:
    request_id: str
    timestamp_utc: str
    model: str
    lang: str
    case_name: str
    tier: str
    expected_behavior: str
    expected_tool: str | None
    user_prompt: str
    # Detection signal: number of symptom question modals the plugin drove.
    modal_call_count: int
    # Ingest-tool approval calls (expected 0 for call_tool cases; present on fp).
    ingest_calls: list[dict]
    final_response_text: str
    # Predicate / classification
    outcome: str
    predicate_pass: bool
    predicate_reason: str
    tool_invoked: bool
    # Perf
    latency_ms: float
    had_error: bool
    error_msg: str | None


# --- single trial -------------------------------------------------------------


async def _run_one_trial(
    provider_id: str,
    user_lang: str,
    case: Case,
    symptoms_url: str,
) -> TrialRecord:
    from claritymed.core.llm.model import build_model
    from claritymed.stores.models import resolve_provider

    base.wipe_bench_user_dir()
    base.reload_runtime()

    prompt = case.prompts[user_lang]
    request_id = new_request_id()
    tokens = apply_context(request_id, base.USER_ID, user_lang)

    channel = _AutoAnswerChannel()
    approval = _NullApprovalChannel()
    final_chunks: list[str] = []
    error_msg: str | None = None
    t0 = time.perf_counter()

    try:
        provider = resolve_provider(override=provider_id)
        model = build_model(provider)
        chat = ChatSession.new(base.USER_ID)
        rag_mode = load_retrieval_config().rag.mode

        async with SymptomsServerClient(symptoms_url) as client:
            service = AskService(
                model=model,
                chat_session=chat,
                provider_id=provider.id,
                model_name=provider.model,
                provider_config=provider,
                rag_mode=rag_mode,
                prompt_channel=channel,
                tool_approval_channel=approval,
                symptoms_factory=_make_symptoms_factory(client),
            )
            try:
                async with asyncio.timeout(PER_TURN_TIMEOUT_S):
                    async for ev in service.run(prompt, user_id=base.USER_ID):
                        if getattr(ev, "type", None) == "error":
                            error_msg = (
                                f"{getattr(ev, 'error_type', 'unknown')}: "
                                f"{getattr(ev, 'message', '')}"
                            )
                            continue
                        text = getattr(ev, "text", None)
                        if isinstance(text, str):
                            final_chunks.append(text)
            except asyncio.TimeoutError:
                error_msg = f"timeout after {PER_TURN_TIMEOUT_S:.0f}s"
    except Exception as exc:  # noqa: BLE001
        error_msg = f"{type(exc).__name__}: {exc}"
    finally:
        reset_context(tokens)

    latency_ms = (time.perf_counter() - t0) * 1000
    modal_count = len(channel.calls)
    final_text = "".join(final_chunks).strip()

    outcome_info = _classify(case, modal_count, error_msg)

    return TrialRecord(
        request_id=request_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model=provider_id,
        lang=user_lang,
        case_name=case.name,
        tier=case.tier,
        expected_behavior=case.expected_behavior,
        expected_tool=case.expected_tool,
        user_prompt=prompt,
        modal_call_count=modal_count,
        ingest_calls=list(approval.calls),
        final_response_text=final_text,
        outcome=outcome_info["outcome"],
        predicate_pass=outcome_info["predicate_pass"],
        predicate_reason=outcome_info["predicate_reason"],
        tool_invoked=outcome_info["tool_invoked"],
        latency_ms=round(latency_ms, 1),
        had_error=error_msg is not None,
        error_msg=error_msg,
    )


# --- classify -----------------------------------------------------------------


def _classify(
    case: Case,
    modal_count: int,
    error_msg: str | None,
) -> dict[str, Any]:
    tool_invoked = modal_count >= MODAL_THRESHOLD

    if error_msg is not None:
        return dict(
            outcome="errored",
            predicate_pass=False,
            predicate_reason=f"trial raised: {error_msg}",
            tool_invoked=tool_invoked,
        )

    if case.expected_behavior == "call_tool":
        pred_ok, pred_reason = case.args_predicate(modal_count)
        outcome = "correct" if pred_ok else "no_tool"
        return dict(
            outcome=outcome,
            predicate_pass=pred_ok,
            predicate_reason=pred_reason,
            tool_invoked=tool_invoked,
        )

    if case.expected_behavior == "decline":
        if tool_invoked:
            return dict(
                outcome="false_positive",
                predicate_pass=False,
                predicate_reason=f"symptoms tool invoked (modal_calls={modal_count})",
                tool_invoked=True,
            )
        return dict(
            outcome="correct",
            predicate_pass=True,
            predicate_reason=f"tool not invoked (modal_calls={modal_count})",
            tool_invoked=False,
        )

    raise ValueError(f"unknown expected_behavior: {case.expected_behavior!r}")


# --- aggregate ----------------------------------------------------------------


def _summary_rows(trials: list[TrialRecord]) -> list[dict]:
    by_cell: dict[tuple[str, str, str], list[TrialRecord]] = {}
    for t in trials:
        key = (t.model, t.lang, t.case_name)
        by_cell.setdefault(key, []).append(t)

    rows: list[dict] = []
    for (model, lang, case_name), cell in sorted(by_cell.items()):
        latencies = [t.latency_ms for t in cell if not t.had_error]
        n = len(cell)
        n_correct = sum(1 for t in cell if t.outcome in CORRECT_OUTCOMES)
        n_err = sum(1 for t in cell if t.had_error)
        n_invoked = sum(1 for t in cell if t.tool_invoked)
        rows.append(
            {
                "model": model,
                "user_lang": lang,
                "case": case_name,
                "tier": cell[0].tier,
                "expected_behavior": cell[0].expected_behavior,
                "n_trials": n,
                "correct_rate": round(n_correct / n, 3) if n else 0.0,
                "tool_invoked_rate": round(n_invoked / n, 3) if n else 0.0,
                "error_count": n_err,
                "mean_latency_ms": round(statistics.mean(latencies), 1)
                if latencies
                else "",
                "p50_latency_ms": round(statistics.median(latencies), 1)
                if latencies
                else "",
                "p95_latency_ms": base.p95(latencies),
            }
        )
    return rows


def _outcomes_rows(trials: list[TrialRecord]) -> list[dict]:
    by_cell: dict[tuple[str, str, str, str, str, str], int] = {}
    for t in trials:
        key = (t.model, t.lang, t.case_name, t.tier, t.expected_behavior, t.outcome)
        by_cell[key] = by_cell.get(key, 0) + 1

    rows: list[dict] = []
    for key, count in sorted(by_cell.items()):
        model, lang, case_name, tier, eb, outcome = key
        rows.append(
            {
                "model": model,
                "user_lang": lang,
                "case": case_name,
                "tier": tier,
                "expected_behavior": eb,
                "outcome": outcome,
                "count": count,
            }
        )
    return rows


# --- CLI ----------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    base.add_common_args(p)
    p.add_argument(
        "--symptoms-url",
        default=_DEFAULT_SYMPTOMS_URL,
        help=f"symptoms server base URL (default: {_DEFAULT_SYMPTOMS_URL})",
    )
    return p.parse_args()


async def _main_async(args: argparse.Namespace) -> int:
    load_env_file()

    symptoms_url: str = args.symptoms_url

    if not _symptoms_server_ready(symptoms_url):
        print(
            f"symptoms server not ready at {symptoms_url}\n"
            "  Start it with: scripts/run.sh symptoms\n"
            "  Check configs/symptoms.yaml has datasets[0].enabled = true",
            file=sys.stderr,
        )
        return 1

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    user_langs = [s.strip() for s in args.user_langs.split(",") if s.strip()]
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    names = (
        [n.strip() for n in args.cases.split(",") if n.strip()] if args.cases else None
    )
    selected = base.select_cases(CASES, tiers, names)
    if not selected:
        print("no cases selected", file=sys.stderr)
        return 2

    _now = datetime.now()
    ts = _now.strftime("%Y%m%d_%H%M%S_") + f"{_now.microsecond // 1000:03d}"
    out_dir = Path(args.out) if args.out else Path("data/bench/symptoms") / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "trials.jsonl"
    csv_path = out_dir / "summary.csv"
    outcomes_csv_path = out_dir / "outcomes.csv"

    total = len(models) * len(user_langs) * len(selected) * args.trials
    print(
        f"models={models} user_langs={user_langs} cases={len(selected)} "
        f"trials={args.trials} → total={total} trials; out={out_dir}"
    )

    trials: list[TrialRecord] = []
    counter = 0
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for model in models:
            for ulang in user_langs:
                for case in selected:
                    for k in range(args.trials):
                        counter += 1
                        t0 = time.perf_counter()
                        rec = await _run_one_trial(model, ulang, case, symptoms_url)
                        trials.append(rec)
                        fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                        fh.flush()
                        elapsed_ms = (time.perf_counter() - t0) * 1000
                        print(
                            f"[{counter}/{total}] {model:<22} u={ulang} "
                            f"{case.name:<40} {k + 1}/{args.trials} "
                            f"{case.expected_behavior:<12} -> "
                            f"{rec.outcome:<16} modal={rec.modal_call_count} "
                            f"{elapsed_ms:>6.0f}ms"
                        )

    base.write_csv(csv_path, _summary_rows(trials))
    base.write_csv(outcomes_csv_path, _outcomes_rows(trials))

    print(f"\nwrote {len(trials)} trials → {jsonl_path}")
    print(
        f"render HTML: uv run python -m tests.benchmarks.tool_invoke.report "
        f"--run {out_dir}"
    )
    return 0


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    try:
        sys.exit(asyncio.run(_main_async(args)))
    except KeyboardInterrupt:
        os._exit(130)


if __name__ == "__main__":
    main()
