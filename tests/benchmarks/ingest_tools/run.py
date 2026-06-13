"""Ingest-tool benchmark runner.

Runs (case × model × lang × trial) matrix. Each trial is independent:

* fresh ``bench`` user dir (so prior trial side effects can't bias the
  current run's grading)
* fresh ``ChatSession`` (no chat history leakage)
* fresh approval + prompt channels (no cross-trial call accumulation)

Outputs land under ``--out`` (default ``data/bench/<timestamp>/``):

* ``trials.jsonl``  — one line per trial, self-contained payload that
                      ``judge.py`` reads without rerunning anything
* ``summary.csv``   — aggregated per (model, lang, case) cell

The runner does NOT call any LLM judge — that's ``judge.py``'s job.
Predicate grading happens here because it's deterministic and free.

Example::

    uv run python -m tests.benchmarks.ingest_tools.run \\
        --models omlx,deepseek-v4-pro \\
        --langs en,zh --trials 5
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib
import json
import logging
import os
import shutil
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from claritymed import config as _cfg
from claritymed.config import load_env_file
from claritymed.context import apply_context, new_request_id, reset_context
from claritymed.core.interaction import ApprovalDecision
from claritymed.core.interaction.prompt_channel import UserDeclinedAnswer
from claritymed.core.interaction.schemas import (
    AskUserQuestionInput,
    AskUserQuestionResult,
)
from claritymed.core.rag import load_retrieval_config
from claritymed.orchestrator.services import AskService
from claritymed.orchestrator.services.chat_session import ChatSession

from tests.benchmarks.ingest_tools.cases import (
    CASES,
    INGEST_TOOLS,
    USER_ID,
    Case,
)

logger = logging.getLogger(__name__)

PER_TURN_TIMEOUT_S = 90.0
BENCH_HOME = Path.home() / ".claritymed"
BENCH_USER_DIR = BENCH_HOME / "data" / "users" / USER_ID


# --- channels --------------------------------------------------------


@dataclass
class _RecordedCall:
    tool_name: str
    args: dict


class _AutoApproveChannel:
    """Approves every ingest tool. Records (tool_name, args) tuples."""

    def __init__(self) -> None:
        self.calls: list[_RecordedCall] = []

    async def request(
        self,
        tool_name: str,
        args: dict,
        *,
        breadcrumb: str | None = None,
    ) -> ApprovalDecision:
        self.calls.append(_RecordedCall(tool_name=tool_name, args=dict(args)))
        return ApprovalDecision(decision="once")


class _RecordingDeclinePromptChannel:
    """Records every ``ask_user_question`` payload and tells the model the
    user declined to answer, so the model stops instead of looping.

    Why decline rather than auto-answer: an auto-answer with the first
    option would let the model proceed to call an ingest tool with
    canned data, contaminating the trial's tool-call sequence. Decline
    cleanly stops the deferred-loop after the ask was recorded.
    """

    def __init__(self) -> None:
        self.calls: list[AskUserQuestionInput] = []

    async def ask(self, payload: AskUserQuestionInput) -> AskUserQuestionResult:
        self.calls.append(payload)
        raise UserDeclinedAnswer("benchmark stub: user declined")


class _AutoAnswerFirstOptionChannel:
    """Records asks and answers every question with option[0].

    Used for ``ask_then_call_tool`` cases: the model asks for a missing
    detail, receives the first presented option, and then proceeds to
    call the ingest tool with that answer.  We always pick option[0] to
    keep the trial deterministic — predicate grading only checks that
    the right tool fired, not which option was chosen.
    """

    def __init__(self) -> None:
        self.calls: list[AskUserQuestionInput] = []

    async def ask(self, payload: AskUserQuestionInput) -> AskUserQuestionResult:
        self.calls.append(payload)
        answers: dict[str, str | list[str]] = {
            q.question: q.options[0].label for q in payload.questions
        }
        return AskUserQuestionResult(answers=answers)


# --- bench env helpers ----------------------------------------------


def _wipe_bench_user_dir() -> None:
    if BENCH_USER_DIR.exists():
        shutil.rmtree(BENCH_USER_DIR)


def _reload_runtime() -> None:
    """Re-resolve config + clear per-user caches so the next trial sees
    a clean store layer.

    We don't touch ``CLARITYMED_HOME`` between trials (we leave it at
    the user's real ``~/.claritymed/``). Instead we wipe the ``bench``
    user subtree directly, then clear caches that hold per-user engines.
    """
    importlib.reload(_cfg)
    _cfg.reload_configs()
    from claritymed.stores import profile as _profile

    _profile._ENGINES.clear()
    from claritymed.stores.account import reset_account_cache

    reset_account_cache()


# --- trial dataclass -------------------------------------------------


@dataclass
class TrialRecord:
    # ``request_id`` is the cross-system correlation key: it travels into
    # the ContextVar that audit.log uses, into OpenTelemetry baggage as
    # ``claritymed.request_id`` (Phoenix span attribute), and is written
    # here so a maintainer can grep audit.log / Phoenix straight from a
    # failed row in trials.jsonl.
    request_id: str
    timestamp_utc: str
    model: str
    lang: str
    case_name: str
    tier: str
    expected_behavior: str
    expected_tool: str | None
    expected_tools: list[str]
    tool_prompt_lang: str  # actual lang forced into CLARITYMED_TOOL_PROMPT_LANG
    user_prompt: str
    seed: dict | None
    tool_calls: list[dict]
    ask_questions: list[dict]
    final_response_text: str
    # Predicate / classification
    outcome: str  # see classify()
    predicate_pass: bool
    predicate_reason: str
    correct_tool: bool
    wrong_tools: list[str]
    ask_user_q_count: int
    no_tool: bool
    # Perf
    latency_ms: float
    error: str | None


# --- single trial ----------------------------------------------------


async def _run_one_trial(
    provider_id: str,
    user_lang: str,
    tool_prompt_lang: str,
    case: Case,
) -> TrialRecord:
    """Execute one independent trial. Always returns a TrialRecord
    (even on error / timeout) so the JSONL row is never lost.

    ``user_lang`` selects the prompt template + the chat-response language
    (apply_context).  ``tool_prompt_lang`` is independently forced into
    ``CLARITYMED_TOOL_PROMPT_LANG`` so the tool-description language can
    be decoupled from the user-input language for cross-axis ablation
    (e.g. zh user input vs. en tool descriptions).
    """
    from claritymed.core.llm.model import build_model
    from claritymed.stores.models import resolve_provider

    os.environ["CLARITYMED_TOOL_PROMPT_LANG"] = tool_prompt_lang
    _wipe_bench_user_dir()
    _reload_runtime()

    seed: dict | None = case.seed() if case.seed else None
    prompt = case.prompts[user_lang]
    if seed:
        prompt = prompt.format(**seed)

    # Pre-allocate request_id so it can be logged + passed into apply_context
    # in one shot. Same value lands in audit log lines and Phoenix span
    # baggage, so a JSONL row identifies the corresponding traces.
    request_id = new_request_id()
    tokens = apply_context(request_id, USER_ID, user_lang)

    approval = _AutoApproveChannel()
    # ask_then_call_tool cases need the model to receive a real answer so it
    # can proceed to call the ingest tool.  All other cases use decline to
    # stop after the ask is recorded and prevent follow-on tool contamination.
    if case.expected_behavior == "ask_then_call_tool":
        prompt_channel: (
            _RecordingDeclinePromptChannel | _AutoAnswerFirstOptionChannel
        ) = _AutoAnswerFirstOptionChannel()
    else:
        prompt_channel = _RecordingDeclinePromptChannel()
    final_chunks: list[str] = []
    error_msg: str | None = None
    t0 = time.perf_counter()

    try:
        provider = resolve_provider(override=provider_id)
        model = build_model(provider)
        chat = ChatSession.new(USER_ID)
        rag_mode = load_retrieval_config().rag.mode

        service = AskService(
            model=model,
            chat_session=chat,
            provider_id=provider.id,
            model_name=provider.model,
            provider_config=provider,
            rag_mode=rag_mode,
            tool_approval_channel=approval,
            prompt_channel=prompt_channel,
        )

        try:
            async with asyncio.timeout(PER_TURN_TIMEOUT_S):
                async for ev in service.run(prompt, user_id=USER_ID):
                    # TokenChunk is the only text-bearing event we care
                    # about; everything else (RetrievalStarted, ToolStarted,
                    # …) is for UI progress and irrelevant to the judge.
                    text = getattr(ev, "text", None)
                    if isinstance(text, str):
                        final_chunks.append(text)
        except asyncio.TimeoutError:
            error_msg = f"timeout after {PER_TURN_TIMEOUT_S:.0f}s"
    except Exception as exc:  # noqa: BLE001
        error_msg = f"{type(exc).__name__}: {exc}"
    finally:
        reset_context(tokens)
        os.environ.pop("CLARITYMED_TOOL_PROMPT_LANG", None)

    latency_ms = (time.perf_counter() - t0) * 1000

    tool_calls_payload = [
        {"tool_name": c.tool_name, "args": c.args} for c in approval.calls
    ]
    ask_qs_payload = [p.model_dump() for p in prompt_channel.calls]

    outcome_info = _classify(case, approval.calls, prompt_channel.calls)

    return TrialRecord(
        request_id=request_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model=provider_id,
        lang=user_lang,
        case_name=case.name,
        tier=case.tier,
        expected_behavior=case.expected_behavior,
        expected_tool=case.expected_tool,
        expected_tools=list(case.expected_tools),
        tool_prompt_lang=tool_prompt_lang,
        user_prompt=prompt,
        seed=seed,
        tool_calls=tool_calls_payload,
        ask_questions=ask_qs_payload,
        final_response_text="".join(final_chunks).strip(),
        outcome=outcome_info["outcome"],
        predicate_pass=outcome_info["predicate_pass"],
        predicate_reason=outcome_info["predicate_reason"],
        correct_tool=outcome_info["correct_tool"],
        wrong_tools=outcome_info["wrong_tools"],
        ask_user_q_count=len(prompt_channel.calls),
        no_tool=outcome_info["no_tool"],
        latency_ms=round(latency_ms, 1),
        error=error_msg,
    )


def _classify(
    case: Case,
    approval_calls: list[_RecordedCall],
    ask_calls: list[AskUserQuestionInput],
) -> dict[str, Any]:
    """Predicate-only outcome label. Judge does the semantic layer later."""
    ingest_calls = [c for c in approval_calls if c.tool_name in INGEST_TOOLS]
    ingest_names = [c.tool_name for c in ingest_calls]
    no_tool = not approval_calls and not ask_calls

    if case.expected_behavior == "call_tool":
        assert case.expected_tool is not None
        target = [c for c in ingest_calls if c.tool_name == case.expected_tool]
        wrong = [n for n in ingest_names if n != case.expected_tool]
        if target:
            pred_ok, pred_reason = case.args_predicate(target[0].args)
            outcome = "correct" if pred_ok else "predicate_fail"
            return dict(
                outcome=outcome,
                predicate_pass=pred_ok,
                predicate_reason=pred_reason,
                correct_tool=True,
                wrong_tools=wrong,
                no_tool=False,
            )
        if wrong:
            return dict(
                outcome="wrong_tool",
                predicate_pass=False,
                predicate_reason=f"wrong tools called: {wrong}",
                correct_tool=False,
                wrong_tools=wrong,
                no_tool=False,
            )
        if ask_calls:
            return dict(
                outcome="asked_instead",
                predicate_pass=False,
                predicate_reason="model asked instead of calling expected tool",
                correct_tool=False,
                wrong_tools=[],
                no_tool=False,
            )
        return dict(
            outcome="no_tool",
            predicate_pass=False,
            predicate_reason="no tool invoked",
            correct_tool=False,
            wrong_tools=[],
            no_tool=True,
        )

    if case.expected_behavior == "call_tools":
        expected = set(case.expected_tools)
        got = set(ingest_names)
        missing = expected - got
        extra = got - expected
        if not missing:
            return dict(
                outcome="correct" if not extra else "correct_with_extra",
                predicate_pass=True,
                predicate_reason=f"called {sorted(got)}; expected ⊇ {sorted(expected)}",
                correct_tool=True,
                wrong_tools=sorted(extra),
                no_tool=False,
            )
        return dict(
            outcome="missing_tools",
            predicate_pass=False,
            predicate_reason=f"missing {sorted(missing)}; called {sorted(got)}",
            correct_tool=False,
            wrong_tools=sorted(extra),
            no_tool=not got and not ask_calls,
        )

    if case.expected_behavior == "decline":
        # Success = no INGEST_TOOL fired. Ask is allowed.
        if not ingest_calls:
            return dict(
                outcome="correct",
                predicate_pass=True,
                predicate_reason=f"no ingest tool called (ask_q={len(ask_calls)})",
                correct_tool=True,
                wrong_tools=[],
                no_tool=no_tool,
            )
        return dict(
            outcome="false_positive",
            predicate_pass=False,
            predicate_reason=f"unexpected ingest tools: {ingest_names}",
            correct_tool=False,
            wrong_tools=ingest_names,
            no_tool=False,
        )

    if case.expected_behavior == "ask":
        if ask_calls:
            return dict(
                outcome="correct" if not ingest_calls else "asked_with_call",
                predicate_pass=True,
                predicate_reason=f"asked {len(ask_calls)} times",
                correct_tool=True,
                wrong_tools=ingest_names if ingest_calls else [],
                no_tool=False,
            )
        if ingest_calls:
            return dict(
                outcome="guessed_instead",
                predicate_pass=False,
                predicate_reason=f"guessed via {ingest_names} instead of asking",
                correct_tool=False,
                wrong_tools=ingest_names,
                no_tool=False,
            )
        return dict(
            outcome="no_ask",
            predicate_pass=False,
            predicate_reason="neither asked nor called any tool",
            correct_tool=False,
            wrong_tools=[],
            no_tool=True,
        )

    if case.expected_behavior == "ask_then_call_tool":
        # Success = asked at least once AND then called the expected ingest tool.
        # The auto-answer channel answered, so ingest tool should have fired.
        assert case.expected_tool is not None
        target = [c for c in ingest_calls if c.tool_name == case.expected_tool]
        wrong = [n for n in ingest_names if n != case.expected_tool]
        if not ask_calls:
            if target:
                return dict(
                    outcome="called_without_asking",
                    predicate_pass=True,
                    predicate_reason=f"skipped ask, called {case.expected_tool} directly",
                    correct_tool=True,
                    wrong_tools=wrong,
                    no_tool=False,
                )
            return dict(
                outcome="no_ask_no_tool",
                predicate_pass=False,
                predicate_reason="neither asked nor called expected tool",
                correct_tool=False,
                wrong_tools=wrong,
                no_tool=True,
            )
        if target:
            pred_ok, pred_reason = case.args_predicate(target[0].args)
            return dict(
                outcome="correct" if pred_ok else "predicate_fail",
                predicate_pass=pred_ok,
                predicate_reason=pred_reason,
                correct_tool=True,
                wrong_tools=wrong,
                no_tool=False,
            )
        return dict(
            outcome="asked_but_no_tool",
            predicate_pass=False,
            predicate_reason=f"asked {len(ask_calls)} time(s) but {case.expected_tool} never called",
            correct_tool=False,
            wrong_tools=wrong,
            no_tool=not ingest_calls,
        )

    raise ValueError(f"unknown expected_behavior: {case.expected_behavior!r}")


# --- aggregate -------------------------------------------------------


CORRECT_OUTCOMES: frozenset[str] = frozenset(
    {"correct", "correct_with_extra", "asked_with_call"}
)


def _summary_rows(trials: list[TrialRecord]) -> list[dict]:
    """One row per (model, user_lang, tool_prompt_lang, case) cell.

    Trimmed to the columns that mean something for *every* expected
    behavior: ``correct_rate`` and ``fail_rate`` are always meaningful,
    ``no_tool_rate`` flags the "model said nothing" failure mode common
    across call_tool / ask cases, and the latency stats are universal.
    The full outcome breakdown lives in ``outcomes.csv`` (long format).
    """
    by_cell: dict[tuple[str, str, str, str], list[TrialRecord]] = {}
    for t in trials:
        key = (t.model, t.lang, t.tool_prompt_lang, t.case_name)
        by_cell.setdefault(key, []).append(t)

    rows: list[dict] = []
    for (model, lang, tool_prompt_lang, case_name), cell in sorted(by_cell.items()):
        latencies = [t.latency_ms for t in cell if t.error is None]
        n = len(cell)
        n_correct = sum(1 for t in cell if t.outcome in CORRECT_OUTCOMES)
        n_no_tool = sum(1 for t in cell if t.no_tool)
        n_ask = sum(t.ask_user_q_count for t in cell)
        n_err = sum(1 for t in cell if t.error)
        n_fail = n - n_correct

        rows.append(
            {
                "model": model,
                "user_lang": lang,
                "tool_prompt_lang": tool_prompt_lang,
                "case": case_name,
                "tier": cell[0].tier,
                "expected_behavior": cell[0].expected_behavior,
                "n_trials": n,
                "correct_rate": round(n_correct / n, 3) if n else 0.0,
                "fail_rate": round(n_fail / n, 3) if n else 0.0,
                "no_tool_rate": round(n_no_tool / n, 3) if n else 0.0,
                "ask_calls_total": n_ask,
                "error_count": n_err,
                "mean_latency_ms": round(statistics.mean(latencies), 1)
                if latencies
                else "",
                "p50_latency_ms": round(statistics.median(latencies), 1)
                if latencies
                else "",
                "p95_latency_ms": _p95(latencies),
            }
        )
    return rows


def _outcomes_rows(trials: list[TrialRecord]) -> list[dict]:
    """Long-format breakdown: one row per (cell × outcome) with count.

    Survives the curse of fixed CSV columns — every outcome label
    (``correct``, ``no_ask``, ``predicate_fail``, ``missing_tools``,
    etc.) gets its own line, so adding a new outcome later doesn't
    silently zero out columns nobody notices.
    """
    by_cell: dict[tuple[str, str, str, str, str, str, str], int] = {}
    for t in trials:
        key = (
            t.model,
            t.lang,
            t.tool_prompt_lang,
            t.case_name,
            t.tier,
            t.expected_behavior,
            t.outcome,
        )
        by_cell[key] = by_cell.get(key, 0) + 1

    rows: list[dict] = []
    for key, count in sorted(by_cell.items()):
        model, lang, tpl, case_name, tier, eb, outcome = key
        rows.append(
            {
                "model": model,
                "user_lang": lang,
                "tool_prompt_lang": tpl,
                "case": case_name,
                "tier": tier,
                "expected_behavior": eb,
                "outcome": outcome,
                "count": count,
            }
        )
    return rows


def _p95(xs: list[float]) -> float | str:
    if not xs:
        return ""
    s = sorted(xs)
    idx = max(0, int(round(0.95 * (len(s) - 1))))
    return round(s[idx], 1)


# --- CLI -------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models",
        required=True,
        help="comma-separated provider ids (e.g. omlx,deepseek-v4-pro)",
    )
    p.add_argument(
        "--user-langs",
        "--langs",
        dest="user_langs",
        default="en,zh",
        help="comma-separated user-input languages (default: en,zh)",
    )
    p.add_argument(
        "--tool-prompt-langs",
        dest="tool_prompt_langs",
        default="inherit",
        help=(
            "comma-separated tool-description languages "
            "(en, zh, or 'inherit' to match the user lang; default: inherit). "
            "Use e.g. 'en,zh' to ablate this axis independently — generates "
            "a separate trial cell for each tool-prompt-lang per user-lang."
        ),
    )
    p.add_argument(
        "--trials",
        type=int,
        default=3,
        help="trials per (model, user-lang, tool-prompt-lang, case) cell (default: 3)",
    )
    p.add_argument(
        "--tiers",
        default="base,hard,fp",
        help="case tiers to include (default: base,hard,fp)",
    )
    p.add_argument(
        "--cases",
        default=None,
        help="optional comma-separated case names to include (filters within tiers)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="output dir (default: data/bench/<timestamp>/)",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _select_cases(tiers: list[str], names: list[str] | None) -> list[Case]:
    out = [c for c in CASES if c.tier in tiers]
    if names:
        wanted = set(names)
        out = [c for c in out if c.name in wanted]
    return out


async def _main_async(args: argparse.Namespace) -> int:
    load_env_file()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    user_langs = [s.strip() for s in args.user_langs.split(",") if s.strip()]
    tool_langs_raw = [s.strip() for s in args.tool_prompt_langs.split(",") if s.strip()]
    for tpl in tool_langs_raw:
        if tpl not in {"en", "zh", "inherit"}:
            print(
                f"invalid --tool-prompt-langs value: {tpl!r} (expected en/zh/inherit)",
                file=sys.stderr,
            )
            return 2
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    names = (
        [n.strip() for n in args.cases.split(",") if n.strip()] if args.cases else None
    )
    selected = _select_cases(tiers, names)
    if not selected:
        print("no cases selected", file=sys.stderr)
        return 2

    _now = datetime.now()
    ts = _now.strftime("%Y%m%d_%H%M%S_") + f"{_now.microsecond // 1000:03d}"
    out_dir = Path(args.out) if args.out else Path("data/bench") / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "trials.jsonl"
    csv_path = out_dir / "summary.csv"
    outcomes_csv_path = out_dir / "outcomes.csv"

    total = (
        len(models)
        * len(user_langs)
        * len(tool_langs_raw)
        * len(selected)
        * args.trials
    )
    print(
        f"models={models} user_langs={user_langs} "
        f"tool_prompt_langs={tool_langs_raw} cases={len(selected)} "
        f"trials={args.trials} → total={total} trials; out={out_dir}"
    )

    trials: list[TrialRecord] = []
    counter = 0
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for model in models:
            for ulang in user_langs:
                for tpl_raw in tool_langs_raw:
                    tpl_effective = ulang if tpl_raw == "inherit" else tpl_raw
                    for case in selected:
                        for k in range(args.trials):
                            counter += 1
                            t0 = time.perf_counter()
                            rec = await _run_one_trial(
                                model, ulang, tpl_effective, case
                            )
                            trials.append(rec)
                            fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                            fh.flush()
                            elapsed_ms = (time.perf_counter() - t0) * 1000
                            print(
                                f"[{counter}/{total}] {model:<22} "
                                f"u={ulang} t={tpl_effective:<7} "
                                f"{case.name:<32} {k + 1}/{args.trials} "
                                f"{case.expected_behavior:<10} -> "
                                f"{rec.outcome:<18} {elapsed_ms:>6.0f}ms"
                            )

    rows = _summary_rows(trials)
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    outcome_rows = _outcomes_rows(trials)
    if outcome_rows:
        with outcomes_csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(outcome_rows[0].keys()))
            writer.writeheader()
            writer.writerows(outcome_rows)

    print(f"\nwrote {len(trials)} trials → {jsonl_path}")
    print(f"wrote {len(rows)} cell summaries → {csv_path}")
    print(f"wrote {len(outcome_rows)} outcome rows → {outcomes_csv_path}")
    print(
        f"\nrender HTML: uv run python -m tests.benchmarks.ingest_tools.report "
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
        # asyncio.run() cleanup calls loop.shutdown_default_executor() which
        # blocks until all ThreadPoolExecutor threads finish — including the
        # in-flight LLM HTTP streaming thread that can't be interrupted. Skip
        # that wait and exit immediately.
        os._exit(130)


if __name__ == "__main__":
    main()
