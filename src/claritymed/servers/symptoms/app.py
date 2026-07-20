"""FastAPI app, lifespan, endpoints, and entry point.

The server is the data-only inference boundary (KTD-7 in the
disease-prediction plan) — it returns raw clinical data and never
ships safety prose or tier labels. Composition into a user-visible
reply is the LLM's job downstream.

Loopback-bound: :data:`HOST` is a module-level constant and the call
to ``uvicorn.run`` asserts against it pre-bind. No env var override
for the host — the only override is :data:`SYMPTOMS_PORT_ENV`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import numpy as np

from claritymed.servers._devices import LOG_CONFIG, add_logging_middleware

try:
    import uvicorn
    from fastapi import APIRouter, FastAPI, HTTPException, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
except ImportError as exc:  # pragma: no cover — import-time guard
    raise SystemExit(
        "claritymed-symptoms-server requires the 'symptoms-server' extra. "
        "Install with:\n    uv sync --extra symptoms-server\n"
        f"(original ImportError: {exc})"
    ) from None

from claritymed.config import load_symptoms_config
from claritymed.core.device import resolve_device
from claritymed.core.symptoms.datasets import LoadedDataset, build_dataset
from claritymed.core.symptoms.init_matcher import InitMatcherEmbedder
from claritymed.core.symptoms.schemas import SymptomsConfig
from claritymed.errors import UnknownDatasetError
from claritymed.ingest.symptoms.xgb.mock_agent import (
    is_mock_enabled as _is_mock_enabled,
)
from claritymed.ingest.symptoms import (  # noqa: F401 — registers adapters
    ddxplus as _register_adapters,
)
from claritymed.ingest.symptoms.typed_basd import (
    AGE_BUCKETS,
    SEX2IDX,
    TypedEnv,
    age_bucket,
)
from claritymed.servers.symptoms.differential import (
    format_cancel_outcome,
    format_differential,
)
from claritymed.servers.symptoms.questions import (
    QuestionPayloadError,
    build_question,
    synth_patient,
)
from claritymed.servers.symptoms.loader import flush_mps_cache
from claritymed.servers.symptoms.state import (
    SERVER_STATE,
    SubSessionState,
    prune_expired_sessions,
)
from claritymed.servers.symptoms.wire import (
    CancelResponse,
    HealthResponse,
    StartSessionRequest,
    StartSessionResponse,
    TurnRequest,
    TurnResponse,
)

logger = logging.getLogger("claritymed.servers.symptoms")

HOST = "127.0.0.1"
DEFAULT_PORT = 8084  # 8082=embedder, 8083=reranker, 8084=symptoms
SYMPTOMS_PORT_ENV = "CLARITYMED_SYMPTOMS_PORT"

# Background TTL purge cadence. Sessions are evicted on a sliding window
# matching ``DatasetSpec.session_ttl_seconds`` (KTD-12), checked every
# minute. Coarser than HTTP-handler granularity but plenty for the
# tens-of-sessions-per-process throughput v1 targets.
TTL_PURGE_INTERVAL_S = 60


# --- lifespan + state plumbing -------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001 — FastAPI signature
    """Load every enabled dataset; spawn the TTL-purge background task."""
    if not SERVER_STATE.config_loaded:
        await _load_config_into_state()
    task = asyncio.create_task(_ttl_purge_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def _load_config_into_state() -> None:
    """Read configs/symptoms.yaml and load every enabled dataset's models.

    Skipped silently when ``CLARITYMED_SYMPTOMS_SKIP_LOAD=1`` — tests
    pre-populate ``SERVER_STATE.datasets`` and don't want the lifespan
    re-loading on top.

    The blocking torch/weights I/O runs in a thread via
    ``asyncio.to_thread`` so the event loop stays responsive during
    startup — concurrent health-check polls won't time out while models
    load.
    """
    if os.environ.get("CLARITYMED_SYMPTOMS_SKIP_LOAD") == "1":
        SERVER_STATE.config_loaded = True
        return
    await asyncio.to_thread(_load_config_sync)


def _load_config_sync() -> None:
    """Blocking body of :func:`_load_config_into_state`; run via asyncio.to_thread.

    Init-symptom matcher: when ``config.init_matcher.enabled`` is True,
    the matcher embedder is constructed once and injected into every
    dataset's ``build_dataset`` call. Catalog encoding happens inline
    during dataset load, so the per-dataset matrix is materialized
    before the server reports ``status=ok``. Construction failures are
    logged but non-fatal — the runtime falls back to zero-init.
    """
    try:
        config: SymptomsConfig = load_symptoms_config()
    except FileNotFoundError:
        # Documented kill switch — missing config means feature is off.
        logger.info("symptoms config missing; server starting with no datasets")
        SERVER_STATE.config_loaded = True
        return
    device = resolve_device("auto")
    SERVER_STATE.init_matcher_model = _maybe_build_init_matcher(config)
    for spec in config.datasets:
        if not spec.enabled:
            continue
        try:
            loaded = build_dataset(
                spec,
                config.models,
                device=device,
                init_matcher=SERVER_STATE.init_matcher_model,
            )
        except (UnknownDatasetError, FileNotFoundError, RuntimeError):
            logger.exception("failed to load dataset %s", spec.id)
            raise
        SERVER_STATE.datasets[spec.id] = loaded
        logger.info(
            "loaded dataset %s with models %s (eval: %s; init_catalog=%s)",
            spec.id,
            sorted(loaded.models),
            {mid: m.manifest.get("eval", {}) for mid, m in loaded.models.items()},
            (
                f"{len(loaded.init_catalog.candidate_idx)} candidates"
                if loaded.init_catalog is not None
                else "disabled"
            ),
        )
    SERVER_STATE.config_loaded = True


def _maybe_build_init_matcher(config: SymptomsConfig) -> InitMatcherEmbedder | None:
    """Construct the shared init-matcher embedder when enabled.

    Failure here is non-fatal: a missing sentence-transformers package
    or a model-id typo logs and returns ``None``; downstream code sees
    no catalog and skips matching. The single thing that ARE fatal are
    config-validation errors, which Pydantic raises before this runs.
    """
    cfg = config.init_matcher
    if not cfg.enabled:
        logger.info("init-matcher disabled by config")
        return None
    try:
        return InitMatcherEmbedder(
            model_id=cfg.model_id,
            device=cfg.device,
            default_threshold=cfg.threshold,
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "init-matcher construction failed for %s; matching disabled",
            cfg.model_id,
        )
        return None


async def _ttl_purge_loop() -> None:
    """Background coroutine that purges expired sessions periodically."""
    while True:
        await asyncio.sleep(TTL_PURGE_INTERVAL_S)
        try:
            purged = prune_expired_sessions()
        except Exception:  # noqa: BLE001 — never let purge kill the server
            logger.exception("session purge loop failed")
            continue
        if purged:
            logger.info("purged %d expired symptoms session(s)", purged)


app = FastAPI(title="claritymed-symptoms-server", lifespan=lifespan)
add_logging_middleware(app, server_logger=logger)


# --- error envelope -------------------------------------------------------


def _error_payload(
    code: str, message: str, *, request_id: str | None = None, **details: Any
) -> dict[str, Any]:
    """Build the uniform ``{"error": {...}}`` envelope body.

    Mirrors :func:`vision/app.py:_error_payload` byte-for-byte so a
    single client error parser works against vision, symptoms, and
    medical-clip responses.
    """
    payload: dict[str, Any] = {"code": code, "message": message}
    if request_id:
        payload["request_id"] = request_id
    if details:
        payload["details"] = details
    return {"error": payload}


def _default_code(status_code: int) -> str:
    return {
        400: "bad_request",
        404: "not_found",
        413: "payload_too_large",
        422: "unprocessable_entity",
        500: "inference_failed",
        503: "service_unavailable",
    }.get(status_code, "error")


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Wrap every HTTPException response in the standard envelope.

    Accepts both legacy bare-string ``detail`` (wraps with a default
    code derived from the status) and a pre-built ``{"error": {...}}``
    dict (used as-is, with ``request_id`` enriched from the request
    headers when the dict didn't carry one).
    """
    request_id = request.headers.get("X-Request-ID")
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        body = detail
        envelope = body.get("error", {})
        if request_id and isinstance(envelope, dict) and "request_id" not in envelope:
            envelope["request_id"] = request_id
    else:
        body = _error_payload(
            code=_default_code(exc.status_code),
            message=str(detail) if detail else _default_code(exc.status_code),
            request_id=request_id,
        )
    return JSONResponse(status_code=exc.status_code, content=body)


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Normalize Pydantic 422s to 400 + the standard envelope.

    Matches vision/medical-clip: a single client error parser keyed on
    ``error.code`` handles all three servers without per-server
    branching on FastAPI's stock 422 shape.
    """
    return JSONResponse(
        status_code=400,
        content=_error_payload(
            code="bad_request",
            message="request body failed validation",
            request_id=request.headers.get("X-Request-ID"),
            errors=exc.errors(),
        ),
    )


# --- helpers --------------------------------------------------------------


def _require_dataset(dataset_id: str) -> LoadedDataset:
    """Return the loaded dataset or raise HTTP 404 if not found."""
    ds = SERVER_STATE.datasets.get(dataset_id)
    if ds is None:
        raise HTTPException(
            status_code=404,
            detail=_error_payload(
                code="dataset_not_loaded",
                message=f"dataset {dataset_id!r} not loaded",
                available=sorted(SERVER_STATE.datasets),
            ),
        )
    return ds


def _require_session(session_id: str) -> SubSessionState:
    """Return the active session state or raise HTTP 404 if expired or unknown."""
    sub = SERVER_STATE.sessions.get(session_id)
    if sub is None:
        raise HTTPException(
            status_code=404,
            detail=_error_payload(
                code="session_not_found",
                message="session expired or unknown",
            ),
        )
    return sub


def _initial_state(ds: LoadedDataset, age_years: int, sex: str) -> np.ndarray:
    """Build the [1, S+context] state vector with age + sex one-hot set."""
    s_size = ds.canonical.layout["sym_size"]
    context_size = len(AGE_BUCKETS) + len(SEX2IDX)
    state = np.zeros((1, s_size + context_size))
    age_idx = age_bucket(age_years)
    state[0, s_size + age_idx] = 1.0
    sex_idx = SEX2IDX.get(sex, 0)
    state[0, s_size + len(AGE_BUCKETS) + sex_idx] = 1.0
    return state


def _maybe_inject_initial_symptom(
    ds: LoadedDataset, state: np.ndarray, complaint_text: str
) -> int | None:
    """Pre-reveal the matched chief-complaint evidence on turn 0.

    Mirrors the training-time mechanism: ``typed_basd.py:247`` writes
    ``Patient.init`` into the state at batch initialization, so the
    BASD agent learned to make turn-0 decisions conditioned on one
    evidence already being present. Skipping this step at runtime
    leaves the agent in a state distribution it never saw during
    training — measurable as a turn-0 ``next_action`` drift.

    Returns the matched evidence idx (for audit) or ``None`` when no
    match was injected. All failure modes — no matcher, no catalog,
    sub-threshold score, encode failure — converge on ``None`` and
    leave ``state`` untouched.
    """
    if _is_mock_enabled():
        return None
    matcher = SERVER_STATE.init_matcher_model
    if matcher is None or ds.init_catalog is None:
        return None
    text = (complaint_text or "").strip()
    if not text:
        return None
    result = matcher.match(text, ds.init_catalog)
    if result.evidence_idx is None:
        logger.debug(
            "init-matcher: no injection (score=%.3f < threshold=%.2f) text_len=%d",
            result.score,
            ds.init_catalog.threshold,
            len(text),
        )
        return None
    ev = ds.canonical.evidence_by_idx(result.evidence_idx)
    # B-only candidate pool (enforced upstream by InitSymptomFilter)
    # → ``bin_pos`` is the right write payload. Asserting here keeps
    # the contract crisp if a future filter relaxation accidentally
    # lets a C / M slip through.
    if ev.dtype != "B":
        logger.warning(
            "init-matcher matched non-B evidence %s (dtype=%s); skipping "
            "injection to avoid malformed state write",
            ev.id,
            ev.dtype,
        )
        return None
    env = _writer_env(ds)
    env._write(
        state[0],
        result.evidence_idx,
        {"bin_pos": {result.evidence_idx}, "cat_val": {}, "multi_val": {}},
    )
    logger.info(
        "init-matcher injected %s (idx=%d, score=%.3f) as turn-0 evidence",
        ev.id,
        result.evidence_idx,
        result.score,
    )
    return result.evidence_idx


def _writer_env(ds: LoadedDataset) -> TypedEnv:
    """Empty-patient TypedEnv used purely as a state-encoder.

    Per-call instantiation is fine — TypedEnv's __init__ only stores
    references into the schema dict; no allocations beyond a 0-element
    numpy order array.
    """
    return TypedEnv([], ds.canonical.layout, ds.canonical.n_conditions)


def _apply_answer(
    ds: LoadedDataset,
    sub: SubSessionState,
    payload: TurnRequest,
) -> None:
    """Mutate ``sub.state`` to encode the user's answer for ``sub.last_ev_idx``."""
    if sub.last_ev_idx is None:
        raise HTTPException(
            status_code=400,
            detail=_error_payload(
                code="no_pending_question",
                message="session has no pending question to answer",
            ),
        )
    try:
        synth = synth_patient(
            ds.canonical,
            ds.spec,
            sub.last_ev_idx,
            payload.answer,
            answer_value=payload.answer_value,
            language=payload.language,
        )
    except QuestionPayloadError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=_error_payload(
                code="question_payload_error",
                message=str(exc.detail),
            ),
        ) from None
    env = _writer_env(ds)
    env._write(sub.state[0], sub.last_ev_idx, synth)
    ev = ds.canonical.evidence_by_idx(sub.last_ev_idx)
    sub.evidence_collected.append(
        {
            "evidence_id": ev.id,
            "evidence_name": ev.id,
            "evidence_type": ev.dtype,
            "answer": payload.answer,
        }
    )
    sub.turn_count += 1


def _try_render_question(
    ds: LoadedDataset, sub: SubSessionState, ev_idx: int
) -> tuple[Any, int]:
    """Render the question for ``ev_idx``, translating QuestionPayloadError."""
    try:
        return build_question(
            ds.canonical, ds.spec, ev_idx, language=sub.language
        ), ev_idx
    except QuestionPayloadError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=_error_payload(
                code="question_payload_error",
                message=str(exc.detail),
            ),
        ) from None


def _diagnose(ds: LoadedDataset, sub: SubSessionState) -> np.ndarray:
    """Run the model's pathology head and return the [n_dis] prob vector."""
    model = ds.select_model()
    _, probs = model.agent.diagnose(sub.state)
    flush_mps_cache()
    return probs


# Number of top classes emitted in the ``probs=`` field of the session-exit
# INFO logs. Three matches the v3 subset-parametric layout (Pne / Inf /
# Other) and stays tolerable for larger legacy datasets — enough to spot-
# check decision correctness without turning the log line into a wall of
# noise. Grep-friendly format is ``id:prob,id:prob,id:prob``.
_LOG_PROBS_TOPK = 3


def _format_probs_topk(ds: LoadedDataset, probs: np.ndarray) -> str:
    """Render the top-K posterior as a grep-friendly ``id:prob`` list.

    Used in the ``session done`` / ``session cap`` / ``session cancelled``
    INFO lines so operators can eyeball the model's final call against a
    known-good Pneumonia / Influenza sample. v3 subset-parametric
    datasets emit a synthetic trailing ``Other`` class with no canonical
    entry — rendered as the literal ``other`` slug. Legacy datasets
    resolve every index through :meth:`condition_by_idx`.
    """
    if probs.ndim == 2:
        probs = probs[0]
    targets = ds.spec.target_condition_ids
    order = np.argsort(-probs)[:_LOG_PROBS_TOPK]
    parts: list[str] = []
    for i in order:
        i = int(i)
        if targets is not None:
            name = targets[i] if i < len(targets) else "other"
        else:
            try:
                name = ds.canonical.condition_by_idx(i).id
            except (KeyError, IndexError):
                name = f"idx{i}"
        parts.append(f"{name}:{float(probs[i]):.3f}")
    return ",".join(parts)


# --- routes ---------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Per-process readiness — lists loaded datasets and the model ids each carries."""
    status = "ok" if SERVER_STATE.config_loaded else "loading"
    dataset_ids = sorted(SERVER_STATE.datasets)
    model_ids = sorted(
        {mid for ds in SERVER_STATE.datasets.values() for mid in ds.models}
    )
    return HealthResponse(
        status=status, datasets_loaded=dataset_ids, models_loaded=model_ids
    )


router = APIRouter(prefix="/v1/datasets/{dataset_id}")


@router.post("/sessions", response_model=StartSessionResponse)
def start_session(
    dataset_id: str, req: StartSessionRequest, http_req: Request
) -> StartSessionResponse:
    """Initialize a sub-session — return the first question.

    Init-symptom injection: when ``req.symptom_summary`` (preferred) or
    ``req.complaint`` (fallback) matches a catalog candidate above the
    dataset's threshold, that evidence is pre-revealed in the state
    vector before the first ``next_action`` call. This mirrors the
    training-time ``Patient.init`` mechanism — see
    :func:`_maybe_inject_initial_symptom` for the train/serve skew
    rationale.
    """
    ds = _require_dataset(dataset_id)
    state = _initial_state(ds, req.profile.age_years, req.profile.sex)
    matcher_text = req.symptom_summary or req.complaint
    initial_evidence_idx = _maybe_inject_initial_symptom(ds, state, matcher_text)
    model = ds.select_model()
    first_ev_idx = int(model.agent.next_action(state)[0])
    flush_mps_cache()
    question, ev_idx = _try_render_question(
        ds,
        SubSessionState(
            session_id="-",
            dataset_id=dataset_id,
            model_id=model.spec.id,
            state=state,
            turn_count=0,
            started_at=0.0,
            last_ev_idx=first_ev_idx,
            language=req.language,
        ),
        first_ev_idx,
    )
    session_id = uuid.uuid4().hex
    sub = SubSessionState(
        session_id=session_id,
        dataset_id=dataset_id,
        model_id=model.spec.id,
        state=state,
        turn_count=0,
        started_at=time.time(),
        last_ev_idx=ev_idx,
        language=req.language,
        profile={"age_years": req.profile.age_years, "sex": req.profile.sex},
    )
    # Record the injected evidence in the trail so audit + final
    # payload reflect it (training does the same — init counts as a
    # collected evidence even though the user didn't answer a
    # question for it).
    if initial_evidence_idx is not None:
        init_ev = ds.canonical.evidence_by_idx(initial_evidence_idx)
        sub.evidence_collected.append(
            {
                "evidence_id": init_ev.id,
                "evidence_name": init_ev.id,
                "evidence_type": init_ev.dtype,
                "answer": "Yes",
                "source": "init_matcher",
            }
        )
    SERVER_STATE.sessions[session_id] = sub
    req_id = http_req.headers.get("X-Request-ID", "")
    first_ev = ds.canonical.evidence_by_idx(first_ev_idx)
    logger.info(
        "session started: session=%s dataset=%s first_evidence=%s "
        "init_matcher=%s req_id=%s",
        session_id,
        dataset_id,
        first_ev.id,
        initial_evidence_idx is not None,
        req_id or "-",
    )
    return StartSessionResponse(session_id=session_id, first_question=question)


@router.post("/sessions/{session_id}/turn", response_model=TurnResponse)
def turn(
    dataset_id: str, session_id: str, req: TurnRequest, http_req: Request
) -> TurnResponse:
    """Apply an answer to the running session; return the next question or done."""
    ds = _require_dataset(dataset_id)
    sub = _require_session(session_id)
    if sub.dataset_id != dataset_id:
        raise HTTPException(
            status_code=404,
            detail=_error_payload(
                code="session_dataset_mismatch",
                message="session does not belong to this dataset",
                expected_dataset_id=sub.dataset_id,
                requested_dataset_id=dataset_id,
            ),
        )
    # Update language for late turns where the plugin's locale changed.
    sub.language = req.language
    _apply_answer(ds, sub, req)
    model = ds.model(sub.model_id)
    req_id = http_req.headers.get("X-Request-ID", "")

    # Stop gate first — same order as the demo's interactive_eval.
    stop = model.agent.should_stop(sub.state)
    flush_mps_cache()
    if bool(stop[0]):
        probs = _diagnose(ds, sub)
        diff, evidence_rows = format_differential(ds, sub, probs)
        SERVER_STATE.sessions.pop(session_id, None)
        logger.info(
            "session done: session=%s turn=%d differential=%d probs=%s req_id=%s",
            session_id,
            sub.turn_count,
            len(diff),
            _format_probs_topk(ds, probs),
            req_id or "-",
        )
        return TurnResponse(
            done=True,
            differential=diff,
            evidence_collected=evidence_rows,
            turn_count=sub.turn_count,
        )

    # Cap hit?
    if sub.turn_count >= model.spec.maxstep:
        probs = _diagnose(ds, sub)
        outcome = format_cancel_outcome(ds, sub, probs)
        SERVER_STATE.sessions.pop(session_id, None)
        logger.info(
            "session cap: session=%s turn=%d confidence=%.3f probs=%s req_id=%s",
            session_id,
            outcome["turn_count"],
            outcome["partial_confidence"],
            _format_probs_topk(ds, probs),
            req_id or "-",
        )
        return TurnResponse(
            hit_cap=True,
            partial_differential=outcome["partial_differential"],
            evidence_collected=outcome["evidence_collected"],
            turn_count=outcome["turn_count"],
            partial_confidence=outcome["partial_confidence"],
        )

    next_ev_idx = int(model.agent.next_action(sub.state)[0])
    flush_mps_cache()
    question, ev_idx = _try_render_question(ds, sub, next_ev_idx)
    sub.last_ev_idx = ev_idx
    next_ev = ds.canonical.evidence_by_idx(ev_idx)
    # Upgraded from DEBUG so every asked question lands in the standard
    # INFO stream — needed for the "sample a Pneumonia/Influenza case,
    # verify the question sequence" spot-check workflow. One line per
    # turn per active session; volume stays bounded by maxstep.
    logger.info(
        "session turn %d: session=%s next_evidence=%s req_id=%s",
        sub.turn_count,
        session_id,
        next_ev.id,
        req_id or "-",
    )
    return TurnResponse(next_question=question, turn_count=sub.turn_count)


@router.delete("/sessions/{session_id}", response_model=CancelResponse)
def cancel_session(
    dataset_id: str, session_id: str, http_req: Request
) -> CancelResponse:
    """Cancel mid-loop; return partial-result outcome (may be empty)."""
    ds = _require_dataset(dataset_id)
    sub = _require_session(session_id)
    if sub.dataset_id != dataset_id:
        raise HTTPException(
            status_code=404,
            detail=_error_payload(
                code="session_dataset_mismatch",
                message="session does not belong to this dataset",
                expected_dataset_id=sub.dataset_id,
                requested_dataset_id=dataset_id,
            ),
        )
    probs = _diagnose(ds, sub)
    outcome = format_cancel_outcome(ds, sub, probs)
    SERVER_STATE.sessions.pop(session_id, None)
    req_id = http_req.headers.get("X-Request-ID", "")
    logger.info(
        "session cancelled: session=%s dataset=%s turn=%d "
        "confidence=%.3f probs=%s threshold_met=%s severity_override=%s req_id=%s",
        session_id,
        dataset_id,
        outcome["turn_count"],
        outcome["partial_confidence"],
        _format_probs_topk(ds, probs),
        outcome["meets_confidence_threshold"],
        outcome["severity_override"],
        req_id or "-",
    )
    return CancelResponse(
        cancelled=True,
        partial_differential=outcome["partial_differential"],
        evidence_collected=outcome["evidence_collected"],
        turn_count=outcome["turn_count"],
        partial_confidence=outcome["partial_confidence"],
        meets_confidence_threshold=outcome["meets_confidence_threshold"],
        severity_override=outcome["severity_override"],
        max_low_severity_seen=outcome["max_low_severity_seen"],
    )


app.include_router(router)


# --- entry point ----------------------------------------------------------


def main() -> None:
    """Boot the server. The HOST constant guard runs pre-bind."""
    port = int(os.environ.get(SYMPTOMS_PORT_ENV, DEFAULT_PORT))
    if HOST != "127.0.0.1":  # pragma: no cover — guard against accidental edit
        raise RuntimeError(
            f"refusing to bind {HOST!r}: symptoms server must be loopback-only"
        )
    uvicorn.run(app, host=HOST, port=port, log_level="info", log_config=LOG_CONFIG)


if __name__ == "__main__":
    main()
