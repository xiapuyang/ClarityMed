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

from claritymed.servers._devices import LOG_CONFIG

try:
    import uvicorn
    from fastapi import APIRouter, FastAPI, HTTPException
except ImportError as exc:  # pragma: no cover — import-time guard
    raise SystemExit(
        "claritymed-symptoms-server requires the 'symptoms-server' extra. "
        "Install with:\n    uv sync --extra symptoms-server\n"
        f"(original ImportError: {exc})"
    ) from None

from claritymed.config import load_symptoms_config
from claritymed.core.device import resolve_device
from claritymed.core.symptoms.datasets import LoadedDataset, build_dataset
from claritymed.core.symptoms.schemas import SymptomsConfig
from claritymed.errors import UnknownDatasetError
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
    """
    if os.environ.get("CLARITYMED_SYMPTOMS_SKIP_LOAD") == "1":
        SERVER_STATE.config_loaded = True
        return
    try:
        config: SymptomsConfig = load_symptoms_config()
    except FileNotFoundError:
        # Documented kill switch — missing config means feature is off.
        logger.info("symptoms config missing; server starting with no datasets")
        SERVER_STATE.config_loaded = True
        return
    device = resolve_device("auto")
    for spec in config.datasets:
        if not spec.enabled:
            continue
        try:
            loaded = build_dataset(spec, config.models, device=device)
        except (UnknownDatasetError, FileNotFoundError, RuntimeError):
            logger.exception("failed to load dataset %s", spec.id)
            raise
        SERVER_STATE.datasets[spec.id] = loaded
        logger.info(
            "loaded dataset %s with models %s (eval: %s)",
            spec.id,
            sorted(loaded.models),
            {mid: m.manifest.get("eval", {}) for mid, m in loaded.models.items()},
        )
    SERVER_STATE.config_loaded = True


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


# --- helpers --------------------------------------------------------------


def _require_dataset(dataset_id: str) -> LoadedDataset:
    ds = SERVER_STATE.datasets.get(dataset_id)
    if ds is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"dataset {dataset_id!r} not loaded; available: "
                f"{sorted(SERVER_STATE.datasets)!r}"
            ),
        )
    return ds


def _require_session(session_id: str) -> SubSessionState:
    sub = SERVER_STATE.sessions.get(session_id)
    if sub is None:
        raise HTTPException(status_code=404, detail="session expired or unknown")
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
            status_code=400, detail="session has no pending question to answer"
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
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from None
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
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from None


def _diagnose(ds: LoadedDataset, sub: SubSessionState) -> np.ndarray:
    """Run the model's pathology head and return the [n_dis] prob vector."""
    model = ds.select_model()
    _, probs = model.agent.diagnose(sub.state)
    return probs


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
def start_session(dataset_id: str, req: StartSessionRequest) -> StartSessionResponse:
    """Initialize a sub-session — return the first question."""
    ds = _require_dataset(dataset_id)
    state = _initial_state(ds, req.profile.age_years, req.profile.sex)
    model = ds.select_model()
    first_ev_idx = int(model.agent.next_action(state)[0])
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
    SERVER_STATE.sessions[session_id] = sub
    return StartSessionResponse(session_id=session_id, first_question=question)


@router.post("/sessions/{session_id}/turn", response_model=TurnResponse)
def turn(dataset_id: str, session_id: str, req: TurnRequest) -> TurnResponse:
    """Apply an answer to the running session; return the next question or done."""
    ds = _require_dataset(dataset_id)
    sub = _require_session(session_id)
    if sub.dataset_id != dataset_id:
        raise HTTPException(
            status_code=404, detail="session does not belong to this dataset"
        )
    # Update language for late turns where the plugin's locale changed.
    sub.language = req.language
    _apply_answer(ds, sub, req)
    model = ds.model(sub.model_id)

    # Stop gate first — same order as the demo's interactive_eval.
    stop = model.agent.should_stop(sub.state)
    if bool(stop[0]):
        probs = _diagnose(ds, sub)
        diff, evidence_rows = format_differential(ds, sub, probs)
        del SERVER_STATE.sessions[session_id]
        return TurnResponse(
            done=True,
            differential=diff,
            evidence_collected=evidence_rows,
            turn_count=sub.turn_count,
        )

    # Cap hit?
    if sub.turn_count >= ds.spec.maxstep:
        probs = _diagnose(ds, sub)
        outcome = format_cancel_outcome(ds, sub, probs)
        del SERVER_STATE.sessions[session_id]
        return TurnResponse(
            hit_cap=True,
            partial_differential=outcome["partial_differential"],
            evidence_collected=outcome["evidence_collected"],
            turn_count=outcome["turn_count"],
            partial_confidence=outcome["partial_confidence"],
        )

    next_ev_idx = int(model.agent.next_action(sub.state)[0])
    question, ev_idx = _try_render_question(ds, sub, next_ev_idx)
    sub.last_ev_idx = ev_idx
    return TurnResponse(next_question=question, turn_count=sub.turn_count)


@router.delete("/sessions/{session_id}", response_model=CancelResponse)
def cancel_session(dataset_id: str, session_id: str) -> CancelResponse:
    """Cancel mid-loop; return partial-result outcome (may be empty)."""
    ds = _require_dataset(dataset_id)
    sub = _require_session(session_id)
    if sub.dataset_id != dataset_id:
        raise HTTPException(
            status_code=404, detail="session does not belong to this dataset"
        )
    probs = _diagnose(ds, sub)
    outcome = format_cancel_outcome(ds, sub, probs)
    del SERVER_STATE.sessions[session_id]
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
