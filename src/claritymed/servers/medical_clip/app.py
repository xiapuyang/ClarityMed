"""FastAPI app for the medical-clip server (BiomedCLIP zero-shot).

PHI-bearing surface — the user's image bytes traverse this server, so
the loopback constraint (KTD-V8) is enforced at two layers:

1. ``HOST`` is a module-level constant; ``main`` asserts against it
   pre-bind so an accidental env override cannot expose the port.
2. The shipped ``configs/medical_clip.yaml::server.base_url`` is
   ``127.0.0.1:8086``; routing infrastructure that would advertise the
   server outside loopback would have to bypass the config.

The endpoints handler stays thin: decode the image, hand off to the
engine, apply the config-driven gating (``min_confidence`` /
``min_medical_confidence`` — the latter is a per-modality dict with a
``default`` fallback so calibration drift on one label doesn't
require touching the others), build the wire response. All real work
lives on the engine so unit tests can stub it via the skip-load env.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml

from claritymed.servers._devices import (
    LOG_CONFIG,
    add_logging_middleware,
    default_device,
)

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
except ImportError as exc:  # pragma: no cover — import-time guard
    raise SystemExit(
        "claritymed-medical-clip-server requires the 'medical-clip-server' "
        "extra. Install with:\n"
        "    uv sync --extra medical-clip-server\n"
        f"(original ImportError: {exc})"
    ) from None

from claritymed.config import CONFIGS_DIR
from claritymed.core.medical_clip.schemas import (
    HealthResponse,
    Modality,
    ModalityRequest,
    ModalityResponse,
    ModalityScore,
)
from claritymed.servers.medical_clip.biomed_clip import (
    BiomedClipEngine,
    ImageDecodeError,
    ModalityCandidate,
)

logger = logging.getLogger("claritymed.servers.medical_clip")

HOST = "127.0.0.1"
DEFAULT_PORT = 8086
MEDICAL_CLIP_PORT_ENV = "CLARITYMED_MEDICAL_CLIP_PORT"
SKIP_LOAD_ENV = "CLARITYMED_MEDICAL_CLIP_SKIP_LOAD"

# Labels classified as `is_medical=true` only when the gate's
# per-modality `min_medical_confidence[label]` is also exceeded. Photo
# / document / unknown never count as medical regardless of score.
# Kept here (not in YAML) because the rule is structural — flipping a
# label here means rewriting the downstream attachment-ingest semantics.
_NON_MEDICAL_LABELS: frozenset[Modality] = frozenset({"photo", "document", "unknown"})

# Hard cap on per-modality `min_medical_confidence` entries. Bench
# numbers on clean public datasets (data/bench/modality_classifier/)
# routinely exceed 0.90 for ct/xray/histopath, but locking real-world
# noisier uploads out at >0.70 is the wrong failure mode for a
# fail-safe gate. Per-modality calibration can *loosen* the floor
# below 0.70 (ultrasound is the standing example) but never tighten
# it above. Raising the cap requires a documented argument, not a
# YAML edit.
_MIN_MEDICAL_CONFIDENCE_CAP: float = 0.70
# Required key inside the `min_medical_confidence` dict that covers
# any modality not explicitly listed (including hypothetical future
# additions to the Modality vocabulary).
_DEFAULT_KEY: str = "default"


# --- module state ---------------------------------------------------------
#
# Mutable engine slot keeps the FastAPI app importable without a live
# model; tests poke ``_state["engine"]`` directly with a stub.


_state: dict[str, Any] = {
    "engine": None,
    "started_at": 0.0,
    "tasks_loaded": [],
    "gating": {
        "min_confidence": 0.55,
        # Per-modality dict — `default` is required, other keys are
        # optional per-Modality overrides. Same shape after lifespan
        # config load.
        "min_medical_confidence": {_DEFAULT_KEY: 0.70},
    },
}


# --- lifespan -------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001 — FastAPI signature
    """Read configs/medical_clip.yaml and load BiomedCLIP at lifespan.

    Skipped silently when ``CLARITYMED_MEDICAL_CLIP_SKIP_LOAD=1`` so
    tests can pre-populate ``_state["engine"]`` with a stub.
    """
    _state["started_at"] = time.monotonic()
    if os.environ.get(SKIP_LOAD_ENV) == "1":
        logger.info("medical-clip lifespan: skip-load env set; no engine loaded")
        yield
        return
    _load_engine_sync()
    yield
    _state["engine"] = None


def _load_engine_sync() -> None:
    """Blocking BiomedCLIP load — called from lifespan."""
    cfg = _read_config()
    model_id = cfg["model"]["model_id"]
    revision = (cfg["model"].get("revision") or "").strip() or None
    device_request = cfg["model"].get("device", "auto")
    device = default_device() if device_request == "auto" else device_request

    modality_cfg = cfg["tasks"]["modality"]
    candidates = [
        ModalityCandidate(label=entry["label"], prompts=tuple(entry["prompts"]))
        for entry in modality_cfg["candidates"]
    ]
    gating = modality_cfg["gating"]
    _validate_gating(gating)
    _state["gating"] = {
        "min_confidence": float(gating["min_confidence"]),
        "min_medical_confidence": {
            key: float(value) for key, value in gating["min_medical_confidence"].items()
        },
    }
    _state["tasks_loaded"] = list(cfg["tasks"])

    engine = BiomedClipEngine.load(
        model_id=model_id,
        revision=revision,
        device=device,
        candidates=candidates,
    )
    _state["engine"] = engine
    logger.info(
        "medical-clip engine ready: model=%s revision=%s device=%s "
        "candidates=%d min_conf=%.2f min_medical=%s",
        engine.model_id,
        engine.model_revision or "unpinned",
        engine.device,
        len(candidates),
        _state["gating"]["min_confidence"],
        # Sorted for deterministic logs across reloads — operators
        # grep the startup line to confirm calibration changes shipped.
        sorted(_state["gating"]["min_medical_confidence"].items()),
    )


def _read_config() -> dict[str, Any]:
    path = Path(CONFIGS_DIR) / "medical_clip.yaml"
    if not path.exists():
        raise RuntimeError(
            f"configs/medical_clip.yaml missing at {path}; the "
            "medical-clip server cannot start without it"
        )
    return yaml.safe_load(path.read_text()) or {}


def _validate_gating(gating: dict[str, Any]) -> None:
    """Range-check the modality gate thresholds.

    ``min_confidence`` is a scalar in (0, 1). ``min_medical_confidence``
    is a dict with a required ``default`` key plus optional
    per-Modality overrides; every entry must be in (0, 1) and at or
    below :data:`_MIN_MEDICAL_CONFIDENCE_CAP`. The cap is the
    project-level guard against shipping bench-clean numbers to
    real-world data — see the constant's comment for rationale.
    """
    min_conf = float(gating["min_confidence"])
    if not (0.0 < min_conf < 1.0):
        raise RuntimeError(
            f"configs/medical_clip.yaml::tasks.modality.gating.min_confidence "
            f"= {min_conf!r} is out of (0, 1)"
        )

    medical = gating["min_medical_confidence"]
    if not isinstance(medical, dict):
        raise RuntimeError(
            f"configs/medical_clip.yaml::tasks.modality.gating.min_medical_confidence "
            f"must be a dict (got {type(medical).__name__}); the scalar shape was "
            f"replaced by a per-modality mapping when per-modality calibration shipped"
        )
    if _DEFAULT_KEY not in medical:
        raise RuntimeError(
            f"configs/medical_clip.yaml::tasks.modality.gating.min_medical_confidence "
            f"is missing the required {_DEFAULT_KEY!r} key — every modality not "
            f"listed explicitly falls through to this entry"
        )
    for key, raw in medical.items():
        value = float(raw)
        if not (0.0 < value < 1.0):
            raise RuntimeError(
                f"configs/medical_clip.yaml::tasks.modality.gating.min_medical_confidence"
                f"[{key!r}] = {value!r} is out of (0, 1)"
            )
        if value > _MIN_MEDICAL_CONFIDENCE_CAP:
            raise RuntimeError(
                f"configs/medical_clip.yaml::tasks.modality.gating.min_medical_confidence"
                f"[{key!r}] = {value!r} exceeds the project cap of "
                f"{_MIN_MEDICAL_CONFIDENCE_CAP}; tightening above the cap would "
                f"reject too many real-world uploads — calibration should loosen, "
                f"not tighten, the floor"
            )


# --- app + error envelope -------------------------------------------------


app = FastAPI(title="claritymed-medical-clip-server", lifespan=lifespan)
add_logging_middleware(app, server_logger=logger)


def _error_payload(
    code: str, message: str, *, request_id: str | None = None, **details: Any
) -> dict[str, Any]:
    """Build the uniform error envelope body."""
    payload: dict[str, Any] = {"code": code, "message": message}
    if request_id:
        payload["request_id"] = request_id
    if details:
        payload["details"] = details
    return {"error": payload}


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Wrap HTTPException so every 4xx/5xx response matches the envelope."""
    # When the handler explicitly raised with a dict detail we trust it
    # — it's already a {"error": ...} envelope. Otherwise build one.
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        body = detail
    else:
        body = _error_payload(
            code=_default_code(exc.status_code),
            message=str(detail) if detail else _default_code(exc.status_code),
            request_id=request.headers.get("X-Request-ID"),
        )
    return JSONResponse(status_code=exc.status_code, content=body)


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Normalize Pydantic validation 422s to 400 + the standard envelope."""
    return JSONResponse(
        status_code=400,
        content=_error_payload(
            code="bad_request",
            message="request body failed validation",
            request_id=request.headers.get("X-Request-ID"),
            errors=exc.errors(),
        ),
    )


def _default_code(status_code: int) -> str:
    return {
        400: "bad_request",
        404: "not_found",
        413: "payload_too_large",
        422: "bad_request",
        503: "service_unavailable",
    }.get(status_code, "error")


# --- routes ---------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Process readiness + the loaded BiomedCLIP fingerprint."""
    engine = _state.get("engine")
    if engine is None:
        return HealthResponse(
            status="loading",
            model_id="(none)",
            model_revision=None,
            tasks_loaded=[],
            device="-",
            uptime_s=_uptime_s(),
        )
    return HealthResponse(
        status="ok",
        model_id=engine.model_id,
        model_revision=engine.model_revision,
        tasks_loaded=list(_state["tasks_loaded"]),
        device=engine.device,
        uptime_s=_uptime_s(),
    )


@app.post("/v1/classify_modality", response_model=ModalityResponse)
def classify_modality(req: ModalityRequest, http_req: Request) -> ModalityResponse:
    """Zero-shot modality classification on one image.

    Gating rules (mirrors origin §5.5):

    * ``top1_score < min_confidence`` → ``modality="unknown"`` so the
      downstream LLM routes to askuserquestion instead of guessing.
    * ``top1`` in ``photo`` / ``document`` / ``unknown`` → ``is_medical=false``
      regardless of score.
    * Otherwise ``is_medical = top1_score >= min_medical_confidence[top1.label]``,
      with the per-modality dict falling back to its ``default`` entry
      when the label is not listed explicitly.
    """
    engine = _state.get("engine")
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail=_error_payload(
                code="service_unavailable",
                message="BiomedCLIP engine not loaded",
                request_id=req.request_id,
            ),
        )

    image_bytes = _decode_b64(req.image.data_b64, req.request_id)
    _verify_sha256(image_bytes, req.image.sha256, req.request_id)

    t0 = time.monotonic()
    try:
        ranked = engine.classify(image_bytes)
    except ImageDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=_error_payload(
                code="image_decode_failed",
                message=str(exc),
                request_id=req.request_id,
            ),
        ) from exc
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    top1 = ranked[0]
    gating = _state["gating"]
    final_label, is_medical = _apply_gating(top1, gating)
    response = ModalityResponse(
        request_id=req.request_id,
        modality=final_label,
        confidence=top1.score,
        is_medical=is_medical,
        scores=_canonical_score_order(ranked),
        elapsed_ms=elapsed_ms,
    )
    logger.debug(
        "classify_modality req=%s top1=%s top1_score=%.3f modality=%s "
        "is_medical=%s elapsed_ms=%d hdr_req=%s",
        req.request_id,
        top1.label,
        top1.score,
        final_label,
        is_medical,
        elapsed_ms,
        http_req.headers.get("X-Request-ID", "-"),
    )
    return response


# --- helpers --------------------------------------------------------------


def _decode_b64(data_b64: str, request_id: str) -> bytes:
    try:
        return base64.b64decode(data_b64, validate=True)
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=_error_payload(
                code="image_decode_failed",
                message=f"image.data_b64 is not valid base64: {exc!s}",
                request_id=request_id,
            ),
        ) from exc


def _verify_sha256(image_bytes: bytes, claimed_sha: str, request_id: str) -> None:
    actual = hashlib.sha256(image_bytes).hexdigest()
    if actual != claimed_sha:
        raise HTTPException(
            status_code=400,
            detail=_error_payload(
                code="image_hash_mismatch",
                message=(
                    "image.data_b64 decoded sha256 does not match "
                    "image.sha256 — refusing to bypass the attachment-"
                    "ingest modality tag"
                ),
                request_id=request_id,
                claimed=claimed_sha,
                actual=actual,
            ),
        )


def _apply_gating(top1: ModalityScore, gating: dict[str, Any]) -> tuple[Modality, bool]:
    """Return (effective_label, is_medical) given the top1 score + gating thresholds.

    ``min_medical_confidence`` is a per-modality dict — look up the
    specific label, fall back to the ``default`` entry (validator
    ensures it exists). The lookup is only reached for medical labels
    since :data:`_NON_MEDICAL_LABELS` short-circuits above; the
    ``default`` entry therefore only ever applies to medical labels
    omitted from the YAML (typically a future-added Modality).
    """
    if top1.score < gating["min_confidence"]:
        return "unknown", False
    if top1.label in _NON_MEDICAL_LABELS:
        return top1.label, False
    per_modality: dict[str, float] = gating["min_medical_confidence"]
    threshold = per_modality.get(top1.label, per_modality[_DEFAULT_KEY])
    is_medical = top1.score >= threshold
    return top1.label, is_medical


def _canonical_score_order(ranked: list[ModalityScore]) -> list[ModalityScore]:
    """Wire scores stay in descending-score order from the engine.

    Kept as a named helper because the brainstorm spec was explicit
    about the order ("scoreboard" — descending). A future tweak that
    moves to canonical-label order would change here.
    """
    return list(ranked)


def _uptime_s() -> int:
    return int(max(0.0, time.monotonic() - _state["started_at"]))


# --- entry point ----------------------------------------------------------


def main() -> None:
    """Boot the server. ``HOST`` guard fires pre-bind."""
    port = int(os.environ.get(MEDICAL_CLIP_PORT_ENV, DEFAULT_PORT))
    if HOST != "127.0.0.1":  # pragma: no cover — guard against accidental edit
        raise RuntimeError(
            f"refusing to bind {HOST!r}: medical-clip server must be loopback-only"
        )
    uvicorn.run(app, host=HOST, port=port, log_level="info", log_config=LOG_CONFIG)


if __name__ == "__main__":
    main()
