"""FastAPI app for the vision server.

PHI-bearing surface (the user's image bytes traverse this server), so
the loopback constraint (KTD-V8) is enforced at two layers:

1. ``HOST`` is a module-level constant; ``main`` asserts against it
   pre-bind so an accidental env override cannot expose the port.
2. The shipped ``configs/vision.yaml::servers[0].base_url`` is
   ``127.0.0.1:8085``; routing infrastructure that would advertise the
   server outside loopback would have to bypass the config.

Endpoints:

* ``GET /health`` — process readiness + summary of loaded models.
* ``GET /v1/catalog`` — full per-model truth (Unit 5 cross-check target,
  KTD-V2). Each entry comes from the validated ``Manifest`` plus the
  ``ModelSpec`` so the client can spot config-vs-server drift at boot.
* ``POST /v1/detect`` — one image → one ``RawDetection``. Modality is
  cross-checked against the loaded model's ``accepted_modality`` (KTD-V3
  defense-in-depth; the tool body also gates upstream).

The actual model orchestration lives in ``inference.run_inference`` so
the handler stays thin and the KTD-V10 override is testable in isolation.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from claritymed.servers._devices import LOG_CONFIG, default_device

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
except ImportError as exc:  # pragma: no cover — import-time guard
    raise SystemExit(
        "claritymed-vision-server requires the 'vision-server' extra. "
        "Install with:\n    uv sync --extra vision-server\n"
        f"(original ImportError: {exc})"
    ) from None

from claritymed.config import load_vision_config
from claritymed.core.vision.schemas import (
    DiseaseSpec,
    ModelSpec,
    VisionConfig,
)
from claritymed.core.vision.wire import (
    CatalogModel,
    CatalogResponse,
    DetectRequest,
    DetectResponse,
    HealthLoadedModel,
    HealthResponse,
)
from claritymed.servers.vision.inference import InferenceResources, run_inference
from claritymed.servers.vision.loader import load_model_for_spec

logger = logging.getLogger("claritymed.servers.vision")

HOST = "127.0.0.1"
DEFAULT_PORT = (
    8085  # 8082=embedder, 8083=reranker, 8084=symptoms, 8085=vision, 8086=medical-clip
)
VISION_PORT_ENV = "CLARITYMED_VISION_PORT"
SKIP_LOAD_ENV = "CLARITYMED_VISION_SKIP_LOAD"


# --- module state ---------------------------------------------------------
#
# Resources keyed by ``model_id`` (globally unique per the ModelSpec.id
# pattern). Tests with ``CLARITYMED_VISION_SKIP_LOAD=1`` poke this dict
# directly to register stub resources.

_state: dict[str, Any] = {
    "started_at": 0.0,
    "resources": {},  # model_id -> InferenceResources
    "diseases": {},  # disease_id -> DiseaseSpec (lookup for /v1/detect routing)
    "config_loaded": False,
}


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001 — FastAPI signature
    """Load every enabled model's manifest chain into ``_state``."""
    _state["started_at"] = time.monotonic()
    if os.environ.get(SKIP_LOAD_ENV) == "1":
        logger.info("vision lifespan: skip-load env set; no models loaded")
        _state["config_loaded"] = True
        yield
        return
    _load_config_sync()
    yield


def _load_config_sync() -> None:
    """Blocking config + model load. Logs each enabled model loaded.

    Refuses to start (re-raises) on any manifest chain error so a
    tampered checkpoint can't half-load — the operator sees the
    mismatch with file paths + expected/actual hashes immediately.
    """
    try:
        cfg: VisionConfig = load_vision_config()
    except FileNotFoundError:
        # Documented kill switch — missing config means feature is off.
        logger.info("vision config missing; server starting with no models")
        _state["config_loaded"] = True
        return
    device = default_device()
    for disease in cfg.diseases:
        _state["diseases"][disease.id] = disease
        if not disease.enabled:
            logger.info("vision disease %s disabled; skipping", disease.id)
            continue
        for model_id in disease.flow:
            spec = _find_model_spec(cfg.models, model_id)
            try:
                manifest, model = load_model_for_spec(spec, device=device)
            except (FileNotFoundError, RuntimeError):
                logger.exception("failed to load model %s", spec.id)
                raise
            _state["resources"][spec.id] = InferenceResources(
                spec_id=spec.id,
                disease_id=disease.id,
                model=model,
                manifest=manifest,
            )
            logger.info(
                "loaded vision model id=%s disease=%s version=%s "
                "framework=%s device=%s",
                spec.id,
                disease.id,
                manifest.model_version,
                manifest.framework,
                device,
            )
    _state["config_loaded"] = True


def _find_model_spec(models: list[ModelSpec], model_id: str) -> ModelSpec:
    """Locate a model by id or raise — config validation already cross-checked."""
    for spec in models:
        if spec.id == model_id:
            return spec
    # VisionConfig._cross_reference catches this at config load; if we
    # land here, the config validator drifted from runtime expectations.
    raise RuntimeError(
        f"model_id {model_id!r} not in configs/vision.yaml::models — "
        f"config cross-reference is out of sync with disease.flow"
    )


# --- app + error envelope -------------------------------------------------


app = FastAPI(title="claritymed-vision-server", lifespan=lifespan)


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
        422: "modality_mismatch",  # defense-in-depth modality refuse
        500: "inference_failed",
        503: "service_unavailable",
    }.get(status_code, "error")


# --- routes ---------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Per-process readiness + the loaded model summary."""
    status = "ok" if _state["config_loaded"] else "loading"
    loaded = [
        HealthLoadedModel(disease_id=r.disease_id, model_id=r.spec_id)
        for r in _state["resources"].values()
    ]
    return HealthResponse(
        status=status,
        models_loaded=loaded,
        uptime_s=_uptime_s(),
    )


@app.get("/v1/catalog", response_model=CatalogResponse)
def catalog() -> CatalogResponse:
    """Server's truth about every loaded model. Boot-time cross-check target.

    Each entry's ``manifest_sha`` is the sha256 of the on-disk
    ``manifest.json`` — same value the config pins. The client
    (Unit 5 ``VisionRegistry``) compares ``manifest_sha`` field-by-field
    against ``configs/vision.yaml::models`` and refuses to boot the
    orchestrator on disagreement.
    """
    models: list[CatalogModel] = []
    for resources in _state["resources"].values():
        spec = _spec_for_resources(resources)
        manifest = resources.manifest
        models.append(
            CatalogModel(
                disease_id=resources.disease_id,
                model_id=resources.spec_id,
                model_version=manifest.model_version,
                framework=manifest.framework,
                task=manifest.task,
                labels=list(manifest.labels),
                cancer_class=manifest.cancer_class,
                accepted_modality=manifest.accepted_modality,
                manifest_sha=spec.manifest_sha256,
                expected_ms=spec.expected_ms,
                supports_saliency=manifest.supports_saliency,
                supports_tta=manifest.supports_tta,
            )
        )
    return CatalogResponse(models=models)


@app.post("/v1/detect", response_model=DetectResponse)
def detect(
    req: DetectRequest, http_req: Request, http_resp: Response
) -> DetectResponse:
    """Run one image through the model resolved by (disease_id, model_id).

    Status-code map (mirrors plan §"Status code map"):

    * 400 — corrupted image bytes / sha mismatch.
    * 404 — unknown disease or unknown model for known disease.
    * 422 — modality mismatch (defense-in-depth catch).
    * 500 — adapter raised; ``inference_failed`` code.
    * 503 — lifespan didn't complete (should have crashed earlier).
    """
    if not _state["config_loaded"]:
        raise HTTPException(
            status_code=503,
            detail=_error_payload(
                code="service_unavailable",
                message="vision lifespan did not complete",
                request_id=req.request_id,
            ),
        )
    # Echo X-Request-ID before any branching so the client always gets it.
    http_resp.headers["X-Request-ID"] = req.request_id

    disease = _require_disease(req.disease_id, req.request_id)
    resources = _resolve_model(req, disease)
    image_bytes = _decode_b64(req.image.data_b64, req.request_id)
    _verify_sha256(image_bytes, req.image.sha256, req.request_id)

    try:
        return run_inference(
            request=req,
            image_bytes=image_bytes,
            resources=resources,
        )
    except Exception as exc:  # noqa: BLE001 — surface as 500
        logger.exception("vision inference failed: req=%s", req.request_id)
        raise HTTPException(
            status_code=500,
            detail=_error_payload(
                code="inference_failed",
                message=f"{type(exc).__name__}: {exc!s}",
                request_id=req.request_id,
            ),
        ) from exc


# --- routing helpers ------------------------------------------------------


def _require_disease(disease_id: str, request_id: str) -> DiseaseSpec:
    disease = _state["diseases"].get(disease_id)
    if disease is None:
        raise HTTPException(
            status_code=404,
            detail=_error_payload(
                code="unknown_disease",
                message=f"disease_id {disease_id!r} not in catalog",
                request_id=request_id,
                available=sorted(_state["diseases"]),
            ),
        )
    if not disease.enabled:
        raise HTTPException(
            status_code=404,
            detail=_error_payload(
                code="unknown_disease",
                message=f"disease_id {disease_id!r} is disabled",
                request_id=request_id,
                available=sorted(
                    d.id for d in _state["diseases"].values() if d.enabled
                ),
            ),
        )
    return disease


def _resolve_model(req: DetectRequest, disease: DiseaseSpec) -> InferenceResources:
    """Look up the loaded resources for the requested (or primary) model."""
    if req.model_id is None:
        target_id = disease.primary_model_id
    elif req.model_id not in disease.flow:
        raise HTTPException(
            status_code=404,
            detail=_error_payload(
                code="unknown_model",
                message=(
                    f"model_id {req.model_id!r} not in disease.flow for {disease.id!r}"
                ),
                request_id=req.request_id,
                available=list(disease.flow),
            ),
        )
    else:
        target_id = req.model_id

    resources = _state["resources"].get(target_id)
    if resources is None:
        raise HTTPException(
            status_code=503,
            detail=_error_payload(
                code="service_unavailable",
                message=f"model {target_id!r} not loaded on this server",
                request_id=req.request_id,
                available=sorted(_state["resources"]),
            ),
        )
    return resources


def _spec_for_resources(resources: InferenceResources) -> ModelSpec:
    """Round-trip the ModelSpec back from disk for the catalog endpoint.

    Cached at boot would be nicer; v1 just re-reads the config each
    catalog hit (one call per orchestrator boot — Unit 5 caches the
    catalog response, so the per-call cost is negligible).
    """
    cfg = load_vision_config()
    return _find_model_spec(cfg.models, resources.spec_id)


# --- image-payload helpers (mirrors medical-clip) -------------------------


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


def _uptime_s() -> int:
    return int(max(0.0, time.monotonic() - _state["started_at"]))


# --- entry point ----------------------------------------------------------


def main() -> None:
    """Boot the server. ``HOST`` guard fires pre-bind."""
    port = int(os.environ.get(VISION_PORT_ENV, DEFAULT_PORT))
    if HOST != "127.0.0.1":  # pragma: no cover — guard against accidental edit
        raise RuntimeError(
            f"refusing to bind {HOST!r}: vision server must be loopback-only"
        )
    uvicorn.run(app, host=HOST, port=port, log_level="info", log_config=LOG_CONFIG)


if __name__ == "__main__":
    main()
