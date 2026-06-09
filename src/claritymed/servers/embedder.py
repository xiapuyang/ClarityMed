"""BGE-M3 embedding server — TEI-compatible wire format.

Why this exists: ``BgeM3HttpEmbedder`` (``core/rag/embedding/bge_m3.py``)
expects a TEI-shaped HTTP server. TEI's official Docker image on Apple
Silicon hits three blockers in a row (no ARM manifest + ``hf-hub-0.3.2``
URL bug + no ONNX shipped for ``BAAI/bge-m3``). Rather than fight all
three, we host the model ourselves via FlagEmbedding (BAAI's official
library) and expose the exact two endpoints the client needs.

Endpoints:

* ``POST /embed`` → 1024-dim dense vectors (CLS pooling, normalized)
* ``POST /embed_sparse`` → flat ``{token_id: weight}`` maps (M3 sparse head)
* ``GET  /health`` → 200 once the model is loaded

Run::

    uv sync --extra rag-server
    uv run --extra rag-server claritymed-embedder

Defaults: ``MODEL_PATH=~/.claritymed/models/bge-m3``, ``PORT=8082``.
Override via env: ``BGE_M3_MODEL_PATH``, ``BGE_M3_PORT``, ``BGE_M3_DEVICE``
(``cpu`` / ``cuda`` / ``mps``). Default auto-detects: ``mps`` on Apple
Silicon, ``cuda`` on NVIDIA, else ``cpu``. Force ``cpu`` if a specific
FlagEmbedding release exhibits MPS kernel instability for XLMRoberta ops.
"""

from __future__ import annotations

import gc
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from claritymed.servers._devices import default_device

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from FlagEmbedding import BGEM3FlagModel
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover — import-time guard
    raise SystemExit(
        "claritymed-embedder requires the 'rag-server' extra. Install with:\n"
        "    uv sync --extra rag-server\n"
        f"(original ImportError: {exc})"
    ) from None

logger = logging.getLogger("claritymed.servers.embedder")

DEFAULT_MODEL_PATH = Path.home() / ".claritymed" / "models" / "bge-m3"
DEFAULT_PORT = 8082
# Match TEI's payload limit (2 MB) — keeps memory bounded under concurrent load.
MAX_BATCH_TEXTS = 64

_state: dict[str, Any] = {"model": None}


class EmbedRequest(BaseModel):
    """TEI-shaped request: a list of strings under ``inputs``."""

    inputs: list[str] = Field(..., min_length=1)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001 — FastAPI signature
    model_path = Path(os.environ.get("BGE_M3_MODEL_PATH", DEFAULT_MODEL_PATH))
    device = os.environ.get("BGE_M3_DEVICE") or default_device()
    if not model_path.exists():
        raise RuntimeError(
            f"BGE-M3 model dir not found: {model_path}\n"
            f"  Run: uv run hf download BAAI/bge-m3 --local-dir {model_path}"
        )
    logger.info("loading BGE-M3 from %s on %s", model_path, device)
    _state["model"] = BGEM3FlagModel(
        str(model_path),
        use_fp16=device != "cpu",
        devices=device,
    )
    logger.info("BGE-M3 ready")
    yield
    _state["model"] = None


app = FastAPI(title="claritymed-embedder", lifespan=lifespan)


def _require_model() -> BGEM3FlagModel:
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return model


def _validate_batch(inputs: list[str]) -> None:
    if len(inputs) > MAX_BATCH_TEXTS:
        raise HTTPException(
            status_code=413,
            detail=f"batch size {len(inputs)} exceeds {MAX_BATCH_TEXTS}",
        )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok" if _state.get("model") is not None else "loading"}


def _flush_mps_cache() -> None:
    # PyTorch's MPS allocator caches freed tensors in-process and never
    # returns them to the OS unprompted. On Apple Silicon this shows up as
    # ever-growing phys_footprint (seen: 28 GB after two days of idle
    # requests). Calling empty_cache() + gc.collect() after each encode
    # releases the cached pages back to macOS immediately.
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    gc.collect()


@app.post("/embed")
def embed(req: EmbedRequest) -> list[list[float]]:
    """Dense embeddings only. Returns ``list[list[float]]`` (1024-dim each)."""
    _validate_batch(req.inputs)
    model = _require_model()
    out = model.encode(
        req.inputs,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    # FlagEmbedding returns numpy arrays; convert to plain lists for JSON.
    result = [vec.tolist() for vec in out["dense_vecs"]]
    _flush_mps_cache()
    return result


@app.post("/embed_sparse")
def embed_sparse(req: EmbedRequest) -> list[dict[str, float]]:
    """Sparse lexical weights. Returns ``[{token_id: weight}]`` per input.

    Keys are stringified token ids (the BgeM3HttpEmbedder client coerces
    them to int on parse — see ``_coerce_sparse`` shape B). Values are
    the BGE-M3 sparse linear head outputs, already non-negative.
    """
    _validate_batch(req.inputs)
    model = _require_model()
    out = model.encode(
        req.inputs,
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    # ``lexical_weights`` is already a list of ``{str(token_id): float}`` dicts.
    result = [
        {str(tok): float(w) for tok, w in entry.items()}
        for entry in out["lexical_weights"]
    ]
    _flush_mps_cache()
    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    port = int(os.environ.get("BGE_M3_PORT", DEFAULT_PORT))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
