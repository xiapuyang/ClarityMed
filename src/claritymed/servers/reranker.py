"""Cross-encoder reranker server — TEI ``/rerank``-compatible wire format.

Hosts any ``AutoModelForSequenceClassification``-compatible reranker
model behind the same wire shape the ``BgeRerankerV2M3HttpReranker``
client expects::

    POST /rerank
    {"query": "...", "texts": ["d1", ...], "raw_scores": false}
    → [{"index": int, "score": float}, ...]   # sorted desc by score

Why not ``FlagEmbedding.FlagReranker``? Its ``compute_score_single_gpu``
calls ``tokenizer.prepare_for_model(...)``, a method that newer
transformers releases removed from slow tokenizers — and BAAI/bge-
reranker-v2-m3 loads the slow XLM-RoBERTa tokenizer by default. Going
straight to ``AutoModelForSequenceClassification`` skips FlagEmbedding's
inference wrapper entirely (it's a thin ~30 LoC wrapper anyway), works
with both slow + fast tokenizers, and keeps us off that compatibility
fault line.

Run::

    uv sync --extra rag-server
    uv run --extra rag-server claritymed-reranker

Env vars:

* ``BGE_RERANKER_MODEL_PATH`` — local model directory. Defaults to
  ``~/.claritymed/models/bge-reranker-v2-m3``; set this to point at a
  different reranker (e.g. ``bge-reranker-v2-gemma``) without code change.
* ``BGE_RERANKER_QUERY_INSTRUCTION`` — optional prefix prepended to the
  query before forming the cross-encoder input. Empty by default, which
  preserves bge-reranker-v2-m3 behaviour. Gemma-based rerankers expect a
  short instruction prefix here.
* ``BGE_RERANKER_PORT`` — listen port (default 8083).
* ``BGE_RERANKER_DEVICE`` — ``cpu`` / ``cuda`` / ``mps``; auto-detects
  when unset.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from claritymed.servers._devices import (
    LOG_CONFIG,
    add_logging_middleware,
    default_device,
)

try:
    import torch
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request
    from pydantic import BaseModel, Field
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as exc:  # pragma: no cover — import-time guard
    raise SystemExit(
        "claritymed-reranker requires the 'rag-server' extra. Install with:\n"
        "    uv sync --extra rag-server\n"
        f"(original ImportError: {exc})"
    ) from None

logger = logging.getLogger("claritymed.servers.reranker")

DEFAULT_MODEL_PATH = Path.home() / ".claritymed" / "models" / "bge-reranker-v2-m3"
DEFAULT_PORT = 8083
MAX_BATCH_TEXTS = 128
# bge-reranker-v2-m3 was trained at this max length; truncating to it
# matches the upstream serving recipe and is safe for v2-gemma too
# (which trains at 1024 but accepts shorter inputs).
MAX_SEQ_LEN = 512

_state: dict[str, Any] = {
    "model": None,
    "tokenizer": None,
    "device": None,
    "query_instruction": "",
}


def _apply_query_instruction(query: str, instruction: str) -> str:
    """Prepend the optional instruction prefix to the query.

    Empty instruction is a no-op — preserves the bge-reranker-v2-m3
    contract where the model takes the raw query/passage pair. Gemma-
    based rerankers (and other instruction-tuned cross-encoders) expect
    a short prefix here so the operator can configure it without code
    changes.
    """
    if not instruction:
        return query
    return f"{instruction}{query}"


class RerankRequest(BaseModel):
    """TEI-shaped rerank request.

    ``raw_scores=False`` (default in our client) means the server applies
    sigmoid to map cross-encoder logits into [0, 1] — what the audit /
    trace layer expects.
    """

    query: str = Field(..., min_length=1)
    texts: list[str] = Field(..., min_length=1)
    raw_scores: bool = False
    # TEI also accepts ``return_text``; we never need it (HybridRetriever
    # already holds the doc strings), so we accept-and-ignore for forward
    # compat without shipping the text back.
    return_text: bool = False
    request_id: str | None = Field(default=None, max_length=64)


class RerankHit(BaseModel):
    index: int
    score: float


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    model_path = Path(os.environ.get("BGE_RERANKER_MODEL_PATH", DEFAULT_MODEL_PATH))
    device = os.environ.get("BGE_RERANKER_DEVICE") or default_device()
    query_instruction = os.environ.get("BGE_RERANKER_QUERY_INSTRUCTION", "")
    if not model_path.exists():
        raise RuntimeError(
            f"reranker model dir not found: {model_path}\n"
            f"  Set BGE_RERANKER_MODEL_PATH to a downloaded model, or run:\n"
            f"    uv run hf download BAAI/bge-reranker-v2-m3 --local-dir {model_path}"
        )
    logger.info(
        "loading reranker from %s on %s (query_instruction=%r)",
        model_path,
        device,
        query_instruction,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()
    if device != "cpu":
        model = model.to(device)
    _state["tokenizer"] = tokenizer
    _state["model"] = model
    _state["device"] = device
    _state["query_instruction"] = query_instruction
    logger.info("reranker ready (model=%s)", model_path.name)
    yield
    _state["model"] = None
    _state["tokenizer"] = None


app = FastAPI(title="claritymed-reranker", lifespan=lifespan)
add_logging_middleware(app, server_logger=logger)


def _flush_mps_cache() -> None:
    # Same MPS allocator leak as embedder.py — see that file for details.
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()


def _require_loaded() -> tuple[Any, Any, str]:
    model = _state.get("model")
    tokenizer = _state.get("tokenizer")
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return model, tokenizer, _state["device"]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok" if _state.get("model") is not None else "loading"}


@app.post("/rerank")
def rerank(req: RerankRequest, http_req: Request) -> list[RerankHit]:
    """Score every (query, text) pair and return them sorted by score desc.

    The client's ``_parse`` only checks that ``index`` is in range and
    that ``score`` is a float — order matters in our retriever (it takes
    the top-k by position), so sort here.
    """
    if len(req.texts) > MAX_BATCH_TEXTS:
        raise HTTPException(
            status_code=413,
            detail=f"batch size {len(req.texts)} exceeds {MAX_BATCH_TEXTS}",
        )
    model, tokenizer, device = _require_loaded()
    req_id = req.request_id or http_req.headers.get("X-Request-ID", "-")
    t0 = time.monotonic()
    effective_query = _apply_query_instruction(
        req.query, _state.get("query_instruction", "")
    )
    pairs = [[effective_query, t] for t in req.texts]
    with torch.no_grad():
        encoded = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=MAX_SEQ_LEN,
        )
        if device != "cpu":
            encoded = {k: v.to(device) for k, v in encoded.items()}
        logits = model(**encoded, return_dict=True).logits.view(-1).float()
        scores_tensor = logits if req.raw_scores else logits.sigmoid()
    scores = scores_tensor.cpu().tolist()
    indexed = list(enumerate(float(s) for s in scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    result = [RerankHit(index=i, score=s) for i, s in indexed]
    _flush_mps_cache()
    top_score = result[0].score if result else 0.0
    logger.debug(
        "rerank: n=%d top_score=%.3f elapsed_ms=%.0f req_id=%s",
        len(req.texts),
        top_score,
        (time.monotonic() - t0) * 1000,
        req_id,
    )
    return result


def main() -> None:
    port = int(os.environ.get("BGE_RERANKER_PORT", DEFAULT_PORT))
    uvicorn.run(
        app, host="127.0.0.1", port=port, log_level="info", log_config=LOG_CONFIG
    )


if __name__ == "__main__":
    main()
