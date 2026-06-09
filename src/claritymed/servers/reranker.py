"""bge-reranker-v2-m3 server — TEI ``/rerank``-compatible wire format.

Companion to ``servers/embedder.py``. ``BgeRerankerV2M3HttpReranker``
(``core/rag/reranking/bge_v2_m3.py``) calls this server with::

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

Defaults: ``MODEL_PATH=~/.claritymed/models/bge-reranker-v2-m3``,
``PORT=8083``. Override via ``BGE_RERANKER_MODEL_PATH`` /
``BGE_RERANKER_PORT`` / ``BGE_RERANKER_DEVICE``.
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
    import torch
    import uvicorn
    from fastapi import FastAPI, HTTPException
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
# matches the upstream serving recipe.
MAX_SEQ_LEN = 512

_state: dict[str, Any] = {"model": None, "tokenizer": None, "device": None}


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


class RerankHit(BaseModel):
    index: int
    score: float


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    model_path = Path(os.environ.get("BGE_RERANKER_MODEL_PATH", DEFAULT_MODEL_PATH))
    device = os.environ.get("BGE_RERANKER_DEVICE") or default_device()
    if not model_path.exists():
        raise RuntimeError(
            f"bge-reranker model dir not found: {model_path}\n"
            f"  Run: uv run hf download BAAI/bge-reranker-v2-m3 "
            f"--local-dir {model_path}"
        )
    logger.info("loading bge-reranker-v2-m3 from %s on %s", model_path, device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()
    if device != "cpu":
        model = model.to(device)
    _state["tokenizer"] = tokenizer
    _state["model"] = model
    _state["device"] = device
    logger.info("reranker ready")
    yield
    _state["model"] = None
    _state["tokenizer"] = None


app = FastAPI(title="claritymed-reranker", lifespan=lifespan)


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
def rerank(req: RerankRequest) -> list[RerankHit]:
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
    pairs = [[req.query, t] for t in req.texts]
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
    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    port = int(os.environ.get("BGE_RERANKER_PORT", DEFAULT_PORT))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
