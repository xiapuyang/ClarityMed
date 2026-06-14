"""Init-symptom matcher: complaint → candidate evidence index.

Bridges the train/serve gap left by removing the model's turn-0 "free
observation". Mila BASD's training loop reveals one ``INITIAL_EVIDENCE``
per patient on turn 0 (``typed_basd.py:247`` mirrors this — every
training batch starts with ``_write(s[i], p["init"], p)``). The
production runtime didn't have an analogous mechanism, so the model
was being asked to make turn-0 decisions on a state distribution it
never saw during training. This module closes that gap by inferring
the most plausible chief complaint from the user's free-text
description and pre-revealing the matched evidence in state before
the first ``next_action`` call.

Two-layer architecture:

* :class:`InitMatcherEmbedder` — process-wide singleton on
  ``SERVER_STATE.init_matcher_model``. Wraps a
  ``sentence_transformers.SentenceTransformer``; loaded once at server
  lifespan startup. Encodes catalog text at dataset load and complaint
  text per session start. CPU is the default so this never competes
  with BASD inference for GPU memory — matching is a one-shot per
  sub-session, not a hot path.
* :class:`~claritymed.core.symptoms.datasets.canonical.InitSymptomCatalog`
  — per-dataset bundle of L2-normalized vectors + their evidence idx
  list + the dataset's match threshold. Lives on the
  :class:`LoadedDataset`. Carries no reference to the embedder model,
  so dataset hot-reload doesn't disturb the shared model and a model
  swap doesn't invalidate other datasets' catalogs (it would,
  however, invalidate the catalog of any dataset re-loaded against
  the new model — embedding distributions are not portable across
  models).

Fail-safe contract: every code path in this module either returns a
usable result or returns ``None`` / skips matching. A matcher that
can't reach its model, a dataset with no candidates, or a complaint
that scores below threshold all converge on the same "no init
injection" branch — the runtime falls back to zero-init state, which
is the pre-matcher behaviour and a valid (if skewed) state for the
trained agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from claritymed.core.symptoms.datasets.canonical import (
        CanonicalEvidence,
        InitSymptomCatalog,
    )
    from claritymed.core.symptoms.schemas import InitSymptomFilter

logger = logging.getLogger(__name__)

# L2-normalization safety floor. Vectors with smaller norms after
# encoding are degenerate (zero output from the model) and would blow
# up cosine; we replace them with zero so they never win the argmax.
_NORM_EPS = 1e-12


@dataclass(frozen=True)
class MatchResult:
    """Outcome of a single complaint → evidence match.

    Returned by :meth:`InitMatcherEmbedder.match` so callers can audit
    the score and the chosen idx without re-running the cosine.
    ``evidence_idx`` is ``None`` when the top candidate scored below
    the catalog threshold — the runtime treats this as "no init
    injection", matching the no-matcher branch.
    """

    evidence_idx: int | None
    score: float


class InitMatcherEmbedder:
    """Process-wide singleton wrapping a sentence-transformers model.

    Lazy-loads the underlying model on first :meth:`encode` so the
    server lifespan can construct this instance early without paying
    the ~3-5s load cost during boot. If model loading fails, every
    subsequent encode returns ``None`` and the matcher degrades
    gracefully — no symptoms session blows up because SapBERT had a
    bad day.

    Tests inject a stub via :meth:`from_callable` so unit coverage
    doesn't pull the 440MB SapBERT checkpoint into the test
    environment.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        *,
        default_threshold: float = 0.55,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._default_threshold = default_threshold
        self._model = None  # lazy
        self._load_failed = False

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def device(self) -> str:
        return self._device

    @property
    def default_threshold(self) -> float:
        """Match threshold to use when a dataset doesn't override.

        Surfaced as an attribute so adapters can read it without
        digging into ``InitMatcherConfig``. The config flows: load
        ``SymptomsConfig.init_matcher.threshold`` at server lifespan,
        construct the embedder with that value, adapters read it back
        out when building per-dataset catalogs.
        """
        return self._default_threshold

    def _ensure_loaded(self) -> bool:
        """Load the model on first use. Returns True if available."""
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        try:
            # Local import: sentence_transformers pulls torch + transformers,
            # roughly 800MB of dependency surface. Importing at module load
            # would slow down every plugin path that touches the symptoms
            # package even when matching is disabled.
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning(
                "init-matcher: sentence-transformers not installed; "
                "matching disabled. Install with "
                "`uv sync --extra symptoms-server`."
            )
            self._load_failed = True
            return False
        try:
            device = self._resolved_device()
            logger.info("init-matcher: loading %s on %s", self._model_id, device)
            self._model = SentenceTransformer(self._model_id, device=device)
        except Exception:  # noqa: BLE001 — never crash the server on this
            logger.exception(
                "init-matcher: failed to load %s; matching disabled",
                self._model_id,
            )
            self._load_failed = True
            return False
        return True

    def _resolved_device(self) -> str:
        if self._device != "auto":
            return self._device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"

    def encode(self, texts: Sequence[str]) -> np.ndarray | None:
        """Encode a batch of strings → ``[N, D]`` L2-normalized fp32.

        Returns ``None`` when the model isn't loaded and can't load —
        callers MUST treat ``None`` as "skip matching", not as a bug.
        """
        if not self._ensure_loaded():
            return None
        try:
            arr = self._model.encode(  # type: ignore[union-attr]
                list(texts),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception:  # noqa: BLE001
            logger.exception("init-matcher: encode failed for %d texts", len(texts))
            return None
        return np.asarray(arr, dtype=np.float32)

    def match(self, complaint_text: str, catalog: "InitSymptomCatalog") -> MatchResult:
        """Run cosine against ``catalog.matrix`` and return the top match.

        Returns ``MatchResult(evidence_idx=None, score=...)`` when the
        top score is below the catalog's threshold, or when encoding
        failed. Both branches converge on "no init injection" at the
        caller.
        """
        if not complaint_text or not complaint_text.strip():
            return MatchResult(evidence_idx=None, score=0.0)
        vecs = self.encode([complaint_text.strip()])
        if vecs is None or vecs.shape[0] == 0:
            return MatchResult(evidence_idx=None, score=0.0)
        vec = vecs[0]
        norm = float(np.linalg.norm(vec))
        if norm < _NORM_EPS:
            return MatchResult(evidence_idx=None, score=0.0)
        # Catalog is pre-normalized at build time, complaint vec is
        # normalized by the encoder — dot product == cosine similarity.
        scores = catalog.matrix @ vec
        best_local = int(np.argmax(scores))
        best_score = float(scores[best_local])
        if best_score < catalog.threshold:
            return MatchResult(evidence_idx=None, score=best_score)
        return MatchResult(
            evidence_idx=catalog.candidate_idx[best_local], score=best_score
        )


# --- catalog builders -------------------------------------------------------


def filter_candidate_evidences(
    evidences: Sequence["CanonicalEvidence"],
    spec: "InitSymptomFilter",
) -> list["CanonicalEvidence"]:
    """Apply the per-dataset filter to a list of canonical evidences.

    Default filter (Mila parity): ``is_antecedent=False`` AND
    ``dtype="B"``. Datasets may relax either side via
    :class:`~claritymed.core.symptoms.schemas.InitSymptomFilter`.
    Returns evidences in their original order so the matrix row
    order is reproducible across reloads.
    """
    allowed = set(spec.allowed_dtypes)
    out: list["CanonicalEvidence"] = []
    for ev in evidences:
        if spec.exclude_antecedent and ev.is_antecedent:
            continue
        if ev.dtype not in allowed:
            continue
        out.append(ev)
    return out


def candidate_text(ev: "CanonicalEvidence") -> str:
    """Pick the embedder-input text for a candidate evidence.

    Prefer the EN native question text (SapBERT is English-only); fall
    back to the evidence id when no English text exists so the catalog
    row stays addressable but scores will be near zero. The id-only
    fallback keeps the matrix dimensions stable even on malformed
    corpora — operators see "no matches" in audit, not a load failure.
    """
    return ev.native_question_text.get("en") or ev.id


def build_catalog(
    evidences: Sequence["CanonicalEvidence"],
    spec: "InitSymptomFilter",
    embedder: InitMatcherEmbedder,
    threshold: float,
) -> "InitSymptomCatalog | None":
    """Build a catalog from a canonical evidence list. ``None`` on failure.

    Returns ``None`` when the candidate pool is empty (dataset has no
    eligible evidences under the filter) or when encoding fails — the
    LoadedDataset stores the ``None`` and runtime skips matching.
    Callers log the cause; this helper stays neutral.
    """
    # Local import keeps the canonical module free of an
    # init_matcher import cycle.
    from claritymed.core.symptoms.datasets.canonical import InitSymptomCatalog

    candidates = filter_candidate_evidences(evidences, spec)
    if not candidates:
        logger.warning(
            "init-matcher: no candidate evidences after filter "
            "(exclude_antecedent=%s, allowed_dtypes=%s); matching disabled",
            spec.exclude_antecedent,
            list(spec.allowed_dtypes),
        )
        return None
    texts = [candidate_text(ev) for ev in candidates]
    matrix = embedder.encode(texts)
    if matrix is None:
        logger.warning("init-matcher: catalog encode returned None; matching disabled")
        return None
    if matrix.shape[0] != len(candidates):
        logger.error(
            "init-matcher: encoder returned %d vectors for %d candidates; "
            "matching disabled",
            matrix.shape[0],
            len(candidates),
        )
        return None
    return InitSymptomCatalog(
        candidate_idx=[ev.idx for ev in candidates],
        matrix=matrix,
        threshold=threshold,
    )
