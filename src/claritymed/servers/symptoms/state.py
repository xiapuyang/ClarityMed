"""Server-side state singletons + sub-session dataclass.

Two layers of state:

* :data:`SERVER_STATE` — process-wide, holds loaded datasets (one
  :class:`LoadedDataset` per ``DatasetSpec`` with ``enabled=True``),
  the shared init-symptom matcher embedder, and the live sub-session
  map. Tests pre-populate ``datasets`` to skip the manifest+weights
  load and pre-populate ``sessions`` to assert lifecycle without a
  real ML run; ``init_matcher_model`` is left ``None`` and matching is
  silently skipped.
* :class:`SubSessionState` — per-session record carrying the live numpy
  state vector, turn count, and the evidence-collected trail.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from claritymed.core.symptoms.datasets import LoadedDataset

if TYPE_CHECKING:
    from claritymed.core.symptoms.init_matcher import InitMatcherEmbedder
    from claritymed.core.symptoms.schemas import InitMatcherConfig


@dataclass
class SubSessionState:
    """Per-session runtime state.

    The state vector is mutated in place each turn; ``turn_count``
    tracks how many evidences the model has actually asked (not the
    HTTP turn number). ``last_ev_idx`` is the evidence the most recent
    question asked about, carried so the next turn can translate the
    incoming answer into a state-vector write before advancing the agent.
    """

    session_id: str
    dataset_id: str
    model_id: str
    state: np.ndarray  # shape (1, S + context_size)
    turn_count: int
    started_at: float
    last_ev_idx: int | None
    language: str = "en"
    evidence_collected: list[dict] = field(default_factory=list)
    profile: dict = field(default_factory=dict)
    # Queue of SapBERT-matched init evidences awaiting user confirmation.
    # Each entry: ``{"ev_idx": int, "score": float}``. The session asks
    # each as a Yes/No question BEFORE handing off to the IG policy —
    # closes the "SapBERT semantic overreach + no negation" bug where
    # a complaint like "I have a cough" would silently inject
    # ``whooping_cough=Yes`` (E_202) into the state. When the queue is
    # empty the session runs the normal IG next_action loop.
    pending_init_confirmations: list[dict] = field(default_factory=list)


class _ServerState:
    """Mutable namespace for process-wide server state."""

    def __init__(self) -> None:
        self.datasets: dict[str, LoadedDataset] = {}
        self.sessions: dict[str, SubSessionState] = {}
        self.config_loaded: bool = False
        # Process-wide init-symptom matcher singleton. Lifespan loads
        # it once at boot; adapters encode their per-dataset catalogs
        # against this instance during ``build_dataset``. ``None`` means
        # matching is disabled — every code path treats absence as a
        # no-op rather than an error.
        self.init_matcher_model: "InitMatcherEmbedder | None" = None
        # Frozen copy of ``SymptomsConfig.init_matcher`` for the injection
        # helper — it needs ``max_matches`` + ``min_confidence_gate`` at
        # request time. Kept as ``None`` when the matcher is disabled.
        self.init_matcher_cfg: "InitMatcherConfig | None" = None

    def reset(self) -> None:
        """Test helper — drops all loaded datasets + sessions + matcher."""
        self.datasets.clear()
        self.sessions.clear()
        self.config_loaded = False
        self.init_matcher_model = None
        self.init_matcher_cfg = None


SERVER_STATE = _ServerState()


def prune_expired_sessions(now: float | None = None) -> int:
    """Drop sessions whose TTL has elapsed; return the number purged.

    TTL is read from each session's dataset spec — sessions from
    different datasets can have different TTLs (KTD-12).
    """
    now = time.time() if now is None else now
    purged = 0
    for sid, sub in list(SERVER_STATE.sessions.items()):
        ds = SERVER_STATE.datasets.get(sub.dataset_id)
        if ds is None:
            continue
        if (now - sub.started_at) > ds.spec.session_ttl_seconds:
            SERVER_STATE.sessions.pop(sid, None)
            purged += 1
    return purged
