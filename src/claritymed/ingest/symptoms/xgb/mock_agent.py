"""Mock XGBoost agent for integration testing of the symptoms server.

Activated by setting ``CLARITYMED_SYMPTOMS_MOCK=1``. When active the real
XGBoost model is replaced with a deterministic rule-based agent that always
asks the six most discriminative evidences for Pneumonia vs Influenza and
returns pre-determined probability distributions.

This lets front-end and end-to-end tests exercise specific probability tiers
(>65 %, 40–65 %, <40 % for Pneumonia; Influenza-leading; Other-leading)
without depending on model accuracy or training state.

Question sequence (fixed):
  1. E_77  — colored/abundant sputum           → decisive for Pneumonia if Yes
  2. E_88  — severe fatigue, can't do daily     → decisive for Influenza if Yes
  3. E_94  — chills or shivers                 → both Pne+Inf if Yes
  4. E_66  — shortness of breath               → Pneumonia-leaning if Yes
  5. E_91  — fever                             → both, mild signal
  6. E_161 — appetite loss                     → both, mild signal

Stop early after Q1 or Q2 if a decisive Yes is given; otherwise ask all 6.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# Evidence IDs in order of diagnostic priority (Pne vs Inf discrimination).
# E_66 (shortness of breath) is always the first question — matches the real
# model's IG-dominant first pick from an empty state (IG=0.165 vs 0.04 for
# everything else). E_77 and E_88 follow as the decisive Pne / Inf signals.
_MOCK_EV_SEQUENCE: list[str] = ["E_77", "E_66", "E_88", "E_94", "E_91", "E_161"]

# Stop immediately after this evidence is answered Yes (decisive).
_EARLY_STOP_EV: frozenset[str] = frozenset({"E_77", "E_88"})


def _answers_from_state(state: np.ndarray, schema: dict) -> dict[str, bool | None]:
    """Read asked binary answers from the typed-BASD state vector.

    Returns {ev_id: True/False/None} for each mock evidence. None means
    the question has not been asked yet.
    """
    ev_names = [ev["name"] for ev in schema["evs"]]
    off = schema["off"]
    out: dict[str, bool | None] = {}
    for eid in _MOCK_EV_SEQUENCE:
        if eid not in ev_names:
            out[eid] = None
            continue
        ev_i = ev_names.index(eid)
        v = float(state[int(off[ev_i])])
        if v > 0:
            out[eid] = True
        elif v < 0:
            out[eid] = False
        else:
            out[eid] = None  # not asked
    return out


def _mock_probs(answers: dict[str, bool | None]) -> list[float]:
    """Rule-based [P(Pne), P(Inf), P(Other)] from mock answers.

    Designed to reliably hit the three Pneumonia tiers the test suite checks:
      >65 %:  E_77=Yes  (colored sputum — very specific for Pne)
      40-65 %: E_94=Yes + E_66=Yes, no decisive Pne/Inf signal
      <40 %:  E_66=Yes only, no strong Pne signals
      Influenza:  E_88=Yes (severe fatigue — very specific for Inf)
      Other:  no strong signals
    """
    p = answers.get

    # --- decisive Pne signal ---
    if p("E_77"):
        if p("E_66"):
            return [0.82, 0.08, 0.10]
        return [0.72, 0.08, 0.20]

    # --- decisive Inf signal ---
    if p("E_88"):
        if p("E_94"):
            return [0.05, 0.88, 0.07]
        return [0.05, 0.78, 0.17]

    # --- mixed / moderate Pne ---
    if p("E_94") and p("E_66"):
        return [0.52, 0.30, 0.18]

    # --- mild chills only ---
    if p("E_94"):
        return [0.32, 0.45, 0.23]

    # --- breath + fever ---
    if p("E_66") and p("E_91"):
        return [0.45, 0.25, 0.30]

    # --- breath only ---
    if p("E_66"):
        return [0.32, 0.15, 0.53]

    # --- fever only ---
    if p("E_91"):
        return [0.20, 0.22, 0.58]

    # --- appetite loss only ---
    if p("E_161"):
        return [0.15, 0.18, 0.67]

    # --- no signals ---
    return [0.05, 0.08, 0.87]


@dataclass
class MockXgbAgent:
    """Drop-in replacement for :class:`~claritymed.ingest.symptoms.xgb.algorithm.XgbAgent`.

    Implements the same ``next_action`` / ``should_stop`` / ``diagnose``
    interface so the symptoms server needs no changes beyond the env switch.
    ``train_step`` is a no-op (not called at serve time).
    """

    schema: dict
    n_classes: int  # 3 for Pne/Inf/Other

    def _ev_index(self, ev_id: str) -> int | None:
        ev_names = [ev["name"] for ev in self.schema["evs"]]
        return ev_names.index(ev_id) if ev_id in ev_names else None

    def next_action(self, state: np.ndarray) -> np.ndarray:
        """Return index of the next unanswered mock evidence, or 0 as fallback."""
        if state.ndim == 1:
            state = state[np.newaxis, :]
        batch = state.shape[0]
        out = np.zeros(batch, dtype=np.int64)
        off = self.schema["off"]
        ev_names = [ev["name"] for ev in self.schema["evs"]]
        for b in range(batch):
            picked = 0
            for eid in _MOCK_EV_SEQUENCE:
                if eid not in ev_names:
                    continue
                ev_i = ev_names.index(eid)
                if float(state[b, int(off[ev_i])]) == 0.0:
                    picked = ev_i
                    break
            out[b] = picked
        return out

    def should_stop(self, state: np.ndarray) -> np.ndarray:
        """Stop when all 6 mock questions are answered, or decisive early Yes."""
        if state.ndim == 1:
            state = state[np.newaxis, :]
        batch = state.shape[0]
        out = np.zeros(batch, dtype=bool)
        ev_names = [ev["name"] for ev in self.schema["evs"]]
        for b in range(batch):
            answers = _answers_from_state(state[b], self.schema)
            # Early stop on decisive Yes
            for eid in _EARLY_STOP_EV:
                if answers.get(eid) is True:
                    out[b] = True
                    break
            if out[b]:
                continue
            # Stop when all mock questions have been answered
            all_asked = all(
                answers.get(eid) is not None or eid not in ev_names
                for eid in _MOCK_EV_SEQUENCE
            )
            out[b] = all_asked
        return out

    def diagnose(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (argmax[B], probs[B, n_classes]) from rule-based mock."""
        if state.ndim == 1:
            state = state[np.newaxis, :]
        batch = state.shape[0]
        probs = np.zeros((batch, self.n_classes), dtype=np.float32)
        for b in range(batch):
            answers = _answers_from_state(state[b], self.schema)
            p = _mock_probs(answers)
            # Pad/truncate to n_classes in case model has different count.
            for k in range(self.n_classes):
                probs[b, k] = p[k] if k < len(p) else 0.0
            # Renormalize to sum to 1.
            s = probs[b].sum()
            if s > 0:
                probs[b] /= s
        return probs.argmax(axis=1), probs

    def train_step(self, *args, **kwargs):  # noqa: ANN002, ANN003
        """No-op — mock is not trainable."""
        raise NotImplementedError("MockXgbAgent does not support train_step")


def is_mock_enabled() -> bool:
    """Return True when ``CLARITYMED_SYMPTOMS_MOCK=1`` is set."""
    return os.environ.get("CLARITYMED_SYMPTOMS_MOCK", "0").strip() in {
        "1",
        "true",
        "yes",
    }


def wrap_with_mock_if_enabled(agent: object, schema: dict, n_classes: int) -> object:
    """Replace *agent* with :class:`MockXgbAgent` when mock mode is active.

    Logs a prominent warning so it's impossible to accidentally run mock in
    production without noticing.
    """
    if not is_mock_enabled():
        return agent
    import logging

    logging.getLogger(__name__).warning(
        "CLARITYMED_SYMPTOMS_MOCK=1 — replacing real XGB agent with MockXgbAgent. "
        "DO NOT use in production."
    )
    return MockXgbAgent(schema=schema, n_classes=n_classes)
