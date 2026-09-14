"""Typed-BASD differential-diagnosis algorithm — dataset-agnostic body.

Ported from ``demo/ddxplus_demo/ddxplus_typed_demo.py``. The demo flattened
DDXPlus loading + the algorithm into one file; this module keeps only the
algorithm + the encoding layout (``build_layout`` operates on a plain
``[{name, dtype, values}]`` list — no DDXPlus paths leak in). Per-dataset
adapters (see :mod:`claritymed.ingest.symptoms.ddxplus`) supply schema +
patient lists in the shape :class:`TypedEnv` consumes.

State representation per evidence (preserves the demo's encoding):

* ``binary``                -> ``[trinary]``                              (width 1)
* ``categorical``           -> ``[asked | one-hot(K)]``                   (width 1+K)
* ``categorical (numeric)`` -> ``[asked | one-hot(K) | ordinal scalar]``  (width 1+K+1)
  when ``use_ordinal=True`` AND every value parses as a number
* ``multi-choice``          -> ``[asked | multi-hot(K)]``                 (width 1+K)

The state vector concatenates these blocks plus a one-hot age-bucket and
sex marker (``context_size`` slots).

torch is imported lazily inside :func:`build_basd` so importing this
module on a host without torch (or for the pure-numpy :class:`TypedEnv`
self-tests) does not require the dep.
"""

from __future__ import annotations

import ast
import random
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

# Age bucketing matches the DDXPlus typed-BASD demo. Right-open intervals
# in [lo, hi] with the final sentinel bucket catching ages over 74. The
# server's per-dataset adapter may override ``age_bucket`` if its model
# expects different bucketing (e.g. a pediatric-only dataset).
AGE_BUCKETS = [
    (0, 0),
    (1, 4),
    (5, 14),
    (15, 29),
    (30, 44),
    (45, 59),
    (60, 74),
    (200, 200),
]

# Sex encoding — male=0, female=1. Datasets with non-binary sex labels
# override this in their adapter rather than mutating the constant.
SEX2IDX = {"M": 0, "F": 1}


def age_bucket(a: int) -> int:
    """Map an age in years to its bucket index.

    Returns the last bucket for ages above the largest bucket's high.
    """
    for i, (_lo, hi) in enumerate(AGE_BUCKETS):
        if a <= hi:
            return i
    return len(AGE_BUCKETS) - 1


def parse_list(s: Any) -> list:
    """Parse a python-literal string into a list.

    DDXPlus stores evidence and differential-diagnosis lists as
    ``repr``-style python strings in CSV cells. ``ast.literal_eval`` is
    safe (it accepts only literals — no callable side effects). Returns
    the input unchanged when it's already a list, or ``[]`` when it's
    falsy.
    """
    if isinstance(s, str):
        return ast.literal_eval(s)
    return s or []


def seed_everything(seed: int) -> None:
    """Seed numpy, stdlib random, and torch (if importable).

    Lazy torch import — call sites that don't need torch (TypedEnv
    self-tests, pure-numpy harnesses) never trigger the dep.
    """
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # noqa: BLE001
        # No torch → numpy + random seed still in place; that's enough
        # for TypedEnv determinism and any non-NN harness.
        pass


def _isnum(v: Any) -> bool:
    """Return True when ``v`` parses as a float, else False."""
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def build_layout(
    evs: Sequence[dict[str, Any]],
    use_ordinal: bool = False,
) -> dict[str, Any]:
    """Return the typed evidence encoding layout for a list of evidences.

    ``evs`` is a list of ``{name, dtype, values}`` dicts; ``dtype`` is
    ``"B"`` (binary), ``"C"`` (categorical), or ``"M"`` (multi-choice).
    ``values`` is the categorical/multi value list (ignored for ``"B"``).
    Each dict may optionally carry ``is_antecedent`` (bool, defaults to
    ``False``); the flag is plumbed through the schema so :func:`interactive_eval`
    can report symptom vs. antecedent metrics separately without changing
    the state encoding or model trunk (Mila BASD parity).

    Returns a dict with ``off`` (per-evidence block offsets), ``typ``,
    ``vmap`` (value → local index), ``sym_size`` (total symptom slots),
    ``n_ev`` (number of evidences), ``is_antecedent`` (bool ndarray of
    shape ``[n_ev]``), and ordinal-scalar metadata used by
    :meth:`TypedEnv._write`.

    ``use_ordinal=True`` opts numeric-categorical evidences (values that
    all parse as numbers) into an extra scalar slot in [0, 1] alongside
    the one-hot. The demo found this redundant with one-hot in the
    large-data regime and slightly worse on DDXPlus; it can help when
    data is scarce.
    """
    index = {e["name"]: i for i, e in enumerate(evs)}
    off: list[int] = []
    typ: list[str] = []
    vmap: list[dict[str, int]] = []
    has_ord: list[bool] = []
    ord_at: list[int] = []
    num_arr: list[np.ndarray | None] = []
    num_lo: list[float] = []
    num_span: list[float] = []
    cur = 0
    for e in evs:
        off.append(cur)
        typ.append(e["dtype"])
        vals = e["values"]
        k = len(vals)
        if e["dtype"] == "B":
            vmap.append({})
            has_ord.append(False)
            ord_at.append(-1)
            num_arr.append(None)
            num_lo.append(0.0)
            num_span.append(1.0)
            cur += 1
        else:
            vmap.append({v: j for j, v in enumerate(vals)})
            numeric = e["dtype"] == "C" and k > 0 and all(_isnum(v) for v in vals)
            if numeric and use_ordinal:
                arr = np.array([float(v) for v in vals])
                lo = float(arr.min())
                has_ord.append(True)
                ord_at.append(cur + 1 + k)
                num_arr.append(arr)
                num_lo.append(lo)
                num_span.append(max(float(arr.max()) - lo, 1e-9))
                cur += 1 + k + 1
            else:
                has_ord.append(False)
                ord_at.append(-1)
                num_arr.append(None)
                num_lo.append(0.0)
                num_span.append(1.0)
                cur += 1 + k
    is_antecedent = np.array(
        [bool(e.get("is_antecedent", False)) for e in evs], dtype=bool
    )
    return dict(
        evs=list(evs),
        index=index,
        off=np.array(off),
        typ=typ,
        vmap=vmap,
        sym_size=cur,
        n_ev=len(evs),
        has_ord=has_ord,
        ord_at=ord_at,
        num_arr=num_arr,
        num_lo=num_lo,
        num_span=num_span,
        is_antecedent=is_antecedent,
    )


class TypedEnv:
    """Pure-numpy patient-simulator environment for typed-BASD.

    Holds a list of pre-parsed patients (one dict per record) and a
    schema produced by :func:`build_layout`. The agent calls
    :meth:`initialize_state` to draw a batch, :meth:`reveal` to apply
    answers, and :meth:`asked_mask` to know which evidences have been
    explored. No torch dependency.
    """

    def __init__(self, patients: list[dict], schema: dict, n_dis: int) -> None:
        self.patients = patients
        self.schema = schema
        self.diag_size = n_dis
        self.S = schema["sym_size"]
        self.n_ev = schema["n_ev"]
        self.off = schema["off"]
        self.typ = schema["typ"]
        self.vmap = schema["vmap"]
        self.is_antecedent = schema["is_antecedent"]
        self.context_size = len(AGE_BUCKETS) + len(SEX2IDX)
        self.idx = 0
        self.order = np.arange(len(patients))

    def reset(self) -> None:
        """Reset cursor + reshuffle patient order in place."""
        self.idx = 0
        np.random.shuffle(self.order)

    def _write(self, srow: np.ndarray, ev: int, p: dict) -> None:
        off = self.off[ev]
        t = self.typ[ev]
        sc = self.schema
        if t == "B":
            srow[off] = 1.0 if ev in p["bin_pos"] else -1.0
        else:
            srow[off] = 1.0  # asked flag
            if t == "C":
                lv = p["cat_val"].get(ev)
                if lv is not None:
                    srow[off + 1 + lv] = 1.0
                    if sc["has_ord"][ev]:
                        srow[sc["ord_at"][ev]] = (
                            sc["num_arr"][ev][lv] - sc["num_lo"][ev]
                        ) / sc["num_span"][ev]
            else:
                for lv in p["multi_val"].get(ev, []):
                    srow[off + 1 + lv] = 1.0

    def initialize_state(self, batch: int) -> tuple[np.ndarray, np.ndarray]:
        """Draw the next ``batch`` patients and return ``(state, disease_ids)``."""
        sel = self.order[self.idx : self.idx + batch]
        self.idx += batch
        b = len(sel)
        s_size, c_size = self.S, self.context_size
        s = np.zeros((b, s_size + c_size))
        self.batch = [self.patients[k] for k in sel]
        self.disease = np.array([p["d"] for p in self.batch])
        self.diff = np.array([p["diff"] for p in self.batch])
        for i, p in enumerate(self.batch):
            self._write(s[i], p["init"], p)
            s[i, s_size + p["age"]] = 1
            s[i, s_size + len(AGE_BUCKETS) + p["sex"]] = 1
        return s, self.disease

    def reveal(self, s: np.ndarray, a: np.ndarray, done: np.ndarray) -> np.ndarray:
        """Apply per-row actions ``a`` to a copy of state ``s``, skipping ``done`` rows."""
        s_ = s.copy()
        for i in range(len(a)):
            if not done[i]:
                self._write(s_[i], a[i], self.batch[i])
        return s_

    def asked_mask(self, s: np.ndarray) -> np.ndarray:
        """Return a ``[B, n_ev]`` boolean mask: True where the block-start slot is set."""
        return s[:, self.off] != 0


@dataclass(frozen=True)
class EvalMetrics:
    """Typed result of :func:`interactive_eval` (replaces the demo's dict).

    The ``P{S,A}{R,P,F1}`` fields mirror Mila BASD's split metrics:
    symptom-side (``PSR``/``PSP``/``PSF1``) is computed over evidences
    flagged ``is_antecedent=False`` in the schema, antecedent-side
    (``PAR``/``PAP``/``PAF1``) over the rest. Each is a macro average
    over patients that have at least one ground-truth evidence (or, for
    precision, at least one ``asked`` evidence) on the corresponding
    side; patients with no such evidence are skipped so the metric is
    not silently dragged toward zero. ``NaN`` indicates no patient
    contributed (e.g. dataset has zero antecedent evidences).
    """

    IL: float
    ACC: float
    GTPA: float
    DDR: float
    DDP: float
    DDF1: float
    DSR: float
    n_severe: int
    PSR: float = float("nan")
    PSP: float = float("nan")
    PSF1: float = float("nan")
    PAR: float = float("nan")
    PAP: float = float("nan")
    PAF1: float = float("nan")


def build_basd(
    env: TypedEnv,
    n_dis: int,
    hidden: int,
    lr: float,
    device: str,
    stop_thres: float,
    stop_mode: str = "learned",
) -> Any:
    """Build a BASD agent (supervised symptom-reconstruction + sympt-prob stop).

    Args:
        env: TypedEnv supplying ``S`` (symptom slots), ``n_ev``,
            ``context_size``, and per-evidence offsets.
        n_dis: Number of pathology classes.
        hidden: Hidden layer width for trunk + heads.
        lr: Adam learning rate.
        device: Torch device string (``"cpu"`` / ``"mps"`` / ``"cuda"``).
            Use :func:`claritymed.core.device.resolve_device` to pick.
        stop_thres: Stop-gate threshold. With ``stop_mode="learned"`` this
            is the P(collected) cutoff; with ``"heuristic"`` it's the
            max symptom-prob cutoff (smaller → ask more questions).
        stop_mode: ``"learned"`` (trains a stop head) or ``"heuristic"``
            (uses max symptom prob). The demo benchmarks ``"heuristic"``
            as the empirically better choice on DDXPlus.

    Raises:
        ValueError: ``hidden <= 0`` or ``lr <= 0``.

    Returns:
        An ``Agent`` instance with ``next_action``, ``should_stop``,
        ``diagnose``, and ``train_step`` methods.
    """
    if hidden <= 0:
        raise ValueError(f"hidden must be > 0, got {hidden!r}")
    if lr <= 0:
        raise ValueError(f"lr must be > 0, got {lr!r}")
    if stop_mode not in {"learned", "heuristic"}:
        raise ValueError(
            f"stop_mode must be 'learned' or 'heuristic', got {stop_mode!r}"
        )

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    s_size = env.S
    n_ev = env.n_ev
    in_size = s_size + env.context_size
    off = env.off

    class Agent:
        def __init__(self) -> None:
            self.trunk = nn.Sequential(
                nn.Linear(in_size, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            ).to(device)
            self.sym = nn.Linear(hidden, n_ev).to(device)
            self.patho = nn.Linear(hidden, n_dis).to(device)
            # Stop head only when learned mode uses it. Training the stop
            # head in heuristic mode steals trunk capacity and degrades
            # sym/patho (DDR/DDF1 drop).
            self.stop = (
                nn.Linear(hidden, 1).to(device) if stop_mode == "learned" else None
            )
            params = (
                list(self.trunk.parameters())
                + list(self.sym.parameters())
                + list(self.patho.parameters())
            )
            if self.stop is not None:
                params += list(self.stop.parameters())
            self.opt = torch.optim.Adam(params, lr=lr)
            self.thres = stop_thres
            self.mode = stop_mode
            self.temp = 1.0

        def _h(self, state: np.ndarray):
            h = self.trunk(torch.as_tensor(state, dtype=torch.float32, device=device))
            return (
                self.sym(h),
                self.patho(h),
                self.stop(h) if self.stop is not None else None,
            )

        def diagnose(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Return ``(argmax_disease, prob_matrix)``. Temperature in ``self.temp``."""
            with torch.no_grad():
                _, pl, _ = self._h(state)
                p = torch.softmax(pl / self.temp, 1).cpu().numpy()
            return p.argmax(1), p

        def _symprob(self, state: np.ndarray) -> np.ndarray:
            asked = state[:, off] != 0
            with torch.no_grad():
                sl, _, _ = self._h(state)
                sp = torch.sigmoid(sl).cpu().numpy()
            sp[asked] = -1.0
            return sp

        def next_action(self, state: np.ndarray) -> np.ndarray:
            return self._symprob(state).argmax(1)

        def should_stop(self, state: np.ndarray) -> np.ndarray:
            if self.mode == "heuristic":
                return self._symprob(state).max(1) < self.thres
            with torch.no_grad():
                _, _, stl = self._h(state)
                return torch.sigmoid(stl).squeeze(-1).cpu().numpy() > self.thres

        def train_step(
            self, batch: list[dict], target: str
        ) -> tuple[float, float, float]:
            b = len(batch)
            state = np.zeros((b, in_size))
            ysym = np.zeros((b, n_ev))
            diff = np.zeros((b, n_dis))
            dis = np.zeros(b, int)
            all_ev = np.arange(n_ev)
            for i, p in enumerate(batch):
                pos = np.array(sorted(p["pos"])) if p["pos"] else np.array([], int)
                ysym[i, pos] = 1
                if len(pos):
                    # When training the stop head we sometimes reveal ALL
                    # positives to give the stop target a positive class.
                    # In heuristic mode the 40%-complete biasing shifts
                    # training toward complete states (away from partial
                    # inference states) and degrades sym/patho.
                    if self.stop is not None and np.random.rand() < 0.4:
                        kpos = len(pos)
                    else:
                        kpos = np.random.randint(1, len(pos) + 1)
                    for ev in np.random.choice(pos, kpos, replace=False):
                        env._write(state[i], ev, p)
                    neg = np.setdiff1d(all_ev, pos)
                    kneg = np.random.randint(0, kpos + 1)
                    if kneg > 0 and len(neg) > 0:
                        for ev in np.random.choice(
                            neg, min(kneg, len(neg)), replace=False
                        ):
                            env._write(state[i], ev, p)
                state[i, s_size + p["age"]] = 1
                state[i, s_size + len(AGE_BUCKETS) + p["sex"]] = 1
                diff[i] = p["diff"]
                dis[i] = p["d"]
            asked = state[:, off] != 0
            self.trunk.train()
            self.opt.zero_grad()

            def t(x):
                return torch.as_tensor(x, dtype=torch.float32, device=device)

            sl, pl, stl = self._h(state)
            mask = t((~asked).astype(np.float32))
            sym_loss = (
                F.binary_cross_entropy_with_logits(sl, t(ysym), reduction="none") * mask
            ).sum() / mask.sum().clamp_min(1)
            if target == "pathology":
                pat = F.cross_entropy(
                    pl, torch.as_tensor(dis, dtype=torch.long, device=device)
                )
            else:
                q = t(diff)
                q = q / q.sum(1, keepdim=True).clamp_min(1e-9)
                pat = -(q * torch.log_softmax(pl, 1)).sum(1).mean()
            if self.stop is not None:
                stop_tgt = ((ysym.astype(bool) & ~asked).sum(1) == 0).astype(np.float32)
                stop_loss = F.binary_cross_entropy_with_logits(
                    stl.squeeze(-1), t(stop_tgt)
                )
                (sym_loss + pat + stop_loss).backward()
                self.opt.step()
                return (
                    float(sym_loss.detach()),
                    float(pat.detach()),
                    float(stop_loss.detach()),
                )
            (sym_loss + pat).backward()
            self.opt.step()
            return float(sym_loss.detach()), float(pat.detach()), 0.0

    return Agent()


def interactive_eval(
    env: TypedEnv,
    agent: Any,
    maxstep: int,
    games: int,
    severity: np.ndarray,
) -> EvalMetrics:
    """Run the agent on the env's patients for ``maxstep`` interactions per game.

    Returns :class:`EvalMetrics` with IL (interaction length), ACC
    (top-1 accuracy), GTPA (ground-truth-in-pred-above), DDR / DDP /
    DDF1 (differential recall/precision/F1 at >0.01 prob), DSR (severe
    recall over diseases with severity < 3), ``n_severe`` (the count of
    severe-disease cases that contributed to DSR), and the Mila-parity
    PSR/PSP/PSF1 + PAR/PAP/PAF1 split metrics (symptom vs. antecedent
    inquiry recall/precision/F1; see :class:`EvalMetrics` doc).
    """
    env.reset()
    env.order = np.arange(len(env.patients))
    il_t = acc_t = gtpa_t = 0.0
    ddr = ddp = ddf1 = 0.0
    nb = npat = 0
    dsr = dsn = 0.0
    sev_mask = severity < 3
    antec = env.is_antecedent  # [n_ev] bool — schema-level split
    psr = psp = psf1 = par = pap = paf1 = 0.0
    n_sym_r = n_sym_p = n_sym_f1 = 0
    n_atcd_r = n_atcd_p = n_atcd_f1 = 0
    while env.idx + games <= len(env.patients):
        s, _ = env.initialize_state(games)
        done = agent.should_stop(s)
        il = np.zeros(games)
        for _ in range(maxstep):
            a = agent.next_action(s)
            s = env.reveal(s, a, done)
            il[~done] += 1
            done = done | agent.should_stop(s)
            if done.all():
                break
        a_d, p_d = agent.diagnose(s)
        gt = env.diff > 0.01
        pred = p_d > 0.01
        asked = env.asked_mask(s)  # [B, n_ev] bool
        il_t += il.mean()
        acc_t += (a_d == env.disease).mean()
        gtpa_t += np.mean([float(pred[i, env.disease[i]]) for i in range(games)])
        for i in range(games):
            it = (gt[i] & pred[i]).sum()
            r = it / max(1, gt[i].sum())
            p = it / max(1, pred[i].sum())
            ddr += r
            ddp += p
            ddf1 += 2 * p * r / (p + r + 1e-10)
            npat += 1
            gs = gt[i] & sev_mask
            if gs.sum() > 0:
                ps = pred[i] & sev_mask
                dsr += (gs & ps).sum() / gs.sum()
                dsn += 1
            # Split symptom / antecedent inquiry metrics. ``pos`` is the
            # union of binary-positive + categorical + multi evidences the
            # patient actually has (built by the dataset adapter).
            gt_mask = np.zeros(env.n_ev, dtype=bool)
            for ev_idx in env.batch[i]["pos"]:
                gt_mask[ev_idx] = True
            asked_row = asked[i]
            gt_sym = gt_mask & ~antec
            gt_atcd = gt_mask & antec
            asked_sym = asked_row & ~antec
            asked_atcd = asked_row & antec
            inter_sym = int((gt_sym & asked_row).sum())
            inter_atcd = int((gt_atcd & asked_row).sum())
            gt_sym_n = int(gt_sym.sum())
            gt_atcd_n = int(gt_atcd.sum())
            asked_sym_n = int(asked_sym.sum())
            asked_atcd_n = int(asked_atcd.sum())
            r_s = inter_sym / gt_sym_n if gt_sym_n else None
            p_s = inter_sym / asked_sym_n if asked_sym_n else None
            r_a = inter_atcd / gt_atcd_n if gt_atcd_n else None
            p_a = inter_atcd / asked_atcd_n if asked_atcd_n else None
            if r_s is not None:
                psr += r_s
                n_sym_r += 1
            if p_s is not None:
                psp += p_s
                n_sym_p += 1
            if r_s is not None and p_s is not None:
                psf1 += 2 * p_s * r_s / (p_s + r_s + 1e-10)
                n_sym_f1 += 1
            if r_a is not None:
                par += r_a
                n_atcd_r += 1
            if p_a is not None:
                pap += p_a
                n_atcd_p += 1
            if r_a is not None and p_a is not None:
                paf1 += 2 * p_a * r_a / (p_a + r_a + 1e-10)
                n_atcd_f1 += 1
        nb += 1

    def _avg(total: float, n: int) -> float:
        return total / n * 100 if n else float("nan")

    return EvalMetrics(
        IL=il_t / nb,
        ACC=acc_t / nb * 100,
        GTPA=gtpa_t / nb * 100,
        DDR=ddr / npat * 100,
        DDP=ddp / npat * 100,
        DDF1=ddf1 / npat * 100,
        DSR=(dsr / dsn * 100 if dsn else float("nan")),
        n_severe=int(dsn),
        PSR=_avg(psr, n_sym_r),
        PSP=_avg(psp, n_sym_p),
        PSF1=_avg(psf1, n_sym_f1),
        PAR=_avg(par, n_atcd_r),
        PAP=_avg(pap, n_atcd_p),
        PAF1=_avg(paf1, n_atcd_f1),
    )
