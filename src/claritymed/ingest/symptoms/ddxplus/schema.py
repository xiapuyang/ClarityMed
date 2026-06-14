"""DDXPlus schema + patient loaders.

Translates the DDXPlus release JSONs into the shape :class:`TypedEnv`
consumes. Per-condition severity is required up-front (no silent default
to a middle tier) so the post_process safety-keyword audit sees real
severities rather than an artifact of the loader.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from claritymed.ingest.symptoms.typed_basd import (
    SEX2IDX,
    age_bucket,
    build_layout,
    parse_list,
)

DDXPLUS_EVIDENCES_JSON = "release_evidences.json"
DDXPLUS_CONDITIONS_JSON = "release_conditions.json"
DDXPLUS_SPLIT_FILES = {
    "train": "release_train_patients.zip",
    "validate": "release_validate_patients.zip",
    "test": "release_test_patients.zip",
}


class DdxplusSchemaError(RuntimeError):
    """The DDXPlus JSON files are present but structurally wrong.

    Distinct from ``FileNotFoundError`` (operator hasn't run prepare.py
    yet) — this fires when the files exist but a condition is missing
    severity, an evidence has no name, etc.
    """


@dataclass(frozen=True)
class Patient:
    """One DDXPlus patient encoded for :class:`TypedEnv`.

    ``pos`` is the union of ``bin_pos | cat_val.keys() | multi_val.keys()``
    used by ``Agent.train_step`` to compute the sym-target mask.
    ``init`` is the index of the initial evidence the simulator starts
    with — falls back to the first positive when DDXPlus's
    ``INITIAL_EVIDENCE`` doesn't resolve.
    """

    bin_pos: set[int]
    cat_val: dict[int, int]
    multi_val: dict[int, list[int]]
    pos: set[int]
    init: int
    d: int
    age: int
    sex: int
    diff: np.ndarray = field(repr=False)

    def to_dict(self) -> dict:
        """Return the dict shape that :class:`TypedEnv` expects internally."""
        return dict(
            bin_pos=self.bin_pos,
            cat_val=self.cat_val,
            multi_val=self.multi_val,
            pos=self.pos,
            init=self.init,
            d=self.d,
            age=self.age,
            sex=self.sex,
            diff=self.diff,
        )


def load_evidence_schema(
    data_dir: str | Path,
    use_ordinal: bool = False,
) -> dict:
    """Load ``release_evidences.json`` and return a typed encoding layout.

    The DDXPlus evidence JSON is keyed by code (``E_91`` etc.) and each
    entry carries ``name``, ``data_type`` (``B``/``C``/``M``), and an
    optional ``possible-values`` list. Evidences are sorted by name so
    the resulting index is stable across reloads.
    """
    path = Path(data_dir) / DDXPLUS_EVIDENCES_JSON
    if not path.exists():
        raise FileNotFoundError(
            f"DDXPlus evidence schema missing: {path}. Run "
            f"`uv run python -m claritymed.ingest.symptoms.ddxplus.prepare` "
            f"to download the dataset."
        )
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    items = list(raw.values()) if isinstance(raw, dict) else raw
    evs = []
    for entry in items:
        name = entry.get("name") or entry.get("code")
        if not name:
            raise DdxplusSchemaError(
                f"DDXPlus evidence entry missing 'name' / 'code': {entry!r}"
            )
        dtype = entry.get("data_type", "B")
        vals = entry.get("possible-values") or entry.get("possible_values") or []
        evs.append(
            dict(
                name=name,
                dtype=dtype,
                values=[str(v) for v in vals],
                is_antecedent=bool(entry.get("is_antecedent", False)),
            )
        )
    evs.sort(key=lambda d: d["name"])
    return build_layout(evs, use_ordinal=use_ordinal)


def load_pidx(
    data_dir: str | Path,
    *,
    strict_severity: bool = True,
) -> tuple[dict[str, int], np.ndarray]:
    """Load ``release_conditions.json``; return ``(name → idx, severity vector)``.

    ``strict_severity=True`` (default) raises :class:`DdxplusSchemaError`
    when any condition has no ``severity`` field. The demo silently
    defaulted to ``3.0`` — that artifact would corrupt the safety
    pipeline's tier mapping (a Critical disease silently downgraded to
    Moderate). Pass ``strict_severity=False`` only in test fixtures where
    a partial JSON is intentional.
    """
    path = Path(data_dir) / DDXPLUS_CONDITIONS_JSON
    if not path.exists():
        raise FileNotFoundError(
            f"DDXPlus conditions schema missing: {path}. Run "
            f"`uv run python -m claritymed.ingest.symptoms.ddxplus.prepare`."
        )
    with path.open("r", encoding="utf-8") as fh:
        conds = json.load(fh)
    items = list(conds.values()) if isinstance(conds, dict) else conds

    def _name(c: dict) -> str | None:
        return c.get("condition_name") or c.get("cond-name-eng")

    names = sorted(n for n in (_name(c) for c in items) if n)
    if not names:
        raise DdxplusSchemaError(
            f"DDXPlus conditions schema {path} has no named conditions"
        )
    pidx = {n: i for i, n in enumerate(names)}
    sev = np.full(len(names), 3.0)
    missing: list[str] = []
    for c in items:
        nm = _name(c)
        if nm not in pidx:
            continue
        sv = c.get("severity")
        if sv is None:
            missing.append(nm)
            continue
        sev[pidx[nm]] = float(sv)
    if missing and strict_severity:
        raise DdxplusSchemaError(
            f"DDXPlus conditions missing 'severity': {sorted(missing)[:5]}... "
            f"({len(missing)} total). Silent defaults would corrupt the "
            f"safety tier mapping — fix the corpus or pass "
            f"strict_severity=False explicitly in tests."
        )
    return pidx, sev


def parse_patient(row, schema: dict, pidx: dict[str, int]) -> Patient:
    """Parse one DDXPlus patient row into a :class:`Patient`.

    ``row`` is a pandas itertuples namedtuple with attributes
    ``EVIDENCES`` (python-literal list string), ``INITIAL_EVIDENCE``,
    ``DIFFERENTIAL_DIAGNOSIS``, ``PATHOLOGY``, ``AGE``, ``SEX``.
    """
    idx, vmap, typ = schema["index"], schema["vmap"], schema["typ"]
    bin_pos: set[int] = set()
    cat_val: dict[int, int] = {}
    multi_val: dict[int, list[int]] = {}
    for token in parse_list(row.EVIDENCES):
        if "_@_" in token:
            code, val = token.split("_@_", 1)
            ei = idx.get(code)
            if ei is None:
                continue
            lv = vmap[ei].get(str(val))
            if lv is None:
                continue
            if typ[ei] == "M":
                multi_val.setdefault(ei, []).append(lv)
            else:
                cat_val[ei] = lv
        elif token in idx:
            bin_pos.add(idx[token])
    pos = set(bin_pos) | set(cat_val) | set(multi_val)
    init_raw = getattr(row, "INITIAL_EVIDENCE", None)
    init_ev = (
        idx.get(init_raw.split("_@_", 1)[0]) if isinstance(init_raw, str) else None
    )
    if init_ev is None and pos:
        init_ev = next(iter(pos))
    diff = np.zeros(len(pidx))
    for entry in parse_list(row.DIFFERENTIAL_DIAGNOSIS):
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            nm, p = entry
            if nm in pidx:
                diff[pidx[nm]] = p
    return Patient(
        bin_pos=bin_pos,
        cat_val=cat_val,
        multi_val=multi_val,
        pos=pos,
        init=init_ev if init_ev is not None else 0,
        d=pidx[row.PATHOLOGY],
        age=age_bucket(int(row.AGE)),
        sex=SEX2IDX.get(row.SEX, 0),
        diff=diff,
    )


def load_patients(
    data_dir: str | Path,
    n: int,
    split: str,
    schema: dict,
    pidx: dict[str, int],
) -> list[dict]:
    """Read up to ``n`` patients from ``release_{split}_patients.zip``.

    Returns the dict shape :class:`TypedEnv` consumes (not ``Patient``
    instances) so the env's inner ``self.batch`` access pattern stays
    unchanged. ``Patient`` is for typed callers (training entry, tests).
    """
    if split not in DDXPLUS_SPLIT_FILES:
        raise ValueError(
            f"split must be one of {sorted(DDXPLUS_SPLIT_FILES)}, got {split!r}"
        )
    import pandas as pd  # local — pandas is a heavy import

    path = Path(data_dir) / DDXPLUS_SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(
            f"DDXPlus split file missing: {path}. Run "
            f"`uv run python -m claritymed.ingest.symptoms.ddxplus.prepare`."
        )
    df = pd.read_csv(path, nrows=n)
    return [
        parse_patient(row, schema, pidx).to_dict() for row in df.itertuples(index=False)
    ]
