"""DDXPlus → canonical-qualifier translator + case generator.

Reads ``release_evidences.json`` to resolve ``E_xxx`` codes and the
patient CSV (``release_test_patients``) to find real symptom sets, then
emits one :class:`evals.emergency.schemas.Case` per sampled patient.

Why hand-author a translation table rather than auto-derive: DDXPlus
evidence questions are written in clinical-survey English ("Have you
been in contact with or ate something that you have an allergy to?")
and the answer space includes 100+ body-location values for E_55 /
E_57 / E_152. Only a small subset of those maps onto our six v1
qualifiers (and the Stage 3 additions). We curate the subset rather
than try to embed-search it, so the mapping stays auditable.

Pathology → rule coverage in this generator:

- ``Possible NSTEMI / STEMI`` (severity 1) → ``acs_acute_coronary_syndrome``
- ``Anaphylaxis``                (severity 1) → ``anaphylaxis``
- ``Pulmonary embolism``         (severity 2) → ``pulmonary_embolism`` (Stage 3)

Other DDXPlus severity-1 pathologies (Acute pulmonary edema, Ebola,
Larygospasm) do not currently map onto a v1 rule. Add them here when
Stage 4+ extends the rule pack.

Run::

    uv run python -m evals.emergency.sources.ddxplus_subset

Writes ``evals/emergency/sources/ddxplus_subset.yaml`` next to this
file, picked up automatically by ``runner.discover_case_files()``.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml

# --- file locations --------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
DDXPLUS_DIR = REPO_ROOT / "demo" / "ddxplus_demo" / "ddxplus"
EVIDENCES_JSON = DDXPLUS_DIR / "release_evidences.json"
PATIENTS_CSV = DDXPLUS_DIR / "release_test_patients"
OUTPUT_YAML = Path(__file__).resolve().parent / "ddxplus_subset.yaml"

# Deterministic sampling — re-runs produce identical case sets so the
# baseline is reproducible. Bump only if you want a fresh sample.
_RNG_SEED = 20260623

# How many patients to pull per pathology. ~30 each lands in the
# ~60-80 total the Stage 1 plan budgeted.
_SAMPLE_SIZES = {
    "Possible NSTEMI / STEMI": 30,
    "Anaphylaxis": 30,
    "Pulmonary embolism": 20,
}

# --- evidence → canonical-qualifier translation -----------------------
#
# Binary evidences map to a single qualifier when E_xxx is positive
# (present in the patient's evidence list with no value suffix).
_BINARY_TO_QUALIFIER: dict[str, str] = {
    "E_50": "diaphoresis",  # significantly increased sweating
    "E_66": "dyspnea",  # shortness of breath
    "E_159": "syncope",  # lost consciousness
    "E_214": "wheeze",  # wheezing on exhale
    "E_194": "stridor",  # high-pitched breath in
    "E_151": "swelling_present",  # swelling somewhere
    "E_82": "lightheadedness",  # near-syncope / hypotension proxy
    "E_42": "allergen_exposure",
}

# Body-location values that count as "chest" for the pain location
# question (E_55) and primary_complaint=chest_pain.
_CHEST_LOCATIONS = frozenset(
    {
        "V_29",  # lower chest
        "V_101",  # upper chest
        "V_55",  # side of chest (R)
        "V_56",  # side of chest (L)
        "V_170",  # posterior chest wall (R)
        "V_171",  # posterior chest wall (L)
    }
)

# E_57 (radiation) values that map to ``radiation_left_arm``.
_LEFT_ARM_LOCATIONS = frozenset(
    {
        "V_28",  # forearm (L)
        "V_31",  # biceps (L)
        "V_46",  # elbow (L)
        "V_195",  # shoulder (L)
        "V_178",  # triceps (L)
    }
)

# E_57 (radiation) values that map to ``radiation_jaw``.
_JAW_LOCATIONS = frozenset({"V_121", "V_163"})  # jaw, under the jaw

# E_152 (swelling location) values that map to lip/tongue swelling.
_LIP_TONGUE_LOCATIONS = frozenset(
    {
        "V_32",  # mouth
        "V_116",  # bottom lip (R)
        "V_117",  # upper lip (R)
        "V_188",  # vermilion (R)
        "V_189",  # vermilion (L)
        "V_61",  # above the tongue
        "V_162",  # under the tongue
    }
)

# E_152 (swelling) values that map to ``throat_tightness``.
_THROAT_LOCATIONS = frozenset(
    {
        "V_148",  # pharynx
        "V_33",  # thyroid cartilage
        "V_174",  # trachea
        "V_115",  # uvula
        "V_20",  # tonsil (R)
        "V_21",  # tonsil (L)
    }
)

# Leg locations that map to ``unilateral_leg_swelling`` (PE qualifier).
_LEG_LOCATIONS = frozenset(
    {
        "V_119",  # calf (R)
        "V_120",  # calf (L)
        "V_51",  # thigh (R)
        "V_52",  # thigh (L)
        "V_34",  # ankle (R)
        "V_35",  # ankle (L)
        "V_92",  # knee (R)
        "V_93",  # knee (L)
        "V_172",  # tibia (R)
        "V_173",  # tibia (L)
    }
)

# Pathology → (action, ground_truth_level, ground_truth_rule_id) mapping.
_PATHOLOGY_LABEL: dict[str, tuple[str, str, str]] = {
    "Possible NSTEMI / STEMI": (
        "acs",
        "critical",
        "acs_acute_coronary_syndrome",
    ),
    "Anaphylaxis": (
        "anaphylaxis",
        "critical",
        "anaphylaxis",
    ),
    "Pulmonary embolism": (
        "pe",
        "critical",
        "pulmonary_embolism",
    ),
}


def _parse_evidence_token(tok: str) -> tuple[str, str | None]:
    """Split ``"E_55_@_V_29"`` → ``("E_55", "V_29")`` or ``("E_50", None)``."""
    if "_@_" in tok:
        code, value = tok.split("_@_", 1)
        return code, value
    return tok, None


def _parse_patient_evidences(raw: str) -> list[tuple[str, str | None]]:
    """``"['E_53', 'E_55_@_V_29']"`` → ``[("E_53", None), ("E_55", "V_29")]``."""
    # CSV field is a python-list-like literal — eval is unsafe; do a
    # tiny tolerant parser instead.
    inner = raw.strip().lstrip("[").rstrip("]")
    if not inner:
        return []
    out: list[tuple[str, str | None]] = []
    for chunk in inner.split(","):
        s = chunk.strip().strip("'").strip('"')
        if not s:
            continue
        out.append(_parse_evidence_token(s))
    return out


def _qualifiers_for_acs(evs: list[tuple[str, str | None]]) -> tuple[str, list[str]]:
    """Build (primary_complaint, qualifiers[]) for an NSTEMI/STEMI patient."""
    qualifiers: list[str] = []
    primary = "chest_pain"  # default; downgraded if no chest location
    has_chest_pain = False
    for code, value in evs:
        if code == "E_55" and value in _CHEST_LOCATIONS:
            has_chest_pain = True
        if code == "E_57" and value in _LEFT_ARM_LOCATIONS:
            qualifiers.append("radiation_left_arm")
        if code == "E_57" and value in _JAW_LOCATIONS:
            qualifiers.append("radiation_jaw")
        if code in _BINARY_TO_QUALIFIER and value is None:
            q = _BINARY_TO_QUALIFIER[code]
            if q in {"diaphoresis", "dyspnea", "syncope"}:
                qualifiers.append(q)
    if not has_chest_pain:
        primary = (
            "chest_pain"  # default to chest_pain for NSTEMI even if E_55 was vague
        )
    return primary, _dedupe(qualifiers)


def _qualifiers_for_anaphylaxis(
    evs: list[tuple[str, str | None]],
) -> tuple[str, list[str]]:
    """Build (primary_complaint, qualifiers[]) for an anaphylaxis patient."""
    qualifiers: list[str] = []
    has_wheeze = False
    has_lightheaded = False
    has_swelling_present = False
    has_throat_loc = False
    has_lip_tongue_loc = False
    has_skin_region = False
    has_stridor = False
    for code, value in evs:
        if code == "E_214" and value is None:
            has_wheeze = True
        if code == "E_194" and value is None:
            has_stridor = True
        if code == "E_82" and value is None:
            has_lightheaded = True
        if code == "E_151" and value is None:
            has_swelling_present = True
        if code == "E_152" and value in _THROAT_LOCATIONS:
            has_throat_loc = True
        if code == "E_152" and value in _LIP_TONGUE_LOCATIONS:
            has_lip_tongue_loc = True
        if code == "E_133" and value not in (None, "V_123"):
            has_skin_region = True
    # Throat tightness = stridor OR throat swelling location.
    if has_stridor or has_throat_loc:
        qualifiers.append("throat_tightness")
    if has_lip_tongue_loc:
        qualifiers.append("lip_tongue_swelling")
    # Urticaria-plus-wheeze and urticaria-plus-hypotension are
    # composite qualifiers — both legs must be true.
    has_urticaria = has_skin_region or has_swelling_present
    if has_urticaria and has_wheeze:
        qualifiers.append("urticaria_plus_wheeze")
    if has_urticaria and has_lightheaded:
        qualifiers.append("urticaria_plus_hypotension")
    return "allergic_reaction", _dedupe(qualifiers)


def _qualifiers_for_pe(evs: list[tuple[str, str | None]]) -> tuple[str, list[str]]:
    """Build (primary_complaint, qualifiers[]) for a PE patient.

    PE rule (Stage 3) expects:
      - primary_complaint = dyspnea OR chest_pain (rule allows both)
      - qualifiers from: pleuritic, unilateral_leg_swelling,
        hemoptysis, sudden_onset, tachycardia, recent_immobilization
    """
    qualifiers: list[str] = []
    has_chest_pain = False
    has_dyspnea = False
    has_unilateral_leg_swelling = False
    for code, value in evs:
        if code == "E_55" and value in _CHEST_LOCATIONS:
            has_chest_pain = True
        if code == "E_66" and value is None:
            has_dyspnea = True
            qualifiers.append("dyspnea")
        if code == "E_152" and value in _LEG_LOCATIONS:
            has_unilateral_leg_swelling = True
        if code == "E_159" and value is None:
            qualifiers.append("syncope")
        if code == "E_220" and value is None:
            # E_220 ≈ pleuritic-type pain in DDXPlus PE pattern.
            qualifiers.append("pleuritic")
    if has_unilateral_leg_swelling:
        qualifiers.append("unilateral_leg_swelling")
    # DDXPlus PE always has dyspnea — prefer that as primary.
    primary = (
        "dyspnea" if has_dyspnea else ("chest_pain" if has_chest_pain else "dyspnea")
    )
    return primary, _dedupe(qualifiers)


def _dedupe(items: list[str]) -> list[str]:
    """Preserve order, drop duplicates."""
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _build_case(
    idx: int, row: dict[str, str], pathology: str, evs: list[tuple[str, str | None]]
) -> dict:
    """Convert one CSV row + parsed evidences into a Case-compatible dict."""
    age = int(row["AGE"])
    sex = "F" if row["SEX"].upper().startswith("F") else "M"
    short_pid, _level, rule_id = _PATHOLOGY_LABEL[pathology]
    if pathology == "Possible NSTEMI / STEMI":
        primary, quals = _qualifiers_for_acs(evs)
    elif pathology == "Anaphylaxis":
        primary, quals = _qualifiers_for_anaphylaxis(evs)
    elif pathology == "Pulmonary embolism":
        primary, quals = _qualifiers_for_pe(evs)
    else:
        raise ValueError(f"unmapped pathology {pathology!r}")
    return {
        "id": f"ddxplus_{short_pid}_{idx:03d}",
        "source": "ddxplus_subset",
        "language": "en",
        "citation": f"DDXPlus release_test_patients row · pathology={pathology}",
        "notes": f"DDXPlus age={age} sex={sex}; auto-translated from evidence codes.",
        "symptoms": {
            "primary_complaint": primary,
            "qualifiers": quals,
            "age": age,
            "sex": sex,
        },
        "ground_truth_level": "critical",
        "ground_truth_rule_id": rule_id,
    }


def generate() -> Path:
    """Read DDXPlus, translate, emit ``ddxplus_subset.yaml``. Returns the path."""
    if not EVIDENCES_JSON.exists() or not PATIENTS_CSV.exists():
        raise FileNotFoundError(
            f"DDXPlus data not found under {DDXPLUS_DIR}. Run the project's "
            "DDXPlus prepare step or unzip the release files first."
        )
    # Load evidences mostly so a future reader can ``import`` this
    # module and inspect ``_BINARY_TO_QUALIFIER`` against the real
    # schema. We do not strictly need it at translation time.
    with EVIDENCES_JSON.open("r", encoding="utf-8") as fh:
        json.load(fh)

    rng = random.Random(_RNG_SEED)
    buckets: dict[str, list[dict]] = defaultdict(list)
    with PATIENTS_CSV.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pathology = row["PATHOLOGY"]
            if pathology in _PATHOLOGY_LABEL:
                buckets[pathology].append(row)

    cases: list[dict] = []
    for pathology, rows in buckets.items():
        target = _SAMPLE_SIZES.get(pathology, 0)
        if target == 0 or not rows:
            continue
        eligible = rows
        # Pre-filter NSTEMI to age >= 35 to match the ACS rule's
        # age_min gate. DDXPlus is synthetic and includes many sub-35
        # "NSTEMI" patients that are an artifact of the data generator,
        # not a real ED population. Including them measures the
        # age-gate behavior (already pinned in unit tests) rather than
        # the qualifier-matching behavior the eval set is built for.
        if pathology == "Possible NSTEMI / STEMI":
            eligible = [r for r in rows if int(r["AGE"]) >= 35]
        picks = rng.sample(eligible, k=min(target, len(eligible)))
        for idx, row in enumerate(picks):
            evs = _parse_patient_evidences(row["EVIDENCES"])
            cases.append(_build_case(idx, row, pathology, evs))

    payload = {"cases": cases}
    OUTPUT_YAML.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return OUTPUT_YAML


if __name__ == "__main__":
    out = generate()
    print(f"wrote {out} ({sum(1 for _ in out.read_text().splitlines())} lines)")
