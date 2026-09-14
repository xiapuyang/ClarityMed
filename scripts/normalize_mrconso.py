#!/usr/bin/env python3
"""Normalize UMLS MRCONSO.RRF into the ConceptRecord JSONL schema.

WHY THIS SCRIPT EXISTS
----------------------
``init_terminology.py`` accepts JSONL via ``--merge``. Each upstream
vocabulary needs its own normalizer; this is the UMLS MRCONSO one.
Like ``normalize_mesh.py``, this script is source-agnostic with respect
to ``SHARED_DIR``: it reads MRCONSO.RRF, writes ConceptRecord JSONL to
stdout (or ``--out FILE``), and the operator decides when to ``--merge``.

INPUT
-----
``MRCONSO.RRF`` — pipe-delimited, one row per (CUI × atom) pair.
Column layout (0-indexed, per UMLS documentation):

    0  CUI  — Concept Unique Identifier
    1  LAT  — Language (ENG, CHI, SPA, …)
    2  TS   — Term status
    3  LUI  — Lexical Unique Identifier
    4  STT  — String Type
    5  SUI  — String Unique Identifier
    6  ISPREF — Preferred-form flag
    7  AUI  — Atom Unique Identifier
    8  SAUI — Source-specific atom id
    9  SCUI — Source-specific concept id
    10 SDUI — Source-specific descriptor id
    11 SAB  — Source abbreviation
    12 TTY  — Term type
    13 CODE — Source-specific code
    14 STR  — String (the surface form we want)
    15 SRL  — Source Restriction Level
    16 SUPPRESS — Suppression flag (N = in use; O/Y/E = suppressed)
    17 CVF  — Content View Flag

We only emit rows where SUPPRESS == "N".

LANGUAGE SUPPORT
----------------
Only ENG (→ "en") and CHI (→ "zh") are mapped; all other languages are
silently skipped. Chinese entries in standard UMLS releases come from
LNC-ZH-CN (LOINC Chinese); richer Chinese terminology typically arrives
via a separate CMeKG --merge pass.

TYPE INFERENCE
--------------
MRCONSO alone does not carry semantic-type information (that lives in
MRSTY.RRF). We infer concept type from the Source Abbreviation (SAB)
of every atom that belongs to a CUI, using priority order:

    drug > disease > symptom > procedure > other

A CUI's winning type is the highest-priority type across all its atoms'
sources. The mapping is conservative: when a source straddles categories
(e.g. SNOMEDCT_US covers everything) we default to "other" rather than
guessing wrong.

Known SAB → type assignments::

    RXNORM, MTHSPL, VANDF, MMSL, MDDB  → drug
    ICD10CM, ICD9CM, ICD10, OMIM        → disease
    ICD10PCS, CPT, HCPCS                → procedure
    everything else                     → other

OUTPUT
------
One JSON object per line::

    {"concept_id":"umls:C0004057","type":"drug","aliases":[{"text":...}]}

USAGE
-----
::

    # Full MRCONSO.RRF → JSONL (all types, ENG + CHI)
    uv run python scripts/normalize_mrconso.py \\
        --input ~/.claritymed/shared/terminology/raw/2026AA/META/MRCONSO.RRF \\
        --out /tmp/umls.jsonl

    # Drug + disease only (saves memory — fewer aliases collected)
    uv run python scripts/normalize_mrconso.py \\
        --input /path/to/MRCONSO.RRF \\
        --types drug,disease \\
        --out /tmp/umls-clinical.jsonl

    # Then merge into concepts.jsonl
    uv run python scripts/init_terminology.py --merge /tmp/umls.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import IO

# --- constants --------------------------------------------------------------

# MRCONSO.RRF column indices (0-based).
_COL_CUI = 0
_COL_LAT = 1
_COL_SAB = 11
_COL_STR = 14
_COL_SUPPRESS = 16

# Only these two languages map to a supported ConceptLanguage value.
_LAT_TO_LANG: dict[str, str] = {
    "ENG": "en",
    "CHI": "zh",
}

# SAB → ConceptType assignment.
# Sources not listed here resolve to "other" via the fallback in _sab_type().
_SAB_TO_TYPE: dict[str, str] = {
    # --- drugs ---------------------------------------------------------------
    "RXNORM": "drug",  # NLM RxNorm — most reliable drug source in UMLS
    "MTHSPL": "drug",  # FDA Structured Product Labels (drug labeling)
    "VANDF": "drug",  # VA National Drug File
    "MMSL": "drug",  # Multum MedSource
    "MDDB": "drug",  # MediSpan Drug Database
    # --- diseases ------------------------------------------------------------
    "ICD10CM": "disease",
    "ICD9CM": "disease",
    "ICD10": "disease",
    "OMIM": "disease",  # Online Mendelian Inheritance in Man (genetic diseases)
    # --- procedures ----------------------------------------------------------
    "ICD10PCS": "procedure",
    "CPT": "procedure",
    "HCPCS": "procedure",
}

# Priority order for resolving a CUI's final type when multiple SABs conflict.
# Lower index = higher priority.
_TYPE_PRIORITY: list[str] = ["drug", "disease", "symptom", "procedure", "other"]

# Print a progress line every N rows scanned.
_PROGRESS_INTERVAL = 500_000

_VALID_TYPES = {"drug", "disease", "symptom", "procedure", "other"}


# --- helpers ----------------------------------------------------------------


def _sab_type(sab: str) -> str:
    return _SAB_TO_TYPE.get(sab, "other")


def _higher_priority(a: str, b: str) -> str:
    """Return whichever type has higher (lower index) priority."""
    return a if _TYPE_PRIORITY.index(a) <= _TYPE_PRIORITY.index(b) else b


# --- core processing --------------------------------------------------------


def _normalize(
    rrf_path: Path,
    *,
    out: IO[str],
    type_filter: set[str] | None,
) -> int:
    """Stream MRCONSO.RRF, group by CUI, emit ConceptRecord JSONL.

    Groups aliases in RAM (one dict entry per CUI). UMLS 2026AA has ~4 M
    unique CUIs; peak RSS is typically 3–5 GB on a standard laptop — fine
    for a one-time ETL run.

    Returns the number of records written.
    """
    # {CUI: (best_type, {lang: {lower_text: original_text}})}
    concepts: dict[str, list] = {}

    scanned = 0
    with rrf_path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            scanned += 1
            if scanned % _PROGRESS_INTERVAL == 0:
                print(
                    f"  [scan] {scanned:>11,} rows | {len(concepts):>7,} CUIs seen",
                    file=sys.stderr,
                )

            cols = raw.rstrip("\n").split("|")
            if len(cols) < 17:
                continue
            if cols[_COL_SUPPRESS] != "N":
                continue

            lang = _LAT_TO_LANG.get(cols[_COL_LAT])
            if lang is None:
                continue

            text = cols[_COL_STR].strip()
            if not text:
                continue

            cui = cols[_COL_CUI]
            row_type = _sab_type(cols[_COL_SAB])

            if cui not in concepts:
                # [best_type, {lang: {lower: original}}]
                concepts[cui] = [row_type, defaultdict(dict)]
            else:
                concepts[cui][0] = _higher_priority(concepts[cui][0], row_type)

            lower = text.lower()
            alias_map: dict[str, dict[str, str]] = concepts[cui][1]
            if lower not in alias_map.get(lang, {}):
                if lang not in alias_map:
                    alias_map[lang] = {}
                alias_map[lang][lower] = text

    print(
        f"  [scan] done: {scanned:,} rows | {len(concepts):,} CUIs",
        file=sys.stderr,
    )

    # Emit
    written = 0
    for cui, (concept_type, alias_map) in concepts.items():
        if type_filter is not None and concept_type not in type_filter:
            continue
        aliases: list[dict] = []
        for lang, lower_to_text in alias_map.items():
            for text in lower_to_text.values():
                aliases.append({"text": text, "language": lang, "source": "umls"})
        if not aliases:
            continue
        obj = {
            "concept_id": f"umls:{cui}",
            "type": concept_type,
            "aliases": aliases,
        }
        out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        written += 1

    return written


# --- CLI --------------------------------------------------------------------


def _parse_types(arg: str | None) -> set[str] | None:
    if arg is None:
        return None
    requested = {t.strip() for t in arg.split(",") if t.strip()}
    bad = requested - _VALID_TYPES
    if bad:
        sys.exit(
            f"--types: unknown type(s) {sorted(bad)}. Valid: {sorted(_VALID_TYPES)}."
        )
    return requested


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Normalize UMLS MRCONSO.RRF into the ConceptRecord JSONL schema. "
            "Output goes to stdout unless --out is given."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        metavar="MRCONSO.RRF",
        help="Path to the unzipped MRCONSO.RRF file.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL file. Defaults to stdout.",
    )
    p.add_argument(
        "--types",
        default=None,
        help=(
            "Comma-separated subset of types to emit "
            "(drug,disease,symptom,procedure,other). "
            "Default: emit all."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.input.exists():
        sys.exit(f"--input: file not found: {args.input}")

    type_filter = _parse_types(args.types)

    out_fh: IO[str]
    close_out: bool
    if args.out is None:
        out_fh = sys.stdout
        close_out = False
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_fh = args.out.open("w", encoding="utf-8")
        close_out = True

    try:
        written = _normalize(args.input, out=out_fh, type_filter=type_filter)
    finally:
        if close_out:
            out_fh.close()

    target = str(args.out) if args.out is not None else "<stdout>"
    print(f"wrote {written:,} records to {target}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
