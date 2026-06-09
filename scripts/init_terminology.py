#!/usr/bin/env python3
"""Initialize the UMLS+CMeKG terminology export at ``shared/terminology/``.

WHY THIS SCRIPT EXISTS
----------------------
``term_service.active: umls_cmekg_local`` (see ``configs/retrieval.yaml``)
needs a ``concepts.jsonl`` file at ``$CLARITYMED_SHARED_DIR/terminology/``
(default ``~/.claritymed/shared/terminology/``). Without it, RAG-enabled
flows fail loud at startup. This script bootstraps that file with one of:

* ``--seed``        — write a built-in starter set of ~20 high-frequency
                      clinical concepts (drug / disease / symptom, EN+ZH).
                      Enough to make query expansion useful for smoke tests
                      and demos without a UMLS license.
* ``--merge FILE``  — append concepts from an external JSONL that already
                      matches the ``ConceptRecord`` schema (see below). Use
                      this after you've normalized a real UMLS / CMeKG /
                      RxNorm export with your preferred tooling.
* ``--validate``    — parse the existing file and print a summary
                      (concept count, alias count, per-language counts).
                      Read-only.

UMLS is *not* redistributed: we never ship MRCONSO.RRF. Operators run
their licensed export through their own normalizer, then ``--merge`` the
result. Same for CMeKG.

SCHEMA (one JSON object per line in ``concepts.jsonl``)
------------------------------------------------------
::

    {
      "concept_id": "umls:C0004057",
      "type": "drug" | "disease" | "symptom" | "procedure" | "other",
      "aliases": [
        {"text": "aspirin", "language": "en", "source": "seed"},
        {"text": "阿司匹林", "language": "zh", "source": "seed"}
      ]
    }

CONCEPT_ID NAMING CONVENTION
----------------------------
``concept_id`` is namespaced with a ``<source>:`` prefix to keep ids
from different upstream vocabularies in disjoint spaces. ``--merge``
upserts by ``concept_id``, so naive collisions across sources (MeSH
``D001241`` accidentally clashing with ICD ``D001241``) would corrupt
the merged dataset. Convention:

* ``umls:C0004057``   — UMLS Metathesaurus CUI
* ``mesh:D001241``    — MeSH Descriptor UI
* ``mesh:C551235``    — MeSH Supplementary Record UI
* ``cmekg:<id>``      — CMeKG entity id
* ``rxnorm:<rxcui>``  — RxNorm concept id
* ``snomed:<sctid>``  — SNOMED-CT concept id

The schema itself stays liberal (any string is accepted), so a
normalizer can pick any prefix scheme; the rule is enforced by the
normalizer's output, not by validation. This is the same trade-off we
make for ``aliases[].source`` — a literal label, not a foreign key.

The expansion layer (``expand_query`` in ``core/rag/terms/expansion.py``)
deduplicates aliases by lowercased ``text``, so two records pointing
at the same drug under different prefixes produce one expansion entry
per unique alias — no need to canonicalize records up front.

USAGE
-----
::

    uv run python scripts/init_terminology.py --seed
    uv run python scripts/init_terminology.py --validate
    uv run python scripts/init_terminology.py --merge path/to/my-umls-export.jsonl
    uv run python scripts/init_terminology.py --seed --dry-run
    uv run python scripts/init_terminology.py --seed --force   # overwrite existing
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Allow ``python scripts/init_terminology.py ...`` without ``uv run`` when
# the working tree is on PYTHONPATH already; fall back to repo layout.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from claritymed.stores.paths import (  # noqa: E402
    shared_terminology_dir,
    shared_terminology_jsonl,
)

# --- starter dataset ----------------------------------------------------
# Curated for first-run usefulness: pairs each surface with at least one
# common synonym + the canonical EN↔ZH alias. Kept small and inlined so
# the script is self-contained — no external download, no license issue.
SEED_CONCEPTS: list[dict] = [
    {
        "concept_id": "seed:aspirin",
        "type": "drug",
        "aliases": [
            {"text": "aspirin", "language": "en", "source": "seed"},
            {"text": "acetylsalicylic acid", "language": "en", "source": "seed"},
            {"text": "ASA", "language": "en", "source": "seed"},
            {"text": "阿司匹林", "language": "zh", "source": "seed"},
            {"text": "乙酰水杨酸", "language": "zh", "source": "seed"},
        ],
    },
    {
        "concept_id": "seed:ibuprofen",
        "type": "drug",
        "aliases": [
            {"text": "ibuprofen", "language": "en", "source": "seed"},
            {"text": "Advil", "language": "en", "source": "seed"},
            {"text": "Motrin", "language": "en", "source": "seed"},
            {"text": "布洛芬", "language": "zh", "source": "seed"},
        ],
    },
    {
        "concept_id": "seed:metformin",
        "type": "drug",
        "aliases": [
            {"text": "metformin", "language": "en", "source": "seed"},
            {"text": "Glucophage", "language": "en", "source": "seed"},
            {"text": "二甲双胍", "language": "zh", "source": "seed"},
        ],
    },
    {
        "concept_id": "seed:acetaminophen",
        "type": "drug",
        "aliases": [
            {"text": "acetaminophen", "language": "en", "source": "seed"},
            {"text": "paracetamol", "language": "en", "source": "seed"},
            {"text": "Tylenol", "language": "en", "source": "seed"},
            {"text": "扑热息痛", "language": "zh", "source": "seed"},
            {"text": "对乙酰氨基酚", "language": "zh", "source": "seed"},
        ],
    },
    {
        "concept_id": "seed:amoxicillin",
        "type": "drug",
        "aliases": [
            {"text": "amoxicillin", "language": "en", "source": "seed"},
            {"text": "Amoxil", "language": "en", "source": "seed"},
            {"text": "阿莫西林", "language": "zh", "source": "seed"},
        ],
    },
    {
        "concept_id": "seed:atorvastatin",
        "type": "drug",
        "aliases": [
            {"text": "atorvastatin", "language": "en", "source": "seed"},
            {"text": "Lipitor", "language": "en", "source": "seed"},
            {"text": "阿托伐他汀", "language": "zh", "source": "seed"},
        ],
    },
    {
        "concept_id": "seed:diabetes-mellitus",
        "type": "disease",
        "aliases": [
            {"text": "diabetes mellitus", "language": "en", "source": "seed"},
            {"text": "diabetes", "language": "en", "source": "seed"},
            {"text": "DM", "language": "en", "source": "seed"},
            {"text": "糖尿病", "language": "zh", "source": "seed"},
        ],
    },
    {
        "concept_id": "seed:hypertension",
        "type": "disease",
        "aliases": [
            {"text": "hypertension", "language": "en", "source": "seed"},
            {"text": "high blood pressure", "language": "en", "source": "seed"},
            {"text": "HTN", "language": "en", "source": "seed"},
            {"text": "高血压", "language": "zh", "source": "seed"},
        ],
    },
    {
        "concept_id": "seed:asthma",
        "type": "disease",
        "aliases": [
            {"text": "asthma", "language": "en", "source": "seed"},
            {"text": "bronchial asthma", "language": "en", "source": "seed"},
            {"text": "哮喘", "language": "zh", "source": "seed"},
            {"text": "支气管哮喘", "language": "zh", "source": "seed"},
        ],
    },
    {
        "concept_id": "seed:coronary-artery-disease",
        "type": "disease",
        "aliases": [
            {"text": "coronary artery disease", "language": "en", "source": "seed"},
            {"text": "CAD", "language": "en", "source": "seed"},
            {"text": "冠心病", "language": "zh", "source": "seed"},
            {"text": "冠状动脉粥样硬化性心脏病", "language": "zh", "source": "seed"},
        ],
    },
    {
        "concept_id": "seed:hemorrhage",
        "type": "symptom",
        "aliases": [
            {"text": "hemorrhage", "language": "en", "source": "seed"},
            {"text": "bleeding", "language": "en", "source": "seed"},
            {"text": "出血", "language": "zh", "source": "seed"},
        ],
    },
    {
        "concept_id": "seed:anemia",
        "type": "disease",
        "aliases": [
            {"text": "anemia", "language": "en", "source": "seed"},
            {"text": "low hemoglobin", "language": "en", "source": "seed"},
            {"text": "贫血", "language": "zh", "source": "seed"},
        ],
    },
    {
        "concept_id": "seed:hemoglobin",
        "type": "other",
        "aliases": [
            {"text": "hemoglobin", "language": "en", "source": "seed"},
            {"text": "Hb", "language": "en", "source": "seed"},
            {"text": "HGB", "language": "en", "source": "seed"},
            {"text": "血红蛋白", "language": "zh", "source": "seed"},
        ],
    },
    {
        "concept_id": "seed:headache",
        "type": "symptom",
        "aliases": [
            {"text": "headache", "language": "en", "source": "seed"},
            {"text": "head pain", "language": "en", "source": "seed"},
            {"text": "cephalalgia", "language": "en", "source": "seed"},
            {"text": "头痛", "language": "zh", "source": "seed"},
        ],
    },
    {
        "concept_id": "seed:nausea",
        "type": "symptom",
        "aliases": [
            {"text": "nausea", "language": "en", "source": "seed"},
            {"text": "queasiness", "language": "en", "source": "seed"},
            {"text": "恶心", "language": "zh", "source": "seed"},
        ],
    },
    {
        "concept_id": "seed:fever",
        "type": "symptom",
        "aliases": [
            {"text": "fever", "language": "en", "source": "seed"},
            {"text": "pyrexia", "language": "en", "source": "seed"},
            {"text": "elevated temperature", "language": "en", "source": "seed"},
            {"text": "发烧", "language": "zh", "source": "seed"},
            {"text": "发热", "language": "zh", "source": "seed"},
        ],
    },
    {
        "concept_id": "seed:chest-pain",
        "type": "symptom",
        "aliases": [
            {"text": "chest pain", "language": "en", "source": "seed"},
            {"text": "thoracic pain", "language": "en", "source": "seed"},
            {"text": "胸痛", "language": "zh", "source": "seed"},
        ],
    },
    {
        "concept_id": "seed:dyspnea",
        "type": "symptom",
        "aliases": [
            {"text": "dyspnea", "language": "en", "source": "seed"},
            {"text": "shortness of breath", "language": "en", "source": "seed"},
            {"text": "SOB", "language": "en", "source": "seed"},
            {"text": "呼吸困难", "language": "zh", "source": "seed"},
        ],
    },
    {
        "concept_id": "seed:cough",
        "type": "symptom",
        "aliases": [
            {"text": "cough", "language": "en", "source": "seed"},
            {"text": "tussis", "language": "en", "source": "seed"},
            {"text": "咳嗽", "language": "zh", "source": "seed"},
        ],
    },
    {
        "concept_id": "seed:fatigue",
        "type": "symptom",
        "aliases": [
            {"text": "fatigue", "language": "en", "source": "seed"},
            {"text": "tiredness", "language": "en", "source": "seed"},
            {"text": "乏力", "language": "zh", "source": "seed"},
            {"text": "疲劳", "language": "zh", "source": "seed"},
        ],
    },
]

_VALID_TYPES = {"drug", "disease", "symptom", "procedure", "other"}
_VALID_LANGS = {"en", "zh"}


# --- helpers ------------------------------------------------------------


def _validate_record(obj: dict, line_no: int | None = None) -> None:
    """Raise ValueError when ``obj`` doesn't match the ConceptRecord schema."""
    loc = f" (line {line_no})" if line_no is not None else ""
    for key in ("concept_id", "type", "aliases"):
        if key not in obj:
            raise ValueError(f"missing {key!r}{loc}")
    if obj["type"] not in _VALID_TYPES:
        raise ValueError(f"type={obj['type']!r} not in {sorted(_VALID_TYPES)}{loc}")
    if not isinstance(obj["aliases"], list) or not obj["aliases"]:
        raise ValueError(f"aliases must be non-empty list{loc}")
    for alias in obj["aliases"]:
        for key in ("text", "language", "source"):
            if key not in alias:
                raise ValueError(f"alias missing {key!r}{loc}")
        if alias["language"] not in _VALID_LANGS:
            raise ValueError(
                f"alias language={alias['language']!r} not in "
                f"{sorted(_VALID_LANGS)}{loc}"
            )


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_no}: bad JSON: {exc}") from exc
            _validate_record(obj, line_no)
            out.append(obj)
    return out


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --- commands -----------------------------------------------------------


def cmd_seed(target: Path, *, force: bool, dry_run: bool) -> int:
    if target.exists() and not force:
        print(
            f"refusing to overwrite existing {target} — pass --force to replace, "
            f"or --merge to add to it.",
            file=sys.stderr,
        )
        return 1
    for rec in SEED_CONCEPTS:
        _validate_record(rec)
    if dry_run:
        print(f"[dry-run] would write {len(SEED_CONCEPTS)} concepts to {target}")
        return 0
    _write_jsonl(target, SEED_CONCEPTS)
    print(f"wrote {len(SEED_CONCEPTS)} concepts to {target}")
    return 0


def cmd_merge(target: Path, source: Path, *, dry_run: bool) -> int:
    if not source.exists():
        print(f"source not found: {source}", file=sys.stderr)
        return 1
    incoming = _read_jsonl(source)
    existing = _read_jsonl(target) if target.exists() else []
    by_id: dict[str, dict] = {r["concept_id"]: r for r in existing}
    added = 0
    replaced = 0
    for rec in incoming:
        if rec["concept_id"] in by_id:
            replaced += 1
        else:
            added += 1
        by_id[rec["concept_id"]] = rec
    merged = list(by_id.values())
    if dry_run:
        print(
            f"[dry-run] merge {source} → {target}: "
            f"+{added} new, {replaced} replaced, total {len(merged)}"
        )
        return 0
    _write_jsonl(target, merged)
    print(
        f"merged {source} → {target}: "
        f"+{added} new, {replaced} replaced, total {len(merged)}"
    )
    return 0


def cmd_validate(target: Path) -> int:
    if not target.exists():
        print(f"not found: {target}", file=sys.stderr)
        return 1
    records = _read_jsonl(target)
    if not records:
        print(f"{target} is empty (0 concepts)")
        return 0
    type_counts = Counter(r["type"] for r in records)
    lang_counts: Counter = Counter()
    source_counts: Counter = Counter()
    total_aliases = 0
    for r in records:
        for a in r["aliases"]:
            lang_counts[a["language"]] += 1
            source_counts[a["source"]] += 1
            total_aliases += 1
    print(f"file:           {target}")
    print(f"concepts:       {len(records)}")
    print(f"aliases total:  {total_aliases}")
    print(f"by type:        {dict(type_counts.most_common())}")
    print(f"by language:    {dict(lang_counts.most_common())}")
    print(f"by source:      {dict(source_counts.most_common())}")
    return 0


# --- CLI ----------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Initialize the UMLS+CMeKG terminology export.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--target",
        type=Path,
        default=None,
        help=(
            "Output path for concepts.jsonl. Defaults to "
            "$CLARITYMED_SHARED_DIR/terminology/concepts.jsonl."
        ),
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--seed",
        action="store_true",
        help="Write the built-in starter set (~20 concepts).",
    )
    mode.add_argument(
        "--merge",
        type=Path,
        metavar="FILE",
        help="Append concepts from an external JSONL (same schema).",
    )
    mode.add_argument(
        "--validate",
        action="store_true",
        help="Read the existing file and print a summary.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="With --seed: overwrite an existing file instead of refusing.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without touching disk.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    target = args.target or shared_terminology_jsonl()
    if args.seed:
        if not target.parent.exists() and not args.dry_run:
            shared_terminology_dir().mkdir(parents=True, exist_ok=True)
        return cmd_seed(target, force=args.force, dry_run=args.dry_run)
    if args.merge is not None:
        return cmd_merge(target, args.merge, dry_run=args.dry_run)
    if args.validate:
        return cmd_validate(target)
    return 2  # unreachable — mutually exclusive group guarantees one branch


if __name__ == "__main__":
    sys.exit(main())
