#!/usr/bin/env python3
"""Normalize MeSH ``desc<YEAR>.xml`` and ``supp<YEAR>.xml`` into our JSONL schema.

WHY THIS SCRIPT EXISTS
----------------------
``init_terminology.py`` only knows how to ``--merge`` JSONL files that
already match our schema. Each upstream vocabulary needs its own
normalizer; this is the MeSH one. Per the architecture decision
documented in init_terminology.py's "CONCEPT_ID NAMING CONVENTION"
section, normalizers stay independent of ``SHARED_DIR``: they read an
upstream file, write our JSONL to stdout (or ``--out FILE``), and let
the operator decide whether to ``--merge`` it.

INPUT SHAPES
------------
``desc<YEAR>.xml`` and ``supp<YEAR>.xml`` (both downloaded from
``nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/``). Schema sketch
for desc::

    DescriptorRecord
      DescriptorUI                  → concept_id base
      DescriptorName/String         → primary alias
      TreeNumberList/TreeNumber[*]  → type inference (first letter)
      ConceptList/Concept[*]
        TermList/Term[*]
          String                    → alias
          @IsPermutedTermYN         → skip when "Y"

Supplementals are identical shape, with ``SupplementalRecord`` /
``SupplementalRecordUI`` / ``SupplementalRecordName`` instead, and an
``SCRClass`` attribute in place of TreeNumberList for type inference.

OUTPUT
------
One JSON object per line, matching the ``ConceptRecord`` schema in
``scripts/init_terminology.py``::

    {"concept_id":"mesh:D000001","type":"drug","aliases":[{"text":...}]}

TYPE INFERENCE
--------------
Descriptors: the **first** ``TreeNumber`` is canonical per MeSH
convention. Mapping by first character:

* ``D*``                       → drug
* ``C23.888.*``                → symptom (Signs and Symptoms subtree)
* ``C*`` (other)               → disease
* ``E*``                       → procedure
* anything else / missing tree → other

Supplementals: by ``SCRClass`` attribute:

* ``1`` (chemical)             → drug
* anything else                → other

USAGE
-----
::

    # MeSH desc + supp, current release
    uv run python scripts/normalize_mesh.py \\
        --desc data/download/desc2026.xml \\
        --supp data/download/supp2026.xml \\
        > /tmp/mesh.jsonl

    # then hand off to the universal merge entry
    uv run python scripts/init_terminology.py --merge /tmp/mesh.jsonl

    # restrict to drug + disease + symptom only (skip 'other' / 'procedure')
    uv run python scripts/normalize_mesh.py \\
        --desc desc2026.xml --supp supp2026.xml \\
        --types drug,disease,symptom \\
        --out /tmp/mesh.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path
from typing import IO

# Streaming parser tag-of-interest. We do not need to track every event;
# only ``end`` events for the two top-level record elements are needed,
# because each record is self-contained and we clear its subtree right
# after emitting.
_DESC_TAG = "DescriptorRecord"
_SUPP_TAG = "SupplementalRecord"

# Output type vocabulary mirrors ``ConceptType`` in core/rag/terms/base.py.
_VALID_TYPES = {"drug", "disease", "symptom", "procedure", "other"}

# Tree-letter → type mapping for descriptors. ``C23.888`` (Signs and
# Symptoms subtree) is a documented MeSH carve-out so we keep that
# override explicit rather than burying it in a generic dispatch table.
_TREE_LETTER_TO_TYPE: dict[str, str] = {
    "D": "drug",
    "C": "disease",
    "E": "procedure",
}

# SCRClass → type mapping for supplementals. Class 1 is "chemical" and
# accounts for >75% of supplementals; everything else is too fuzzy for
# v1 (organism / protocol / biocode) and gets bucketed to "other".
_SCRCLASS_TO_TYPE: dict[str, str] = {
    "1": "drug",
}

# Report progress every N records to stderr — large enough to not spam,
# small enough to feel responsive on a slow disk (~30s between updates
# on a typical laptop's SSD for desc; supp is ~10× more records).
_PROGRESS_INTERVAL = 5000


# --- type inference -----------------------------------------------------


def _descriptor_type(elem: ET.Element) -> str:
    """Pick the canonical type for a DescriptorRecord.

    MeSH places the canonical tree number first; secondary placements
    follow. We respect that order rather than re-prioritising
    ourselves, so output stays predictable against the upstream
    intent.
    """
    tree_list = elem.find("TreeNumberList")
    if tree_list is None:
        return "other"
    first = tree_list.find("TreeNumber")
    if first is None or not first.text:
        return "other"
    code = first.text.strip()
    if code.startswith("C23.888"):
        return "symptom"
    return _TREE_LETTER_TO_TYPE.get(code[:1], "other")


def _supplemental_type(elem: ET.Element) -> str:
    return _SCRCLASS_TO_TYPE.get(elem.get("SCRClass", ""), "other")


# --- alias extraction ---------------------------------------------------


def _record_aliases(elem: ET.Element, name_tag: str) -> list[str]:
    """Return the alias surface forms for a single MeSH record.

    Collapses every ``Term/String`` across every ``Concept`` in the
    record's ``ConceptList`` and prepends the record's primary name
    (``DescriptorName`` / ``SupplementalRecordName``) so it always leads
    the alias list even when the same string appears later as a Term.
    Permuted terms (``IsPermutedTermYN="Y"``) are skipped because they
    are inverted forms that pollute substring matching without adding
    semantic recall.

    Deduplication is case-insensitive on the lowered text — MeSH often
    repeats the preferred form as both DescriptorName and the first
    Concept's first Term.
    """
    seen_lower: set[str] = set()
    out: list[str] = []

    primary = elem.find(f"{name_tag}/String")
    if primary is not None and primary.text:
        text = primary.text.strip()
        key = text.lower()
        if text and key not in seen_lower:
            seen_lower.add(key)
            out.append(text)

    for term in elem.iterfind("ConceptList/Concept/TermList/Term"):
        if term.get("IsPermutedTermYN") == "Y":
            continue
        s = term.find("String")
        if s is None or not s.text:
            continue
        text = s.text.strip()
        if not text:
            continue
        key = text.lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        out.append(text)

    return out


# --- record streaming ---------------------------------------------------


def _iter_records(path: Path, tag: str) -> Iterator[tuple[str, ET.Element]]:
    """Yield ``(record_ui, element)`` pairs without holding the file in RAM.

    The caller must call ``element.clear()`` after it's done with the
    yielded element. We **also** clear it here on the next ``yield``
    cycle — but doing it eagerly in the caller's ``finally`` is what
    actually bounds memory when the caller does heavy work between
    records.
    """
    # ``iterparse`` keeps the root element growing as it appends new
    # children. Manually clearing each record element after ``end`` is
    # the documented pattern for keeping memory flat regardless of file
    # size.
    context = ET.iterparse(str(path), events=("end",))
    for event, elem in context:
        if elem.tag != tag:
            continue
        if tag == _DESC_TAG:
            ui_elem = elem.find("DescriptorUI")
        else:
            ui_elem = elem.find("SupplementalRecordUI")
        if ui_elem is None or not ui_elem.text:
            elem.clear()
            continue
        yield ui_elem.text.strip(), elem
        elem.clear()


def _emit_record(
    out: IO[str],
    *,
    concept_id: str,
    record_type: str,
    aliases: list[str],
) -> None:
    obj = {
        "concept_id": concept_id,
        "type": record_type,
        "aliases": [
            {"text": text, "language": "en", "source": "mesh"} for text in aliases
        ],
    }
    out.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _process_file(
    path: Path,
    *,
    tag: str,
    type_fn,
    out: IO[str],
    type_filter: set[str] | None,
    label: str,
) -> int:
    """Stream-process one MeSH XML file; return the number of records written.

    Records dropped by ``type_filter`` still count toward progress so
    the caller can see scan rate, but only emitted records contribute
    to the return value (used for the end-of-run summary).
    """
    written = 0
    scanned = 0
    for ui, elem in _iter_records(path, tag):
        scanned += 1
        record_type = type_fn(elem)
        if type_filter is not None and record_type not in type_filter:
            if scanned % _PROGRESS_INTERVAL == 0:
                print(
                    f"  [{label}] scanned {scanned:>7,} | written {written:>7,}",
                    file=sys.stderr,
                )
            continue
        aliases = _record_aliases(
            elem,
            name_tag="DescriptorName" if tag == _DESC_TAG else "SupplementalRecordName",
        )
        if not aliases:
            continue
        prefix = "mesh:"
        _emit_record(
            out,
            concept_id=f"{prefix}{ui}",
            record_type=record_type,
            aliases=aliases,
        )
        written += 1
        if scanned % _PROGRESS_INTERVAL == 0:
            print(
                f"  [{label}] scanned {scanned:>7,} | written {written:>7,}",
                file=sys.stderr,
            )
    print(
        f"  [{label}] done: scanned {scanned:,} | written {written:,}",
        file=sys.stderr,
    )
    return written


# --- CLI ----------------------------------------------------------------


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
            "Normalize MeSH desc / supp XML into the ConceptRecord JSONL "
            "schema. Output goes to stdout unless --out is given."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--desc",
        type=Path,
        default=None,
        help="Path to desc<YEAR>.xml (MeSH descriptors).",
    )
    p.add_argument(
        "--supp",
        type=Path,
        default=None,
        help="Path to supp<YEAR>.xml (MeSH supplemental records).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output file. Defaults to stdout.",
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
    if args.desc is None and args.supp is None:
        sys.exit("pass at least one of --desc / --supp.")
    for label, path in (("desc", args.desc), ("supp", args.supp)):
        if path is not None and not path.exists():
            sys.exit(f"--{label}: file not found: {path}")

    type_filter = _parse_types(args.types)

    out_fh: IO[str]
    if args.out is None:
        out_fh = sys.stdout
        close_out = False
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_fh = args.out.open("w", encoding="utf-8")
        close_out = True

    total = 0
    try:
        if args.desc is not None:
            total += _process_file(
                args.desc,
                tag=_DESC_TAG,
                type_fn=_descriptor_type,
                out=out_fh,
                type_filter=type_filter,
                label="desc",
            )
        if args.supp is not None:
            total += _process_file(
                args.supp,
                tag=_SUPP_TAG,
                type_fn=_supplemental_type,
                out=out_fh,
                type_filter=type_filter,
                label="supp",
            )
    finally:
        if close_out:
            out_fh.close()

    target = args.out if args.out is not None else "<stdout>"
    print(f"wrote {total:,} records to {target}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
