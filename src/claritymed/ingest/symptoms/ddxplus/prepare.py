"""Data preparation for DDXPlus + the evidence-concept sidecar.

Three jobs:

1. ``--download``: pull the DDXPlus zips + JSONs from the upstream URL,
   verify each SHA-256 against the pinned manifest, and unpack to
   ``CLARITYMED_HOME/data/symptoms/ddxplus/``.
2. ``--check``: re-compute SHA-256 against the pinned manifest for an
   existing data dir; non-zero exit on mismatch.
3. ``--build-sidecar``: produce ``evidence_concepts.json`` for the
   ``term_service`` eligibility strategy by intersecting DDXPlus
   evidence names with the active TermService's concept lookup. Reports
   coverage %; <80% suggests a manual alias table is needed.

CLAUDE.md hygiene: no hardcoded auth in the URL constants. The SHA-256
table below stays as placeholders until an operator runs ``--download``
once, captures the real hashes, and commits the populated dict — at
which point downstream installs verify reproducibly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from claritymed import config as _cfg
from claritymed.ingest.symptoms.ddxplus.schema import (
    DDXPLUS_CONDITIONS_JSON,
    DDXPLUS_EVIDENCES_JSON,
    DDXPLUS_SPLIT_FILES,
)

# Public DDXPlus mirror. Verified against the figshare manifest the demo's
# README references. Override per-file by passing --base-url to operate
# against a local mirror.
DEFAULT_BASE_URL = "https://figshare.com/ndownloader/files/"

# {filename: sha256}. Populated on first successful --download run;
# committed so subsequent runs verify reproducibly. Empty values fall
# back to "warn but continue" — the chain still depends on the dataset
# URL + manifest sha256 on the model weights side (KTD-6).
DDXPLUS_SHA256: dict[str, str] = {
    DDXPLUS_EVIDENCES_JSON: "",
    DDXPLUS_CONDITIONS_JSON: "",
    DDXPLUS_SPLIT_FILES["train"]: "",
    DDXPLUS_SPLIT_FILES["validate"]: "",
    DDXPLUS_SPLIT_FILES["test"]: "",
}

# Per-filename upstream id mapping. Filled in after the first manual
# fetch from figshare lands; placeholders fail the download path loud
# until operators set them, so we never silently fetch the wrong file.
DDXPLUS_UPSTREAM_IDS: dict[str, str] = {}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _data_dir() -> Path:
    return _cfg.CLARITYMED_HOME / "data" / "symptoms" / "ddxplus"


def cmd_check(data_dir: Path) -> int:
    """Re-hash every known file under ``data_dir`` and compare to the table."""
    if not data_dir.exists():
        print(f"data dir not found: {data_dir}", file=sys.stderr)
        return 2
    mismatches = 0
    for filename, expected in DDXPLUS_SHA256.items():
        path = data_dir / filename
        if not path.exists():
            print(f"MISSING {filename}")
            mismatches += 1
            continue
        actual = _sha256_file(path)
        if not expected:
            print(f"UNVERIFIED {filename}: {actual} (no pinned hash)")
            continue
        if actual != expected:
            print(f"MISMATCH {filename}: expected {expected}, got {actual}")
            mismatches += 1
        else:
            print(f"OK {filename}")
    return 0 if mismatches == 0 else 1


def cmd_download(data_dir: Path, base_url: str) -> int:
    """Download every file in the table to ``data_dir`` and verify hashes."""
    if not DDXPLUS_UPSTREAM_IDS:
        print(
            "DDXPLUS_UPSTREAM_IDS is empty — populate it in prepare.py "
            "with the per-file figshare ids before --download will work. "
            "(Manual one-time setup; commit the populated dict.)",
            file=sys.stderr,
        )
        return 2
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, sha256 in DDXPLUS_SHA256.items():
        out = data_dir / filename
        if out.exists():
            print(f"SKIP {filename}: already present")
            continue
        upstream = DDXPLUS_UPSTREAM_IDS.get(filename)
        if not upstream:
            print(f"MISSING UPSTREAM ID {filename}", file=sys.stderr)
            return 2
        url = base_url.rstrip("/") + "/" + upstream
        print(f"GET {url} -> {out}")
        urllib.request.urlretrieve(url, out)
        actual = _sha256_file(out)
        if sha256 and actual != sha256:
            out.unlink(missing_ok=True)
            print(
                f"sha256 mismatch on {filename}: "
                f"expected {sha256}, got {actual}; deleted",
                file=sys.stderr,
            )
            return 1
        print(f"OK {filename} sha256={actual}")
    return 0


def build_evidence_concept_sidecar(
    data_dir: Path,
    *,
    out_path: Path | None = None,
) -> tuple[Path, float]:
    """Produce ``evidence_concepts.json`` for the term_service strategy.

    Iterates every DDXPlus evidence name and asks the active TermService
    for its top concept hit (language=``en``). The resulting map is
    consumed by ``core/symptoms/eligibility/term_service.py``. Returns
    ``(written_path, coverage)`` where coverage is the fraction of
    evidences with at least one concept hit (target ≥ 0.8).
    """
    from claritymed.core.rag.terms.factory import get_term_service
    from claritymed.ingest.symptoms.ddxplus.schema import load_evidence_schema

    schema = load_evidence_schema(data_dir)
    service = get_term_service()
    sidecar: dict[str, str] = {}
    total = 0
    for ev in schema["evs"]:
        name = ev["name"]
        total += 1
        hits = service.lookup(name, language="en")
        if not hits:
            continue
        # First hit is the highest-confidence match by TermService contract.
        concept_id = getattr(hits[0], "concept_id", None) or hits[0].id
        sidecar[name] = str(concept_id)
    coverage = (len(sidecar) / total) if total else 0.0
    target = out_path or (data_dir / "evidence_concepts.json")
    target.write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return target, coverage


def main() -> None:
    """CLI: ``--check`` / ``--download`` / ``--build-sidecar``."""
    ap = argparse.ArgumentParser(description="DDXPlus data preparation.")
    ap.add_argument("--data-dir", type=Path, default=_data_dir())
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    mx = ap.add_mutually_exclusive_group(required=True)
    mx.add_argument("--check", action="store_true")
    mx.add_argument("--download", action="store_true")
    mx.add_argument("--build-sidecar", action="store_true")
    args = ap.parse_args()
    if args.check:
        sys.exit(cmd_check(args.data_dir))
    if args.download:
        sys.exit(cmd_download(args.data_dir, args.base_url))
    if args.build_sidecar:
        path, coverage = build_evidence_concept_sidecar(args.data_dir)
        print(f"wrote {path}; coverage={coverage:.1%}")
        if coverage < 0.8:
            print(
                "WARNING: coverage below 80% — consider a manual alias "
                "backstop before relying on the term_service strategy.",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
