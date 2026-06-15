"""Calibrate BiomedCLIP modality candidate prompts and gating thresholds.

One-shot helper. Extracts ~30-50 images per modality from the zip archives
in ``data/download/``, scores them through the same :class:`BiomedClipEngine`
the FastAPI server uses, and writes a recall / threshold report to
``data/cache/medical_clip_eval/report.md``.

The script intentionally **does not** edit ``configs/medical_clip.yaml`` --
threshold changes are a human review step (the report tells you which two
numbers to bump).

Usage::

    uv sync --extra medical-clip-server
    uv run python scripts/calibrate_medical_clip.py [--max-per-modality 40]

Sources are declared at the top of this file in ``MODALITY_SOURCES``.
Modalities without a configured source land in the report as MISSING; the
script does not invent fallbacks.
"""

from __future__ import annotations

import argparse
import logging
import random
import re
import statistics
import sys
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

# We import the engine and the canonical Modality vocabulary directly from
# the runtime package so the calibration uses exactly the same math (label
# embedding averaging, softmax temperature) as the server.
from claritymed.core.medical_clip.schemas import Modality
from claritymed.servers._devices import default_device
from claritymed.servers.medical_clip.biomed_clip import (
    BiomedClipEngine,
    ModalityCandidate,
)

# --- constants -----------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "medical_clip.yaml"
DATA_DIR = REPO_ROOT / "data" / "download"
REPORT_PATH = REPO_ROOT / "data" / "cache" / "medical_clip_eval" / "report.md"

DEFAULT_MAX_PER_MODALITY = 40
DEFAULT_SEED = 42

# Medical labels for the is_medical leakage analysis. Mirrors the server's
# definition in core/medical_clip/schemas.py: everything except photo /
# document / unknown counts as medical.
MEDICAL_LABELS: frozenset[Modality] = frozenset(
    {"ultrasound", "ct", "xray", "dermoscopy"}
)
ALL_MODALITIES: tuple[Modality, ...] = (
    "ultrasound",
    "ct",
    "xray",
    "dermoscopy",
    "photo",
    "document",
)

# Where the labeled images come from. Add a sibling entry to extend a modality
# (the script picks images across all sources up to ``max_per_modality``).
# ``include`` is a regex matched against the archive member's path; ``exclude``
# is an optional regex that filters matching members back out (used to drop
# BUSI's segmentation masks).
MODALITY_SOURCES: dict[Modality, list[dict[str, str]]] = {
    "ultrasound": [
        {
            "zip": "Breast Ultrasound Images Dataset(BUSI).zip",
            "include": r"Dataset_BUSI_with_GT/(benign|malignant|normal)/[^/]+\.png$",
            "exclude": r"_mask",
        },
    ],
    "ct": [
        {
            "zip": "Chest CT-Scan images Dataset.zip",
            "include": r"Data/(train|test|valid)/[^/]+/[^/]+\.png$",
            "exclude": "",
        },
    ],
    # xray / dermoscopy / photo / document: no zip in data/download/ yet.
    # Add them by dropping a CC0-licensed archive in data/download/ and
    # registering a sibling entry above. Until then the report will mark
    # them MISSING and skip them.
}

logger = logging.getLogger("calibrate_medical_clip")


# --- dataclasses ---------------------------------------------------------


@dataclass(frozen=True)
class ImageSample:
    """One image carried through the calibration pipeline."""

    bytes_: bytes
    true_label: Modality
    source_ref: str  # "<zip basename>::<member path>" -- useful in the report


@dataclass(frozen=True)
class Classification:
    """One scored sample."""

    sample: ImageSample
    top1_label: Modality
    top1_score: float
    scores: dict[Modality, float]

    @property
    def correct(self) -> bool:
        return self.top1_label == self.sample.true_label


# --- sampling ------------------------------------------------------------


def sample_modality(
    modality: Modality,
    sources: list[dict[str, str]],
    max_count: int,
    rng: random.Random,
) -> tuple[list[ImageSample], list[str]]:
    """Pick up to ``max_count`` images for one modality across its sources.

    Returns (samples, warnings). Missing zips produce a warning rather than
    a hard failure -- a single missing source shouldn't tank the run.
    """
    warnings: list[str] = []
    candidates: list[tuple[str, str]] = []  # (zip basename, member path)
    open_zips: dict[str, zipfile.ZipFile] = {}

    try:
        for source in sources:
            zip_name = source["zip"]
            zip_path = DATA_DIR / zip_name
            if not zip_path.exists():
                warnings.append(f"missing zip: {zip_name}")
                continue
            include = re.compile(source["include"])
            exclude_pattern = source.get("exclude") or ""
            exclude = re.compile(exclude_pattern) if exclude_pattern else None

            zf = zipfile.ZipFile(zip_path, "r")
            open_zips[zip_name] = zf
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if not include.search(info.filename):
                    continue
                if exclude is not None and exclude.search(info.filename):
                    continue
                candidates.append((zip_name, info.filename))

        if not candidates:
            warnings.append(f"no matching members for modality {modality!r}")
            return [], warnings

        rng.shuffle(candidates)
        picked = candidates[:max_count]

        samples: list[ImageSample] = []
        for zip_name, member in picked:
            zf = open_zips[zip_name]
            data = zf.read(member)
            samples.append(
                ImageSample(
                    bytes_=data,
                    true_label=modality,
                    source_ref=f"{zip_name}::{member}",
                )
            )
        return samples, warnings
    finally:
        for zf in open_zips.values():
            zf.close()


def gather_samples(
    max_per_modality: int, seed: int
) -> tuple[dict[Modality, list[ImageSample]], dict[Modality, list[str]]]:
    """Build the labeled set across every configured modality."""
    rng = random.Random(seed)
    by_modality: dict[Modality, list[ImageSample]] = {}
    warnings_by_modality: dict[Modality, list[str]] = {}
    for modality in ALL_MODALITIES:
        sources = MODALITY_SOURCES.get(modality, [])
        if not sources:
            warnings_by_modality[modality] = [
                "no source configured -- add an entry to MODALITY_SOURCES"
            ]
            by_modality[modality] = []
            continue
        samples, warnings = sample_modality(modality, sources, max_per_modality, rng)
        by_modality[modality] = samples
        warnings_by_modality[modality] = warnings
        logger.info(
            "sampled %d images for modality=%s (warnings=%d)",
            len(samples),
            modality,
            len(warnings),
        )
    return by_modality, warnings_by_modality


# --- engine + classification --------------------------------------------


def load_engine_from_yaml(cfg_path: Path) -> tuple[BiomedClipEngine, dict[str, Any]]:
    """Load the engine using exactly the YAML the server reads from."""
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    model_cfg = cfg["model"]
    modality_cfg = cfg["tasks"]["modality"]

    device_request = model_cfg.get("device", "auto")
    device = default_device() if device_request == "auto" else device_request

    candidates = [
        ModalityCandidate(label=entry["label"], prompts=tuple(entry["prompts"]))
        for entry in modality_cfg["candidates"]
    ]

    engine = BiomedClipEngine.load(
        model_id=model_cfg["model_id"],
        revision=(model_cfg.get("revision") or "").strip() or None,
        device=device,
        candidates=candidates,
    )
    return engine, cfg


def classify_all(
    engine: BiomedClipEngine, samples: list[ImageSample]
) -> list[Classification]:
    """Score every sample through the engine."""
    results: list[Classification] = []
    for sample in samples:
        scoreboard = engine.classify(sample.bytes_)
        score_map = {row.label: row.score for row in scoreboard}
        top1 = scoreboard[0]
        results.append(
            Classification(
                sample=sample,
                top1_label=top1.label,
                top1_score=top1.score,
                scores=score_map,
            )
        )
    return results


# --- analytics -----------------------------------------------------------


def confusion_matrix(
    classifications: list[Classification],
) -> dict[Modality, dict[Modality, int]]:
    """Build a true → predicted count matrix."""
    matrix: dict[Modality, dict[Modality, int]] = {
        true: dict.fromkeys(ALL_MODALITIES, 0) for true in ALL_MODALITIES
    }
    for c in classifications:
        matrix[c.sample.true_label][c.top1_label] += 1
    return matrix


def percentiles(values: list[float]) -> dict[str, float]:
    """5 / 10 / 25 / 50 / 75 / 95 percentiles. Empty list → all zeros."""
    if not values:
        return dict.fromkeys(("p5", "p10", "p25", "p50", "p75", "p95"), 0.0)
    sorted_vals = sorted(values)
    quantiles = statistics.quantiles(sorted_vals, n=100, method="inclusive")
    # statistics.quantiles returns n-1 cut points; index i is the (i+1)th percentile.
    return {
        "p5": quantiles[4],
        "p10": quantiles[9],
        "p25": quantiles[24],
        "p50": quantiles[49],
        "p75": quantiles[74],
        "p95": quantiles[94],
    }


def suggest_thresholds(
    classifications: list[Classification],
) -> dict[str, float | None]:
    """Propose loose + tight gates for both confidence floors.

    The loose recommendation tracks correct-prediction recall (P10 of correct
    top-1 scores keeps ≥ ~90% of correct predictions). The tight one tracks
    leakage prevention (max of P95-of-wrong and the loose floor).
    """
    correct_all = [c.top1_score for c in classifications if c.correct]
    wrong_all = [c.top1_score for c in classifications if not c.correct]
    correct_medical = [
        c.top1_score
        for c in classifications
        if c.correct and c.sample.true_label in MEDICAL_LABELS
    ]
    # Worst-case "non-medical-image looked medical": top-1 score on photo /
    # document samples that the model labelled with a medical class.
    leak_into_medical = [
        c.top1_score
        for c in classifications
        if c.sample.true_label not in MEDICAL_LABELS
        and c.sample.true_label != "unknown"
        and c.top1_label in MEDICAL_LABELS
    ]

    correct_pcts = percentiles(correct_all)
    wrong_pcts = percentiles(wrong_all)
    medical_pcts = percentiles(correct_medical)

    loose_min_conf = correct_pcts["p10"] if correct_all else None
    tight_min_conf = (
        max(wrong_pcts["p95"], loose_min_conf)
        if (wrong_all and loose_min_conf is not None)
        else loose_min_conf
    )
    loose_min_medical = medical_pcts["p10"] if correct_medical else None
    tight_min_medical = (
        max(max(leak_into_medical), loose_min_medical)
        if (leak_into_medical and loose_min_medical is not None)
        else loose_min_medical
    )

    return {
        "loose_min_confidence": loose_min_conf,
        "tight_min_confidence": tight_min_conf,
        "loose_min_medical_confidence": loose_min_medical,
        "tight_min_medical_confidence": tight_min_medical,
    }


# --- report rendering ----------------------------------------------------


def _fmt_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _fmt_percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator / denominator * 100:.1f}%"


def render_report(
    *,
    engine: BiomedClipEngine,
    cfg: dict[str, Any],
    samples_by_modality: dict[Modality, list[ImageSample]],
    warnings_by_modality: dict[Modality, list[str]],
    classifications: list[Classification],
    max_per_modality: int,
    seed: int,
) -> str:
    """Build the markdown report. Pure function -- no I/O."""
    lines: list[str] = []
    now = datetime.now(UTC).isoformat(timespec="seconds")
    gating = cfg["tasks"]["modality"]["gating"]

    lines.append("# Medical-CLIP modality calibration report")
    lines.append("")
    lines.append(f"- Generated: {now}")
    lines.append("- Config: configs/medical_clip.yaml")
    lines.append(f"- Model: {engine.model_id}")
    lines.append(f"- Revision: {engine.model_revision or 'unpinned'}")
    lines.append(f"- Device: {engine.device}")
    lines.append(f"- Seed: {seed}")
    lines.append(f"- Max per modality: {max_per_modality}")
    lines.append("")

    lines.append("## Coverage")
    lines.append("")
    lines.append("| Modality | Sampled | Notes |")
    lines.append("|---|---|---|")
    total = 0
    for modality in ALL_MODALITIES:
        samples = samples_by_modality.get(modality, [])
        warnings = warnings_by_modality.get(modality, [])
        total += len(samples)
        note = "; ".join(warnings) if warnings else "ok"
        if not samples:
            note = f"**MISSING** — {note}"
        lines.append(f"| {modality} | {len(samples)} | {note} |")
    lines.append("")
    lines.append(f"Total images scored: {total}")
    lines.append("")

    # Per-modality recall.
    lines.append("## Top-1 recall")
    lines.append("")
    lines.append("| Modality | Recall | Correct / Total |")
    lines.append("|---|---|---|")
    for modality in ALL_MODALITIES:
        per_mod = [c for c in classifications if c.sample.true_label == modality]
        correct = sum(1 for c in per_mod if c.correct)
        recall = _fmt_percent(correct, len(per_mod))
        lines.append(f"| {modality} | {recall} | {correct} / {len(per_mod)} |")
    lines.append("")

    lines.append("## Confusion matrix (rows = true label, cols = predicted)")
    lines.append("")
    matrix = confusion_matrix(classifications)
    header_cells = ["true \\ pred"] + list(ALL_MODALITIES)
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_cells)) + " |")
    for true_label in ALL_MODALITIES:
        cells = [true_label] + [str(matrix[true_label][p]) for p in ALL_MODALITIES]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Score distributions.
    lines.append("## Top-1 score distribution")
    lines.append("")
    correct_all = [c.top1_score for c in classifications if c.correct]
    wrong_all = [c.top1_score for c in classifications if not c.correct]
    correct_medical = [
        c.top1_score
        for c in classifications
        if c.correct and c.sample.true_label in MEDICAL_LABELS
    ]
    cohorts = [
        ("Correct (all)", correct_all),
        ("Incorrect (all)", wrong_all),
        ("Correct (medical only)", correct_medical),
    ]
    lines.append("| Cohort | n | P5 | P10 | P25 | P50 | P75 | P95 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for cohort_name, values in cohorts:
        pcts = percentiles(values)
        lines.append(
            "| "
            + " | ".join(
                [
                    cohort_name,
                    str(len(values)),
                    _fmt_score(pcts["p5"]),
                    _fmt_score(pcts["p10"]),
                    _fmt_score(pcts["p25"]),
                    _fmt_score(pcts["p50"]),
                    _fmt_score(pcts["p75"]),
                    _fmt_score(pcts["p95"]),
                ]
            )
            + " |"
        )
    lines.append("")

    # Suggested thresholds.
    lines.append("## Suggested thresholds")
    lines.append("")
    lines.append(
        "Current values in `configs/medical_clip.yaml::tasks.modality.gating`:"
    )
    lines.append("")
    lines.append(f"- `min_confidence`: {gating['min_confidence']}")
    lines.append(f"- `min_medical_confidence`: {gating['min_medical_confidence']}")
    lines.append("")
    suggestions = suggest_thresholds(classifications)
    lines.append(
        "**Loose** — preserve ~90% of correct predictions (use P10 of correct top-1 scores):"
    )
    lines.append("")
    lines.append(
        f"- `min_confidence`: {_fmt_score(suggestions['loose_min_confidence'])}"
    )
    lines.append(
        f"- `min_medical_confidence`: {_fmt_score(suggestions['loose_min_medical_confidence'])}"
    )
    lines.append("")
    lines.append(
        "**Tight** — also block leakage. `min_confidence` lifts to ≥ P95 of incorrect "
        "top-1 scores; `min_medical_confidence` lifts to ≥ the worst non-medical → medical "
        "top-1 score seen (when photo / document samples are available):"
    )
    lines.append("")
    lines.append(
        f"- `min_confidence`: {_fmt_score(suggestions['tight_min_confidence'])}"
    )
    lines.append(
        f"- `min_medical_confidence`: {_fmt_score(suggestions['tight_min_medical_confidence'])}"
    )
    lines.append("")
    lines.append(
        "Trade-off: the tighter the floor, the more correct predictions get demoted "
        "to `unknown` (LLM falls back to asking the user). When `loose` and `tight` "
        "agree to two decimals you can pick either. When they diverge, the gap is the "
        "ambiguity zone: think about whether a missed scan or an incorrectly accepted "
        "scan hurts more in your deployment context."
    )
    lines.append("")
    if not any(
        c for c in classifications if c.sample.true_label in {"photo", "document"}
    ):
        lines.append(
            "> **Note**: no `photo` / `document` samples were available, so the "
            "non-medical → medical leakage analysis could not run. The "
            "`min_medical_confidence` suggestions above are recall-only floors, not "
            "leakage-safe. Drop CC0 photo / screenshot fixtures into `data/download/` "
            "and rerun for a complete picture."
        )
        lines.append("")

    # Closing checklist.
    lines.append("## Recommended next steps")
    lines.append("")
    lines.append(
        "1. Inspect the confusion matrix above. Persistent confusion between two "
        "modalities (e.g. ct ↔ xray) is usually fixable by adding a more specific "
        "candidate prompt in `configs/medical_clip.yaml::tasks.modality.candidates`."
    )
    lines.append(
        "2. Pick a row from **Suggested thresholds** and edit "
        "`configs/medical_clip.yaml::tasks.modality.gating` by hand. This script "
        "intentionally does not auto-apply changes."
    )
    missing_modalities = [m for m in ALL_MODALITIES if not samples_by_modality.get(m)]
    if missing_modalities:
        joined = ", ".join(missing_modalities)
        lines.append(
            f"3. Backfill missing modalities ({joined}) by dropping CC0 zips into "
            "`data/download/` and registering them in `MODALITY_SOURCES` at the top "
            "of `scripts/calibrate_medical_clip.py`, then rerun."
        )
    lines.append("")
    return "\n".join(lines)


# --- entrypoint ----------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--max-per-modality",
        type=int,
        default=DEFAULT_MAX_PER_MODALITY,
        help=f"images sampled per modality (default {DEFAULT_MAX_PER_MODALITY})",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="RNG seed for deterministic sampling",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPORT_PATH,
        help=f"report path (default {REPORT_PATH.relative_to(REPO_ROOT)})",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not CONFIG_PATH.exists():
        logger.error("missing %s", CONFIG_PATH)
        return 2

    samples_by_modality, warnings_by_modality = gather_samples(
        args.max_per_modality, args.seed
    )
    flat_samples = [s for samples in samples_by_modality.values() for s in samples]
    if not flat_samples:
        logger.error("no samples gathered -- check MODALITY_SOURCES and %s", DATA_DIR)
        return 3

    try:
        engine, cfg = load_engine_from_yaml(CONFIG_PATH)
    except ImportError as exc:
        # open_clip is behind the optional medical-clip-server extra.
        logger.error(
            "BiomedCLIP runtime missing (%s); install with "
            "`uv sync --extra medical-clip-server`",
            exc,
        )
        return 4

    logger.info("scoring %d samples on %s", len(flat_samples), engine.device)
    classifications = classify_all(engine, flat_samples)

    report = render_report(
        engine=engine,
        cfg=cfg,
        samples_by_modality=samples_by_modality,
        warnings_by_modality=warnings_by_modality,
        classifications=classifications,
        max_per_modality=args.max_per_modality,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
