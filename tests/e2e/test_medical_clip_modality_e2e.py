"""All-modality production-link smoke test for medical-clip.

What this covers
----------------

The bench (``tests/benchmarks/modality_classifier/run.py``) measures
per-modality accuracy with rich aggregation, but a bench run is opt-in
and never gates a PR. This e2e is the **always-on gate**: it walks the
same dataset pool but only enough samples per modality to catch a
regression cheaply (BiomedCLIP weights drift, a candidate-prompt YAML
edit that breaks one modality, an HTTP-shape change in the server
response). The call shape is the *exact* one the OCR worker uses
in production — :class:`MedicalClipClient` over loopback HTTP — so a
break in the wire path or the server's lifespan-precomputed embeddings
trips this test before it ever reaches a real OCR job.

Scope: the five concrete medical modalities. ``photo`` / ``document``
/ ``unknown`` are non-medical buckets the server uses as catch-alls —
they have no labeled dataset on disk, and asserting on them would
require hand-curated CC0 fixtures the repo doesn't ship yet. Adding
that bucket later is one ``DatasetSource`` away.

Skip semantics: a modality whose dataset isn't extracted on disk skips
cleanly — the test reports which modalities ran in the summary so a
partial environment (no histopath ZIP yet) still produces a useful
signal instead of a red X.

Tolerance: each modality must hit ``MIN_CORRECT_PER_MODALITY`` out of
``SAMPLES_PER_MODALITY``. The threshold is set to *almost* exact
match (4/5 = 80%) so a single hard image — a known long-tail —
doesn't flake the gate, but a real regression (whole modality
miscategorised) trips it.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import httpx
import pytest

from claritymed.core.medical_clip.client import MedicalClipClient
from tests.benchmarks.modality_classifier.samples import build_sample_pool

logger = logging.getLogger(__name__)

MEDICAL_CLIP_BASE_URL = "http://127.0.0.1:8086"

# Modalities backed by an on-disk dataset under ``~/.claritymed/data/vision``.
# Order matches the canonical ``Modality`` literal so a regression report
# reads in the same order the catalog YAML uses.
COVERED_MODALITIES: tuple[str, ...] = (
    "ultrasound",
    "ct",
    "xray",
    "dermoscopy",
    "histopathology",
)

# 5 samples / modality keeps the full sweep under ~1s on MPS yet leaves
# room for a 1-image miss without flaking. Raise for tighter calibration,
# lower for faster CI iteration.
SAMPLES_PER_MODALITY = 5
MIN_CORRECT_PER_MODALITY = 4

# Seeded so a re-run reproduces the same image set. Use a different
# constant from the bench's default seed (0) so this test and the bench
# don't accidentally tune against the same images.
SAMPLE_SEED = 42


@pytest.fixture(scope="module", autouse=True)
def _require_medical_clip_server() -> None:
    """Skip the module when medical-clip-server isn't reachable."""
    try:
        resp = httpx.get(f"{MEDICAL_CLIP_BASE_URL}/health", timeout=2.0)
    except httpx.HTTPError as exc:
        pytest.skip(
            f"medical-clip-server unreachable at {MEDICAL_CLIP_BASE_URL}: {exc}\n"
            "  Start it with: uv run --extra medical-clip-server "
            "claritymed-medical-clip-server"
        )
    if resp.status_code != 200:
        pytest.skip(f"medical-clip-server /health returned {resp.status_code}")


@pytest.fixture(scope="module")
def _sample_pool() -> list:
    """Build the full pool once; tests filter by modality."""
    pool = build_sample_pool(per_dataset=SAMPLES_PER_MODALITY, seed=SAMPLE_SEED)
    if not pool:
        pytest.skip(
            "no modality datasets extracted under ~/.claritymed/data/vision/. "
            "See claritymed.ingest.vision.<dataset>.download modules."
        )
    return pool


@pytest.mark.parametrize("expected_modality", COVERED_MODALITIES)
async def test_classify_modality_production_link(
    expected_modality: str,
    _sample_pool: list,
) -> None:
    """Each medical modality classifies correctly through the prod HTTP path."""
    samples = [s for s in _sample_pool if s.modality == expected_modality]
    if not samples:
        pytest.skip(
            f"no {expected_modality!r} samples on disk — run the matching "
            f"claritymed.ingest.vision.<dataset>.download module."
        )

    client = MedicalClipClient(base_url=MEDICAL_CLIP_BASE_URL)
    try:
        correct = 0
        misses: list[tuple[str, str, float]] = []
        for sample in samples:
            data = _read_image(sample.path)
            sha = hashlib.sha256(data).hexdigest()
            response = await client.classify_modality(
                data,
                request_id=f"e2e-mod-{sha[:8]}",
                sha256=sha,
            )
            if response.modality == expected_modality:
                correct += 1
            else:
                misses.append(
                    (sample.path.name, response.modality, float(response.confidence))
                )
    finally:
        await client.aclose()

    logger.info(
        "[medical_clip e2e %s] %d/%d correct (misses=%s)",
        expected_modality,
        correct,
        len(samples),
        misses,
    )
    assert correct >= MIN_CORRECT_PER_MODALITY, (
        f"{expected_modality}: {correct}/{len(samples)} correct, "
        f"need ≥ {MIN_CORRECT_PER_MODALITY}.\n"
        f"  misses: {misses}\n"
        "  Likely cause: a configs/medical_clip.yaml prompt edit broke this "
        "modality's embedding, or the BiomedCLIP revision shifted. Inspect "
        "the trial paths above before tuning prompts."
    )


def _read_image(path: Path) -> bytes:
    """Read raw image bytes. Pulled out so a future TIFF/DICOM converter
    can slot in here without touching the test bodies."""
    return path.read_bytes()
