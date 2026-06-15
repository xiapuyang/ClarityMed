# Vision e2e fixtures — provenance

These images drive Unit 9's `tests/e2e/test_vision_*` tests and the
in-process integration test at
`tests/integration/orchestrator/test_vision_full_pipeline.py`. The
benchmark runner (`tests/benchmarks/tool_invoke/vision/`) reads the
same files via its own `seed` map.

Every image MUST be public-domain or CC0. ClarityMed is "open-source
hygiene by default" (CLAUDE.md) and any fixture leak into a published
repo carries legal risk. When adding a new file, append a row to the
table below with source URL + license.

## Subdirectories

| Subdir | What goes here | Use case |
|---|---|---|
| `busi/` | Real BUSI ultrasound images (benign / malignant / normal) | Happy-path e2e; modality match; specialist-keyword check |
| `modality_mismatch/` | CT / X-ray images | KTD-V3 hard refuse; tool never reaches `/v1/detect` |
| `report_overlay/` | Ultrasound with embedded clinician report text (FINDINGS/IMPRESSION) | KTD-V6 OCR override; tool short-circuits |
| `non_medical/` | Pet photos, document screenshots | R6 / `is_medical=false` filter |

## File rows

| Path | Source URL | License | Added |
|---|---|---|---|
| `busi/_PLACEHOLDER.txt` | — | — | 2026-06-14 |
| `modality_mismatch/_PLACEHOLDER.txt` | — | — | 2026-06-14 |
| `report_overlay/_PLACEHOLDER.txt` | — | — | 2026-06-14 |
| `non_medical/_PLACEHOLDER.txt` | — | — | 2026-06-14 |

Operators populating these directories should source from:

- **BUSI samples** — Kaggle `aryashah2k/breast-ultrasound-images-dataset`
  (Dataset_BUSI_with_GT) is CC0 per the dataset card.
- **CT / X-ray mismatch** — RSNA challenge public archives, NIH
  ChestX-ray14 (both public-domain for research).
- **Dermoscopy ceiling cases** — ISIC CC0 archive.
- **Report-overlay** — compose synthetic FINDINGS/IMPRESSION text onto
  a CC0 ultrasound base. Never use a real radiologist's report verbatim
  (PHI risk + copyright).
- **Non-medical** — Unsplash, Wikimedia Commons (filter for CC0).

The e2e tests `pytest.skip` when the relevant subdir is empty — they
do NOT generate fixtures on the fly. Empty subdirs are visible in CI as
"skipped" rather than "passed-but-unverified".
