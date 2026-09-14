# Medical-clip fixture image set — provenance manifest

Required by Unit 2 of the vision-detection plan
([2026-06-14-001-feat-vision-detection-plan.md](../../../docs/plans/2026-06-14-001-feat-vision-detection-plan.md)).
The BiomedCLIP candidate-prompt tuning + threshold calibration smoke
test runs against this set.

## Hard rules

1. **Every image must be CC0 / public-domain.** This repo may be
   open-sourced (CLAUDE.md "Open-Source Hygiene"); shipping a
   copyrighted scan in `tests/fixtures/` would force a relicensing
   conversation later. Source from datasets whose license is
   explicitly CC0 — ISIC for dermoscopy, RSNA challenge releases for
   chest X-ray / CT, the public BUSI dataset for ultrasound, and any
   CC0 photo / screenshot for the non-medical buckets.

2. **30 images total — 5 per modality across 6 buckets.** The buckets
   match the `Modality` Literal in
   `src/claritymed/core/medical_clip/schemas.py`:
   `ultrasound`, `ct`, `xray`, `dermoscopy`, `photo`, `document`.
   The `unknown` bucket is excluded — it's a runtime fallback, not a
   training target.

3. **No PHI.** Even with a permissive license, scans must be either
   synthetic or de-identified. The dataset citations below have been
   sanity-checked; if you swap a source, re-check.

## Layout

```
tests/fixtures/medical_clip/
├── README.md                 (this file)
├── ultrasound/
│   ├── benign_001.png
│   ├── malignant_002.png
│   ├── normal_003.png
│   ├── benign_004.png
│   └── normal_005.png
├── ct/
│   ├── lung_001.png
│   ├── ...
├── xray/
├── dermoscopy/
├── photo/
└── document/
```

Filenames are `<short_descriptor>_<NNN>.<ext>` (3-digit index;
descriptors are flavor, not required to encode a label). Keep each
image under 1 MB to keep the repo lean.

## Provenance table

| Filename                       | Modality   | Source dataset / URL                                                  | License | Notes                          |
| ------------------------------ | ---------- | --------------------------------------------------------------------- | ------- | ------------------------------ |
| `ultrasound/benign_001.png`    | ultrasound | <to-be-filled by Unit 2 implementer; e.g. Kaggle BUSI public sample>  | CC0     | benign lesion, ~256×256        |
| `ultrasound/malignant_002.png` | ultrasound | <to-be-filled>                                                        | CC0     | malignant lesion               |
| `ultrasound/normal_003.png`    | ultrasound | <to-be-filled>                                                        | CC0     | no lesion                      |
| `ultrasound/benign_004.png`    | ultrasound | <to-be-filled>                                                        | CC0     |                                |
| `ultrasound/normal_005.png`    | ultrasound | <to-be-filled>                                                        | CC0     |                                |
| `ct/lung_001.png`              | ct         | <to-be-filled; e.g. RSNA Pulmonary Embolism Detection Challenge>       | CC0     | axial slice, lung window       |
| `ct/lung_002.png`              | ct         | <to-be-filled>                                                        | CC0     |                                |
| `ct/abdomen_003.png`           | ct         | <to-be-filled>                                                        | CC0     | abdominal slice                |
| `ct/head_004.png`              | ct         | <to-be-filled>                                                        | CC0     | head slice                     |
| `ct/chest_005.png`             | ct         | <to-be-filled>                                                        | CC0     | chest window                   |
| `xray/chest_001.png`           | xray       | <to-be-filled; e.g. RSNA Pneumonia Detection Challenge>                | CC0     | PA view chest                  |
| `xray/chest_002.png`           | xray       | <to-be-filled>                                                        | CC0     |                                |
| `xray/hand_003.png`            | xray       | <to-be-filled>                                                        | CC0     | extremity                      |
| `xray/spine_004.png`           | xray       | <to-be-filled>                                                        | CC0     | lateral spine                  |
| `xray/abdomen_005.png`         | xray       | <to-be-filled>                                                        | CC0     | KUB                            |
| `dermoscopy/nevus_001.jpg`     | dermoscopy | <to-be-filled; e.g. ISIC Archive CC0 subset>                          | CC0     | benign nevus                   |
| `dermoscopy/melanoma_002.jpg`  | dermoscopy | <to-be-filled>                                                        | CC0     | melanoma                       |
| `dermoscopy/seb_ker_003.jpg`   | dermoscopy | <to-be-filled>                                                        | CC0     | seborrheic keratosis           |
| `dermoscopy/bcc_004.jpg`       | dermoscopy | <to-be-filled>                                                        | CC0     | basal cell carcinoma           |
| `dermoscopy/scc_005.jpg`       | dermoscopy | <to-be-filled>                                                        | CC0     | squamous cell carcinoma        |
| `photo/landscape_001.jpg`      | photo      | <to-be-filled; e.g. Unsplash public-domain or Wikimedia Commons CC0>   | CC0     | outdoor landscape              |
| `photo/portrait_002.jpg`       | photo      | <to-be-filled>                                                        | CC0     | non-identifiable portrait      |
| `photo/object_003.jpg`         | photo      | <to-be-filled>                                                        | CC0     | inanimate object               |
| `photo/food_004.jpg`           | photo      | <to-be-filled>                                                        | CC0     |                                |
| `photo/pet_005.jpg`            | photo      | <to-be-filled>                                                        | CC0     |                                |
| `document/screenshot_001.png`  | document   | self-generated screenshot of a CC0 page                               | CC0     | text-heavy webpage             |
| `document/pdfpage_002.png`     | document   | self-generated; render a CC0 PDF page                                  | CC0     | research paper page            |
| `document/receipt_003.jpg`     | document   | self-generated; synthetic receipt or CC0 retail                        | CC0     |                                |
| `document/form_004.png`        | document   | self-generated; blank form                                            | CC0     |                                |
| `document/letter_005.png`      | document   | self-generated; Lorem ipsum letter                                    | CC0     |                                |

## Tuning loop

The values currently shipped in
`configs/medical_clip.yaml::tasks.modality` are starting points only.
Once this fixture set is in place:

1. Run the integration test
   `tests/integration/medical_clip/test_classify_modality.py` (Unit 2
   adds it) over the full set.
2. Inspect per-modality recall; tune `candidates[*].prompts` and the
   gating thresholds (`min_confidence`, `min_medical_confidence`) until
   each medical bucket clears ≥ 90 % top-1 recall and `photo` /
   `document` never claim `is_medical=true`.
3. Commit both the updated `configs/medical_clip.yaml` and this README
   in the same change so the calibration is reproducible.

## License footer

This README and the provenance table are CC0 — the same as every
image they reference. Adding a non-CC0 source is a license drift
event and must be caught at review time.
