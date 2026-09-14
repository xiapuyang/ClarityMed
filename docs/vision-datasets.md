# Vision datasets — sizes & splits

Sample counts come from an on-disk walk of the upstream archives at
`~/.claritymed/data/vision/<dataset>/`. The 9 shipped vision datasets
are listed below, grouped by `disease_id`.

## Common split rule

Every `dataset.py` discards the upstream `train/val/test` partition,
pools all images, and re-runs the same deterministic per-class
hash-stratified split:

| Split | Fraction |
|---|---|
| `train` | 0.70 |
| `val` | 0.15 |
| `test` | 0.15 |

Code: `stratified_split()` in each adapter. The salt is
`<dataset>-v1` (e.g. `busi-v1`, `rsna-pneumonia-v1`) so the
assignment is reproducible across machines and independent of
filesystem ordering. Per-class hashing guarantees rare classes
(e.g. BUSI `normal` at ~17%, RSNA `pneumonia` at ~22%) survive in
val + test.

Per-class arithmetic uses `int(n * 0.70)` / `int(n * 0.15)` and the
remainder goes to test, so test ends up marginally larger than val
on small classes.

## Summary

| Dataset | `disease_id` | Task | Classes | Total | Input | Notes |
|---|---|---|---|---|---|---|
| `busi` | `breast_cancer_ultrasound` | cls + seg | 3 | 780 | 256² | Ships lesion masks; multi-lesion samples OR-fused |
| `breast_us_kaggle` | `breast_cancer_ultrasound` | cls only | 2 | 9 016 | 256² | Upstream-augmented (rotation + sharpening) |
| `chest_ct` | `lung_cancer_chest_ct` | cls only | 4 | 1 000 | 256² | Class folder names carry staging suffixes |
| `chest_xray_pneumonia` | `chest_xray_pneumonia` | cls only | 2 | 5 856 | 256² | Pediatric (Kermany 2018, Guangzhou) |
| `colon_histopath` | `colon_cancer_histopathology` | cls only | 2 | 10 000 | 256² | LC25000 colon subset |
| `lung_histopath` | `lung_cancer_histopathology` | cls only | 3 | 14 999 | 256² | LC25000 lung subset (shares archive with colon) |
| `rsna_pneumonia` | `chest_xray_pneumonia` | cls only | 2 | 26 684 | **128²** | Adult DICOMs → cached PNG; input downsized for IO |
| `rsna_pneumonia_yolo` | `chest_xray_pneumonia` | detection | 1 | 26 684 | native | Same archive as above; YOLO label files + symlinks |
| `skin_lesion` | `skin_lesion` | cls only | 9 | 2 357 | 256² | ISIC 9-class, long-tailed |

Total: **~96.6k images** across 9 adapters / 8 underlying archives.

## Per-class breakdown

### `busi` — 780 images

| Class | Count | Share |
|---|---|---|
| benign | 437 | 56.0% |
| malignant | 210 | 26.9% |
| normal | 133 | 17.1% |

`unet_resnet50` ModelSpec; segmentation masks paired with every
image (normal carries an all-zero mask).

### `breast_us_kaggle` — 9 016 images

| Class | Count | Share |
|---|---|---|
| benign | 4 574 | 50.7% |
| malignant | 4 442 | 49.3% |

Upstream-augmented set; no `normal` class. Alternate 2-class training
source for `breast_cancer_ultrasound`.

### `chest_ct` — 1 000 images

| Class | Count | Share |
|---|---|---|
| adenocarcinoma | 338 | 33.8% |
| large_cell_carcinoma | 187 | 18.7% |
| normal | 215 | 21.5% |
| squamous_cell_carcinoma | 260 | 26.0% |

Upstream class-folder names embed TNM staging (e.g.
`adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/`); the discover step
normalises to the four canonical labels.

### `chest_xray_pneumonia` — 5 856 images (Kermany 2018)

| Class | Count | Share |
|---|---|---|
| normal | 1 583 | 27.0% |
| pneumonia | 4 273 | 73.0% |

Skewed toward positive. Upstream `val/` is only 16 images so the
re-split is mandatory.

### `colon_histopath` — 10 000 images (LC25000 colon subset)

| Class | Count | Share |
|---|---|---|
| adenocarcinoma | 5 000 | 50.0% |
| normal | 5 000 | 50.0% |

Perfectly balanced upstream.

### `lung_histopath` — 14 999 images (LC25000 lung subset)

| Class | Count | Share |
|---|---|---|
| adenocarcinoma | 5 000 | 33.3% |
| normal | 4 999 | 33.3% |
| squamous_cell_carcinoma | 5 000 | 33.3% |

Shares the LC25000 archive with `colon_histopath` — one download,
two disease modules.

### `rsna_pneumonia` — 26 684 unique patients

| Class | Count | Share |
|---|---|---|
| normal | 20 672 | 77.5% |
| pneumonia | 6 012 | 22.5% |

Image count = unique `patientId` count after collapsing the
bbox-row CSV (`stage_2_train_labels.csv` has 30 228 rows; positives
have multiple rows per patient, one per bbox — collapsed with
`max(Target)`).

Input is **128²**, not 256² like the rest, because per-epoch PIL
resize on ~14k 1024² PNGs dominated wall-clock at the original
size. Full-res PNGs stay in `png_cache/` for the YOLO adapter.

### `rsna_pneumonia_yolo` — 26 684 unique patients

| Class | Bbox-positive images | Empty-label images |
|---|---|---|
| pneumonia (id 0) | 6 012 | 20 672 |

Single-class detection. Negative images get an empty label file
per the Ultralytics convention. The classification adapter's
DICOM→PNG cache is reused, so this adapter pays no decode cost on
top.

### `skin_lesion` — 2 357 images (ISIC 9-class)

| Class | Count | Share |
|---|---|---|
| actinic_keratosis | 130 | 5.5% |
| basal_cell_carcinoma | 392 | 16.6% |
| dermatofibroma | 111 | 4.7% |
| melanoma | 454 | 19.3% |
| nevus | 373 | 15.8% |
| pigmented_benign_keratosis | 478 | 20.3% |
| seborrheic_keratosis | 80 | 3.4% |
| squamous_cell_carcinoma | 197 | 8.4% |
| vascular_lesion | 142 | 6.0% |

Long-tailed: the largest class (pigmented_benign_keratosis) is
~6× the smallest (seborrheic_keratosis).

## Why per-epoch wall-clock varies so much

Same forge framework, same `batch_size=16`, same backbone pool, but
training time per epoch tracks sample count almost linearly. Order
of magnitude at bs=16:

| Dataset | Train batches / epoch | Relative |
|---|---|---|
| `busi` | ~34 | 1× |
| `chest_ct` | ~44 | 1.3× |
| `skin_lesion` | ~103 | 3× |
| `chest_xray_pneumonia` | ~256 | 7.5× |
| `breast_us_kaggle` | ~395 | 12× |
| `colon_histopath` | ~438 | 13× |
| `lung_histopath` | ~656 | 19× |
| `rsna_pneumonia` | ~1 168 | 34× |

(`train_batches ≈ int(total × 0.7) / 16`.)

`busi/dataset_spec.py` caps `search_num_workers=(2, 2)` because at
~34 batches/epoch the (8, 4) default's worker-spawn cost
dominates — every other dataset is large enough to amortise the
default workers.
