# Vision model workflow

Five-phase pipeline for training and promoting a per-disease vision
model. Mirrors `docs/symptoms-model-workflow.md` so operators can move
between features without relearning the steps.

Two parallel CLIs share the same artifact layout + MLflow experiment:

* `claritymed-vision-forge` — classification / segmentation (this doc's
  main subject).
* `claritymed-vision-yolo-forge` — Ultralytics-backed bounding-box
  detection (see "Detection pipeline (yolo_forge)" below).

Both select a per-(dataset, architecture) ModelSpec via
`--model <module>:<ATTR>`. The shipped specs:

| Disease | Spec | Task | Architectures searched |
|---|---|---|---|
| `breast_cancer_ultrasound` (BUSI) | `claritymed.ingest.vision.busi.models.unet_resnet50:UNET_RESNET50` | cls + seg | `resnet50` / `efficientnet_b0` / `custom_unet` U-Nets |
| `breast_cancer_ultrasound` (breast_us_kaggle) | `claritymed.ingest.vision.breast_us_kaggle.models.resnet50_v1:RESNET50_V1` | cls only | `resnet50` / `efficientnet_b0` / `efficientnet_b3` |
| `lung_cancer_chest_ct` (chest CT) | `claritymed.ingest.vision.chest_ct.models.resnet50_v1:RESNET50_V1` | cls only | `resnet50` / `efficientnet_b0` / `efficientnet_b3` |
| `chest_xray_pneumonia` (Kermany) | `claritymed.ingest.vision.chest_xray_pneumonia.models.resnet50_v1:RESNET50_V1` | cls only | `resnet50` |
| `chest_xray_pneumonia` (RSNA classification) | `claritymed.ingest.vision.rsna_pneumonia.models.resnet50_v1:RESNET50_V1` | cls only | `resnet50` |
| `chest_xray_pneumonia` (RSNA detection) | `claritymed.ingest.vision.rsna_pneumonia_yolo.models.yolov8n_v1:RSNA_YOLOV8N_V1` | detection (YOLO) | `yolov8n` |

Two of these diseases ship **multiple datasets** — `breast_cancer_ultrasound`
has BUSI (3-class with masks) and breast_us_kaggle (2-class augmented);
`chest_xray_pneumonia` carries three adapters against two archives:
Kermany (pediatric, classification training source), RSNA classification
(adult, classification training source — also the drift-bench target for
Kermany), and RSNA detection (same archive, YOLO bbox training via the
parallel yolo_forge pipeline). RSNA's bbox CSV rows get collapsed into
image-level labels by the classification adapter (`rsna_pneumonia/`)
and preserved as bbox label files by the detection adapter
(`rsna_pneumonia_yolo/`); both share the same DICOM→PNG cache. All
three ModelSpecs share `disease_id="chest_xray_pneumonia"`, so artifact
root, `LATEST.jsonl`, and MLflow experiment are one — `model_id`
discriminates rows and the regression gate filters per-model.

This doc walks the workflow against BUSI; chest CT and Kermany run the
same commands with their own `--model` flags. See "Other diseases"
below for per-spec specifics and "Adding a new dataset" for the
spec-only extension path.

## Prerequisites

- `uv sync --extra vision-server` — installs torch, torchvision,
  segmentation-models-pytorch (+ timm), mlflow, optuna, Pillow. The
  extra is self-contained — operators don't also need
  `--extra symptoms-server`.
- Kaggle credentials configured. Either `~/.kaggle/kaggle.json` or
  the `KAGGLE_USERNAME` + `KAGGLE_KEY` env vars in `.env`.
- An MPS or CUDA device. CPU-only training is unworkable (the train
  phase raises if neither is available).
- An MLflow tracking sink reachable. The default is the **shared** SQLite
  DB at `~/.claritymed/tracking/mlflow.db` (one DB across every feature +
  dataset; experiments disambiguated by name like
  `claritymed-vision-breast_cancer_ultrasound`). Override with
  `MLFLOW_TRACKING_URI` if a remote server is preferred.

## Step 1 — Download

```bash
uv run claritymed-vision-download-busi
```

Pulls `aryashah2k/breast-ultrasound-images-dataset` into
`~/.claritymed/data/vision/busi/Dataset_BUSI_with_GT/`. Idempotent;
pass `--force` to refetch.

Chest CT has its own download command —
`claritymed-vision-download-chest-ct` — pulling
`mohamedhanyyy/chest-ctscan-images` into
`~/.claritymed/data/vision/chest_ct/`. Same Kaggle credential story.

## Step 2 — Hyperparameter search

```bash
uv run claritymed-vision-forge hparam \
    --model claritymed.ingest.vision.busi.models.unet_resnet50:UNET_RESNET50 \
    --trials 20 --epochs 15
```

Optuna sweeps the spec's `hparam_space`. For BUSI's `UNET_RESNET50`
that's `backbone × lr × seg_loss_weight`. Trials persist to the shared
`~/.claritymed/tracking/optuna.db` under study name
`claritymed-vision-breast_cancer_ultrasound-hparam-<task_id>`;
Ctrl-C and resume safely by passing `--task-id` of the running study.

The composite score is `0.6 * malignant_recall + 0.4 * dice` — declared
on `UNET_RESNET50.task.composite_weights`. Malignant recall is weighted
higher because **a missed malignant finding is the worst error mode**
(plan §"Unit 6 Verification"). Overall accuracy is gated by floors but
not in the composite — it hides class imbalance.

The `backbone` axis is three real architectures, not a label-only knob:

| Value | Encoder | Pretrained |
|---|---|---|
| `resnet50` | torchvision ResNet50 inside an smp U-Net + aux classifier | ImageNet (~98 MB, one-time download to `~/.cache/torch/hub/`) |
| `efficientnet_b0` | torchvision EfficientNet-B0 inside an smp U-Net + aux classifier | ImageNet (~21 MB, same cache) |
| `custom_unet` | Hand-rolled base=32 U-Net + classification head from `forge_torch.py` | None (from-scratch baseline) |

First trial that picks `resnet50` or `efficientnet_b0` triggers the
ImageNet checkpoint fetch; subsequent trials reuse the cache. Inference
paths (server boot, tune phase) build the same architectures with
`pretrained=False` since the trained checkpoint already provides every
weight — no network round-trip at load time.

`--epochs` defaults to 15; lower values starve the segmentation head
(dice needs more iterations than classification to converge) and surface
as a `dice` deficit at the search-phase floor gate. See "Search-phase
gate failures" below if you hit it.

Smoke-run option for wiring verification:

```bash
uv run claritymed-vision-forge hparam \
    --model claritymed.ingest.vision.busi.models.unet_resnet50:UNET_RESNET50 \
    --smoke --trials 1
```

## Step 3 — Production training

```bash
uv run claritymed-vision-forge train \
    --model claritymed.ingest.vision.busi.models.unet_resnet50:UNET_RESNET50 \
    --max-epochs 100 --patience 15
```

Reads the best HP set from the search study (fails fast if step 2 didn't
run) and trains with patience-based early stopping on the val composite.
The high `--max-epochs` ceiling is deliberate: it lets the training
curve **visibly overfit** before patience triggers, so the operator can
confirm via `training_curve.json` that the early-stop window picked the
right epoch.

Outputs land in a timestamped staging dir under
`~/.claritymed/models/vision/breast_cancer_ultrasound/run/<model_id>_<ts>/`:

- `weights.pt` — **best-epoch** checkpoint (not last)
- `manifest.json` — fully populated per `core/vision/schemas.py::Manifest`;
  `tuned_inference` is `null` until step 4 runs. The `task` field is
  `"classification+segmentation"` for BUSI and `"classification"` for
  chest CT — `forge_torch` keys off this at server boot.
- `eval_metrics.json` — val + held-out **test** split breakdowns
  (per-spec metric keys — e.g. `malignant_recall`, `dice`, `accuracy`,
  `composite` for BUSI); the test composite is the deploy phase's
  regression-gate metric.
- `training_curve.json` — per-epoch `{train_loss, val_loss, val_score,
  val_*_breakdown}` so the overfitting onset is inspectable
- `provenance.json` — tracking URIs + run / study ids; deploy reads
  this verbatim into `LATEST.jsonl`

Manifest cross-check fields (validated at write time):

- `cancer_class: true` (driven by `DatasetSpec.cancer_class`)
- `cancer_status_mapping` / `clinical_action_mapping` — derived from
  `DatasetSpec.labels_meta`; no hand-edit needed in the model spec.
- `labels_meta` — per-label `{description, cancer_status,
  clinical_action}` triple, lifted from the DatasetSpec.
- `backbone` — the trained encoder choice. The adapter reads this at
  server boot to construct the matching architecture before loading
  weights — so a checkpoint trained on `resnet50` won't accidentally get
  loaded into the `custom_unet` shape. Not duplicated in
  `configs/vision.yaml::models[]`; the `manifest_sha256` pin already
  protects against config drift.

If the manifest validator rejects the write, fix the offending
`DatasetSpec` / `ModelSpec` field — **not** the validator — see CLAUDE.md
"Test Integrity".

## Step 4 — Tune (inference-time params)

```bash
uv run claritymed-vision-forge tune \
    --model claritymed.ingest.vision.busi.models.unet_resnet50:UNET_RESNET50 \
    --trials 30
```

**Inference-only Optuna**: cache one forward pass per (TTA on, TTA off)
on the val split, then sweep every inference-time parameter that
materially shifts the production composite. No retraining.

Search space comes from `ModelSpec.inference_space`. For BUSI's
`UNET_RESNET50`:

| Param | Range / values | Effect |
|---|---|---|
| `temperature` | 0.5–3.0 log-uniform | Logit scaling before softmax → calibrates `top1_prob` |
| `critical_threshold` | 0.20–0.70 uniform | Per-class cutoff for the critical label; mapped to `malignant` in the BUSI manifest |
| `seg_threshold` | 0.20–0.80 uniform | Sigmoid cutoff for mask binarization → dice (cls+seg only) |
| `confidence_low_max` / `confidence_medium_max` | 0.40–0.95 with `low_max < medium_max` | Confidence-tier boundaries; drives KTD-V10 frequency |
| `tta_default` | `{False, True}` | TTA-averaged logits as the catalog default |

Objective is the **same composite** as step 2, evaluated on the val
split. Single objective → no tradeoff knobs, no human cell-picking.

Naming note: `critical_threshold` is the generic forge name; the cls+seg
adapter exposes it as `malignant_threshold` on BUSI checkpoints via the
spec's `critical_labels=("malignant",)`. Other datasets pick their own
critical labels — chest CT's `RESNET50_V1` lists all three malignant
subtypes.

Outputs (in the same staging dir):

- `manifest.json` — `tuned_inference` field populated with the
  Optuna-picked params; loaded by `forge_torch` at server boot
- `eval_metrics.json` — adds `tuned_test_breakdown` + `tuned_test_score`
  for the deploy regression gate
- `provenance.json` — adds the tune phase's MLflow run id + Optuna
  study name

## Step 5 — Deploy (versioned, gated, auto-edit)

```bash
uv run claritymed-vision-forge deploy \
    --model claritymed.ingest.vision.busi.models.unet_resnet50:UNET_RESNET50
```

Promotes the most recent tune-finalised staging dir for the
`(disease_id, model_id)` pair if **both** gates pass:

1. **Floors** — read from `ModelSpec.task.floors.deploy`. For BUSI's
   `UNET_RESNET50`: `malignant_recall ≥ 0.85`, `accuracy ≥ 0.85`,
   `dice ≥ 0.70` on the held-out test split.
2. **Regression vs active** — candidate's `tuned_test_composite` **>**
   the last `LATEST.jsonl` row **for the same `model_id`**. Different
   architectures on the same dataset (e.g. a future YOLO alongside the
   U-Net) compare apples-to-apples because the gate filters by
   `model_id` before reading the previous composite. First deploy for a
   `(dataset, model_id)` pair skips this gate.

On pass:

- Copies the staging dir to a **versioned sibling stable path**:
  `~/.claritymed/models/vision/breast_cancer_ultrasound/breast_busi_unet_v1__<UTC-stamp>/`.
  Previous deploys stay on disk for rollback — the deploy step refuses
  to overwrite an existing versioned dir.
- Re-hashes the promoted `manifest.json` and **surgically patches**
  `configs/vision.yaml::models[i]` for the matching `model_id`,
  updating `weights_subpath` + `manifest_sha256`. The edit is
  line-based so the YAML's hand-written comments survive.
- Appends one row to
  `~/.claritymed/models/vision/breast_cancer_ultrasound/LATEST.jsonl`
  carrying the full audit trail — including the `model_id` so the
  per-model regression gate has the key it needs (see "Observability"
  below).

On fail: clear error explaining which floor or which composite delta
was too small; **no filesystem changes**, no YAML edit, no LATEST.jsonl
row.

First-time setup only (post first deploy):

```bash
# Flip the kill switch in configs/vision.yaml so the orchestrator picks
# up the disease. (Deploy never flips this — it's an explicit "we're
# turning the feature on" gesture.)
# diseases:
#   - id: breast_cancer_ultrasound
#     enabled: true          # <-- was false

# Register the unified pytorch factory (one-time, covers BUSI cls+seg
# and chest CT cls-only).
# Add to src/claritymed/servers/vision/adapters/__init__.py:
#     from claritymed.servers.vision.adapters import forge_torch
#     forge_torch.register()

# Restart the vision-server.
pkill -f claritymed-vision-server
uv run claritymed-vision-server &
```

The vision-server's startup will:

1. Hash `manifest.json` and compare to `manifest_sha256` (KTD-V7 root).
2. Hash `weights.pt` and compare to `manifest.json::sha256_weights`.
3. Read `manifest.task` and route to `ClsSegAdapter` (cls+seg) or
   `ClsAdapter` (cls-only) — both registered under the `pytorch`
   framework key.
4. Construct the matching architecture and run a one-time forward pass
   on a tiny random tensor (smoke check).
5. Read `manifest.tuned_inference` and apply the tuned params at
   inference time (temperature, critical threshold, seg threshold,
   confidence-tier boundaries, TTA default).

Mismatch at any step aborts startup with a clear error pointing at the
offending file.

## One-shot orchestrator

```bash
uv run claritymed-vision-forge pipeline \
    --model claritymed.ingest.vision.busi.models.unet_resnet50:UNET_RESNET50 \
    --trials 20 --search-epochs 15 \
    --max-epochs 100 --patience 15 --tune-trials 30
```

Runs steps 2 → 5 in order. `--phases tune,deploy` (etc.) runs a
subset; `--staging-dir <path>` resumes from a specific staging dir.

Two short-circuit modes for first runs (see also "Smoke test" and
"Quick mode" below):

- `--smoke` — synthetic everything (HP, weights, metrics). Sub-second.
  Verifies CLI / phase wiring without torch or a dataset.
- `--quick` — real torch + real dataset, microscopic budgets, gates
  downgraded. Minutes, not hours. Smallest honest end-to-end.

`--search-epochs` is the per-trial epoch budget inside the search phase
and is independent of `--max-epochs` (which is the train-phase ceiling).
Bump it before `--trials` if a search-phase gate fails on a seg metric:
extra trials at too-few epochs only re-sample the same underfit ceiling.

`--force` downgrades the inter-phase floor gates to warnings (diagnostic
runs only). `--deploy-force` additionally skips the deploy-phase floor
gate; combined, you can let a sub-floor run flow all the way through to
artifact promotion, but the regression gate still applies. Don't lower
the floors in the spec to make a run pass; that defeats the purpose.

## Other diseases — chest CT

Chest CT runs identically with its own `--model` flag:

```bash
uv run claritymed-vision-forge pipeline \
    --model claritymed.ingest.vision.chest_ct.models.resnet50_v1:RESNET50_V1
```

Per-spec differences from BUSI:

| Knob | BUSI `UNET_RESNET50` | chest CT `RESNET50_V1` |
|---|---|---|
| Task | `classification+segmentation` | `classification` |
| Critical metric | `malignant_recall` | `cancer_recall` (union of three malignant labels vs `normal`) |
| Composite | `0.6 × malignant_recall + 0.4 × dice` | `0.6 × cancer_recall + 0.4 × accuracy` |
| Deploy floors | `malignant_recall ≥ 0.85`, `accuracy ≥ 0.85`, `dice ≥ 0.70` | `cancer_recall ≥ 0.85`, `accuracy ≥ 0.80` (no dice) |
| Hparam space | `backbone × lr × seg_loss_weight` | `backbone × lr × weight_decay` |
| Backbones | `resnet50` / `efficientnet_b0` / `custom_unet` | `resnet50` / `efficientnet_b0` / `efficientnet_b3` |
| Artifact root | `~/.claritymed/models/vision/breast_cancer_ultrasound/` | `~/.claritymed/models/vision/lung_cancer_chest_ct/` |

Accuracy is the secondary composite metric on chest CT because dice is
N/A (the upstream dataset doesn't ship masks). Accuracy is intentionally
floored lower (0.80 vs BUSI's 0.85) because 4-class is intrinsically
harder than 3-class (random baseline 0.25 vs 0.33) — too tight a floor
would gate out trainable models.

## Other diseases — chest X-ray pneumonia (Kermany)

The Kermany 2018 dataset (pediatric chest X-rays, ~5,856 images,
NORMAL/PNEUMONIA) is the trained source for `chest_xray_pneumonia`:

```bash
uv run python -m claritymed.ingest.vision.chest_xray_pneumonia.download
uv run claritymed-vision-forge pipeline \
    --model claritymed.ingest.vision.chest_xray_pneumonia.models.resnet50_v1:RESNET50_V1
```

Per-spec differences from chest CT (the closest analog — both are
cls-only, both use ResNet-50):

| Knob | chest CT `RESNET50_V1` | Kermany `RESNET50_V1` |
|---|---|---|
| Classes | 4 (cancer subtypes + normal) | 2 (normal / pneumonia) |
| Critical metric | `cancer_recall` (union of 3 cancers vs normal) | `pneumonia_recall` (single class) |
| Composite | `0.6 × cancer_recall + 0.4 × accuracy` | `0.7 × pneumonia_recall + 0.3 × accuracy` |
| Deploy floors | `cancer_recall ≥ 0.85`, `accuracy ≥ 0.80` | `pneumonia_recall ≥ 0.85`, `accuracy ≥ 0.80` |
| Backbones | `resnet50` / `efficientnet_b0` / `efficientnet_b3` | `resnet50` only |
| Artifact root | `~/.claritymed/models/vision/lung_cancer_chest_ct/` | `~/.claritymed/models/vision/chest_xray_pneumonia/` |

The single-backbone search is intentional: Kermany exists primarily as
a **drift probe** (pediatric → adult shift via RSNA), not as a search
target. Composite weights also lean harder on recall (0.7 vs 0.6)
because pneumonia is a single critical class with no per-subtype
trade-off to balance.

## Adding a new dataset

Two files + a download wrapper, no pipeline code:

1. **Dataset module** under `ingest/vision/<disease>/` — Kaggle
   download wrapper, `dataset.py` exposing `discover()` /
   `stratified_split()` / `build_dataset()`, and a `dataset_spec.py`
   declaring a `DatasetSpec` (labels, label-metadata, modality,
   download slug, split-builder factory).
2. **Model module** under `ingest/vision/<disease>/models/<arch>.py` —
   instantiate `ModelSpec` with the chosen `Task` subclass
   (`ClassificationTask` or `ClassificationSegmentationTask`),
   composite weights, per-phase `FloorBundle`, hparam space, inference
   space. One file per (disease, architecture) variant.
3. **Console scripts** in `pyproject.toml` — only the download wrapper
   needs an entry point. The training pipeline reuses the single
   `claritymed-vision-forge` entry.

The forge CLI resolves `--model module.path:ATTR` via `importlib`, so
new specs become reachable the moment the file exists. The runtime
adapter (`servers/vision/adapters/forge_torch.py`) keys off
`manifest.task`; an unrecognised task value raises a loud error
pointing at where to add the next adapter branch.

## Lineage — `task_id`

One pipeline execution is identified by a single `task_id` (format:
`YYYYMMDDTHHMMSSZ-<8 hex>`). The orchestrator generates it once and
threads it into:

- **Optuna study name** — `claritymed-vision-<dataset>-<phase>-<task_id>`
  (one study per pipeline run per phase)
- **Optuna trials** — `user_attrs["task_id"]` (kept for cross-study
  grep recipes even though the study name already encodes it)
- **MLflow runs** — tag `claritymed.task_id` (search + train + tune runs)
- **Staging `provenance.json`** — top-level `task_id`, plus the
  winning search trial's `search_trial_number` + `search_trial_task_id`
  (always equals the top-level `task_id` since the study is scoped per
  task; the field is kept so the schema stays stable)
- **`LATEST.jsonl`** — top-level `task_id` + `model_id`

Pass `--task-id <id>` to extend an existing pipeline run — re-running
the same `task_id` resumes the same Optuna study (`load_if_exists=True`)
from the last completed trial. Omitting `--task-id` mints a fresh id
and starts a clean study, so prior runs can never contaminate
`study.best_trial`. To train from a prior search's winning HP, run the
train phase alone with that run's `--task-id`.

Pivot recipes:

```bash
# All MLflow runs for one task
mlflow runs list --experiment-name claritymed-vision-breast_cancer_ultrasound \
    --filter "tags.claritymed.task_id = '<id>'"

# All Optuna trials for one task (search + tune share the storage)
sqlite3 ~/.claritymed/tracking/optuna.db <<SQL
SELECT s.study_name, t.trial_id, t.value
FROM trials t
JOIN trial_user_attributes u ON t.trial_id = u.trial_id
JOIN studies s ON t.study_id = s.study_id
WHERE u.key = 'task_id' AND u.value_json = '"<id>"';
SQL

# All staging dirs / deploys for one task
grep -rl '"task_id": "<id>"' ~/.claritymed/models/vision/
```

## Observability — LATEST.jsonl

Every successful deploy appends one JSON line to
`~/.claritymed/models/vision/<disease_id>/LATEST.jsonl`.
This file is the **single observation entry point** — from one row a
viewer can reach every backing store. The `model_id` field is the
per-architecture key the regression gate filters on, so multiple model
variants on the same dataset can coexist in one log:

```json
{
  "version_tag": "20260615T120000Z",
  "deployed_at": "2026-06-15T12:00:00Z",
  "task_id": "20260615T115000Z-abcdef12",
  "model_id": "breast_busi_unet_v1",
  "weights_subpath": "vision/breast_cancer_ultrasound/breast_busi_unet_v1__20260615T120000Z",
  "manifest_path": "/.../breast_busi_unet_v1__20260615T120000Z/manifest.json",
  "manifest_sha256": "…",
  "mlflow": {
    "tracking_uri": "sqlite:///.../tracking/mlflow.db",
    "experiment_name": "claritymed-vision-breast_cancer_ultrasound",
    "train_run_id": "…",
    "tune_run_id": "…"
  },
  "optuna": {
    "storage_uri": "sqlite:///.../tracking/optuna.db",
    "search_study_name":    "claritymed-vision-breast_cancer_ultrasound-hparam-<task_id>",
    "search_trial_number":  18,
    "search_trial_task_id": "20260615T115000Z-abcdef12",
    "tune_study_name":      "claritymed-vision-breast_cancer_ultrasound-tune-<task_id>"
  },
  "best_hp": { "backbone": "…", "lr": 1.2e-3, "seg_loss_weight": 0.9 },
  "best_inference_params": { "temperature": 1.2, "critical_threshold": 0.45, … },
  "metrics": {
    "best_val_score": 0.65,
    "test_score": 0.63,
    "tuned_test_breakdown": { "composite": 0.66, "malignant_recall": 0.88, "dice": 0.72, "accuracy": 0.85 },
    "tuned_test_composite": 0.66,
    "best_epoch": 24,
    "early_stopped": true,
    "epochs_trained": 38
  },
  "previous_tuned_test_composite": 0.62,
  "delta": 0.04,
  "floors_passed": { "malignant_recall": true, "accuracy": true, "dice": true }
}
```

Quick commands:

```bash
# Show the active deploy for any model_id on this dataset.
tail -n1 ~/.claritymed/models/vision/breast_cancer_ultrasound/LATEST.jsonl | jq

# Filter to one model variant when multiple coexist.
jq -c 'select(.model_id == "breast_busi_unet_v1")' \
    ~/.claritymed/models/vision/breast_cancer_ultrasound/LATEST.jsonl | tail -n1 | jq

# Diff metrics deploy-over-deploy for one model_id.
jq -c 'select(.model_id == "breast_busi_unet_v1")
       | {version_tag, score: .metrics.tuned_test_composite}' \
    ~/.claritymed/models/vision/breast_cancer_ultrasound/LATEST.jsonl | tail -n2

# Open the MLflow run for the active deploy.
mlflow ui --backend-store-uri sqlite:///$HOME/.claritymed/tracking/mlflow.db --port 5000
# → filter experiment name from LATEST.jsonl[-1].mlflow.experiment_name, run id from train_run_id / tune_run_id

# Open the Optuna trials for the active deploy.
optuna-dashboard sqlite:///$HOME/.claritymed/tracking/optuna.db
# → pick study from LATEST.jsonl[-1].optuna.{search_study_name, tune_study_name}
```

## Floors + regression gate

```
Pass gate iff:
  every metric in ModelSpec.task.floors.deploy clears its floor
    (BUSI:     malignant_recall >= 0.85, accuracy >= 0.85, dice >= 0.70)
    (chest CT: cancer_recall    >= 0.85, accuracy >= 0.80)
  tuned_test_composite > previous tuned_test_composite for the same model_id
```

If any check fails:

- **Iterate Optuna** (more eng-days; budget accordingly).
- **Hold v1 launch** — a vision tool that misses cancer is worse than
  no vision tool at all.
- Do NOT silently expand v1 to a different disease ("fall back to
  skin"). The skin pipeline is a separate planning conversation.

## Search-phase gate failures

The pipeline aborts between phases when the winner's breakdown misses
the next phase's floors (`ModelSpec.task.floors.{search,train,deploy}`).
Example:

```
phase 'search' did not clear its floors: malignant_recall=0.645
(floor 0.650, deficit 0.005), dice=0.282 (floor 0.400, deficit 0.118).
```

Read the deficits, not the failure count:

- **Small deficit on one floor** (≤ ~0.02). Likely one good trial away.
  Re-run with more `--trials`; same `--search-epochs`.
- **Large deficit on a segmentation metric** (≥ ~0.10). The mask head
  is underfit, not unlucky. Bump `--search-epochs` first; extra trials
  at too-few epochs re-sample the same underfit ceiling.
- **Large deficit on the critical-recall metric** that doesn't move
  when epochs go up. The search space probably can't reach the floor
  under any HP combo — widen the space in the spec rather than
  re-rolling trials. The shipped backbones are already pretrained where
  pretraining helps; the next levers are loss composition (e.g. an
  independent `dice_loss_weight`) and augmentation strength, not more
  encoders.

`--force` is for diagnostic runs only. It demotes the inter-phase gates
to warnings but the deploy gate stays strict, so a forced run that
fails search will still fail deploy — wasted compute. Don't lower the
floors in the spec to make a run pass; that defeats the purpose of
having them.

## Smoke test

A synthetic 1-epoch run that exercises every phase
(search→train→tune→deploy) **without touching torch or the real
dataset**:

```bash
uv run claritymed-vision-forge pipeline \
    --model claritymed.ingest.vision.busi.models.unet_resnet50:UNET_RESNET50 \
    --smoke
```

What's faked: HP set (`_smoke_hp` picks the first backbone + LR=0),
weights file (6-byte `b"smoke"` sentinel), per-phase metric breakdowns
(`Task.smoke_breakdown()` returns feasibly-above-floor numbers so the
gates clear). Pipeline finishes in sub-second. CI runs this nightly.

**Smoke stops at the staging dir.** Floor gates run as a wiring check,
but the deploy phase intentionally does not:

- copy the staging dir to `<model_id>__<tag>/` (the stable, server-loaded path)
- update the `<model_id>` stable symlink
- append a row to `LATEST.jsonl`
- patch `configs/vision.yaml`
- run the regression gate

The only on-disk artifact a smoke run leaves is the staging directory
under `~/.claritymed/models/vision/<disease>/run/<model_id>_<ts>/`,
which is ephemeral by design. This is why you can run `--smoke` twice
in a row without the second run hitting a regression-gate self-block —
nothing is written for the gate to compare against.

Useful for: catching a broken CLI / dataset spec / Task subclass
without paying for download + training time, **and** without leaving
fake state the vision-server would later try to load. Not useful for:
catching real training bugs — the smoke path skips every torch call
site. Use `--quick` below for a real-data check.

Works for any spec — swap the `--model` value to smoke-test chest CT
or a new dataset.

## Quick mode

The smallest honest end-to-end on a real dataset:

```bash
uv run claritymed-vision-forge pipeline \
    --model claritymed.ingest.vision.chest_ct.models.resnet50_v1:RESNET50_V1 \
    --quick
```

`--quick` overrides the pipeline budgets to: `--trials 3
--search-epochs 3 --max-epochs 5 --patience 3 --tune-trials 5`, and
auto-sets `--force` + `--deploy-force` because microscopic budgets
won't clear the spec's floors. Minutes on a modern Apple Silicon /
CUDA box.

Requires:

- Real dataset already downloaded (step 1).
- torch + MPS or CUDA available (CPU-only is unworkable).

Mutually exclusive with `--smoke`. Use `--quick` as the first command
after a fresh `claritymed-vision-download-<dataset>` to confirm the
real pipeline works end-to-end. Once that succeeds, drop `--quick` and
run with full budgets for a production model.

The deploy artifact from a `--quick` run is **not** suitable for
serving — the floors are off and the model is undertrained. Delete the
staging + stable dirs before the first real run, or expect the next
real deploy's regression gate to compare against the `--quick`
composite.

## Per-dataset commands

Quick reference — download + forge invocation for every dataset that
currently ships an ingest module. All commands assume
`uv sync --extra vision-server` and Kaggle credentials are configured.
The forge `pipeline` subcommand runs steps 2 → 5 in one go; substitute
`--smoke` for a wiring check or `--quick` for the smallest honest
end-to-end on real data.

Disease identity is `(condition × modality)` — see the comment at the
top of `configs/vision.yaml`. Multiple ingest modules can share one
disease_id when they target the same (condition × modality) coordinate
with alternate training data; those models are enumerated in the
disease's `flow:` array as fallbacks (current example: BUSI primary +
breast_us_kaggle fallback, both under `breast_cancer_ultrasound`).

### `breast_cancer_ultrasound` — BUSI (primary, cls + seg)

```bash
uv run claritymed-vision-download-busi
uv run claritymed-vision-forge pipeline \
    --model claritymed.ingest.vision.busi.models.unet_resnet50:UNET_RESNET50
```

Promoted and enabled in `configs/vision.yaml`. The only spec that
ships a segmentation head — BUSI is the upstream that provides masks.

### `breast_cancer_ultrasound` — Kaggle alternate (fallback, cls only)

```bash
uv run python -m claritymed.ingest.vision.breast_us_kaggle.download
uv run claritymed-vision-forge pipeline \
    --model claritymed.ingest.vision.breast_us_kaggle.models.resnet50_v1:RESNET50_V1
```

Scaffolded. Same `disease_id` as BUSI — alternate training data
(~9000 augmented 2-class images, no `normal` class). Sits in
`configs/vision.yaml::models` as `breast_us_kaggle_resnet50_v1` but is
NOT yet in `breast_cancer_ultrasound.flow`; promote to the flow only
after deploy passes (the vision server loads flow models at startup,
so placeholder-sha entries must stay out of any enabled disease's
flow).

### `lung_cancer_chest_ct` — Chest CT (cls only)

```bash
uv run claritymed-vision-download-chest-ct
uv run claritymed-vision-forge pipeline \
    --model claritymed.ingest.vision.chest_ct.models.resnet50_v1:RESNET50_V1
```

Promoted and enabled in `configs/vision.yaml`. 4-class (3 malignant
subtypes + `normal`). See "Other diseases — chest CT" above for the
per-spec deltas vs BUSI.

### `chest_xray_pneumonia` — RSNA Pneumonia (classification, cls only)

```bash
uv run python -m claritymed.ingest.vision.rsna_pneumonia.download
uv run claritymed-vision-forge pipeline \
    --model claritymed.ingest.vision.rsna_pneumonia.models.resnet50_v1:RESNET50_V1
```

Adult-population sibling of Kermany. Same 2-class task, same backbone,
different training source (~30k multi-center adult chest X-rays vs
Kermany's ~5,856 pediatric Guangzhou hospital). Bbox rows are
collapsed to image-level labels (any bbox → `pneumonia`).

Shares `disease_id="chest_xray_pneumonia"` with Kermany, which means
the forge artifact root, `LATEST.jsonl`, and MLflow experiment are
**shared**; `model_id` is the discriminator
(`rsna_pneumonia_resnet50_v1` vs `chest_xray_pneumonia_resnet50_v1`).
The regression gate filters by `model_id` before comparing the
previous composite, so the two never gate against each other —
apples-to-apples across deploys.

Floors are inherited from Kermany as a starting point and are **not**
empirically validated against RSNA (noisier labels, multi-center
variance, ~20% positive vs Kermany's ~73%). Expect first runs to land
near the deploy floor; tune the floor (or add class-weighted loss to
`hparam_space`) once there's a baseline number. Do **not** lower the
floor to make a run pass without a recorded rationale.

Drift complement: training RSNA enables the symmetric drift pair
(Kermany→RSNA *and* RSNA→Kermany) for `tests/benchmarks/cross_dataset_drift/`.

### `chest_xray_pneumonia` — RSNA Pneumonia (detection, YOLO)

```bash
uv run python -m claritymed.ingest.vision.rsna_pneumonia.download
uv sync --extra yolo-forge
uv run claritymed-vision-yolo-forge pipeline \
    --model claritymed.ingest.vision.rsna_pneumonia_yolo.models.yolov8n_v1:RSNA_YOLOV8N_V1
```

Driven by the parallel `yolo_forge` framework (see "Detection pipeline
(yolo_forge)" above). Same RSNA archive the classification drift
adapter uses; the detection adapter re-parses the labels CSV keeping
the bbox rows and emits YOLO-format labels alongside the shared PNG
cache. Artifact root is `rsna_pneumonia_detection` (suffixed to keep
it distinct from a future trained-on-RSNA classification artifact),
but the `disease_id` matches the classification side so the MLflow
experiment is shared.

### `skin_cancer_dermoscopy` — ISIC 9-class (cls only)

```bash
uv run python -m claritymed.ingest.vision.skin_lesion.download
uv run claritymed-vision-forge pipeline \
    --model claritymed.ingest.vision.skin_lesion.models.resnet50_v1:RESNET50_V1
```

Scaffolded; disease ships `enabled: false` until weights are
promoted. 9-class — the largest label set in the project; the floor
on accuracy is correspondingly looser (0.65 vs BUSI's 0.85) because
random baseline drops to 0.11.

### `lung_cancer_histopathology` — LC25000 lung subset (cls only)

```bash
uv run python -m claritymed.ingest.vision.lung_colon_histopath.download
uv run claritymed-vision-forge pipeline \
    --model claritymed.ingest.vision.lung_histopath.models.resnet50_v1:RESNET50_V1
```

Scaffolded; disease disabled. 3-class
(`adenocarcinoma` / `normal` / `squamous_cell_carcinoma`). Shares the
LC25000 Kaggle archive on disk with the colon module below — running
either download command covers both.

### `colon_cancer_histopathology` — LC25000 colon subset (cls only)

```bash
uv run python -m claritymed.ingest.vision.lung_colon_histopath.download
uv run claritymed-vision-forge pipeline \
    --model claritymed.ingest.vision.colon_histopath.models.resnet50_v1:RESNET50_V1
```

Scaffolded; disease disabled. 2-class (`adenocarcinoma` / `normal`).
Same LC25000 archive as the lung sibling above; the download is
idempotent so running it again is a no-op if the lung side already
fetched it.

## Detection pipeline (yolo_forge)

Sibling framework to `forge` for bounding-box detection. Lives at
`src/claritymed/ingest/vision/yolo_forge/`; per-dataset adapters sit
as siblings of the classification ingest modules (current: RSNA
Pneumonia detection at `vision/rsna_pneumonia_yolo/`, parallel to the
classification adapter at `vision/rsna_pneumonia/`).

Shared with `forge`:

* Same on-disk artifact layout
  (`~/.claritymed/models/vision/<dataset_id>/run/<model_id>_<ts>/`),
  same `LATEST.jsonl` per dataset, same `task_id` lineage convention.
* Same MLflow experiment — `claritymed-vision-<disease_id>`. yolo_forge
  runs tag themselves with `pipeline=yolo_forge` + `task=detection`
  so a detection + classification model on the same disease land in
  one UI view, separable on demand.
* Same `--model module.path:ATTR` resolution and `--phases` filter
  semantics on the pipeline subcommand.

Different from `forge`:

* **5 phases** instead of 4: `prepare → search → train → tune → deploy`
  (forge folds prepare into `DatasetSpec.build_splits`; yolo_forge
  keeps it as an observable phase because the YOLO-format on-disk
  materialisation is user-inspectable).
* **Detection-specific spec types** (`yolo_forge/spec.py`):
  `DetectionDatasetSpec`, `YoloModelSpec`, `YoloTrainHparams`,
  `DetectionSplits`. SearchSpace DSL is re-exported from `forge.spec`
  — same `Categorical` / `LogUniform` / `Uniform` / `BoolChoice`.
* **HPO objective is Ultralytics' fitness composite**
  (`0.1·mAP50 + 0.9·mAP50-95`), used identically by `search` (train-time
  hparams) and `tune` (inference-time conf / iou). The clinical
  fail-safe metric — `image_recall` — stays enforced by the deploy
  gate's `eval_thresholds`, not by the HPO objective; keeps the HPO
  loop aligned with how the rest of the ecosystem ranks detection
  models while the deploy phase catches missed-positive regressions.
* **Standalone `eval` subcommand** for one-off re-evaluation at
  arbitrary `(conf, iou)` thresholds — not part of the pipeline
  itself.

### Install

```bash
uv sync --extra yolo-forge   # ultralytics + pydicom + torch + pillow + mlflow
```

mlflow is required at pipeline execution (search/train/tune/deploy
all open MLflow runs); missing mlflow fails loud with a remediation
hint rather than silently dropping metrics.

### Invocation

```bash
# Full pipeline (search HPO → train → tune inference params → deploy):
uv run claritymed-vision-yolo-forge pipeline \
    --model claritymed.ingest.vision.rsna_pneumonia_yolo.models.yolov8n_v1:RSNA_YOLOV8N_V1

# Fresh-box dry run (1 train epoch, skip HPO):
uv run claritymed-vision-yolo-forge pipeline \
    --model claritymed.ingest.vision.rsna_pneumonia_yolo.models.yolov8n_v1:RSNA_YOLOV8N_V1 \
    --quick --skip-search

# Single-phase entry points:
uv run claritymed-vision-yolo-forge prepare --model <spec>
uv run claritymed-vision-yolo-forge search  --model <spec> --trials 10 --epochs-per-trial 5
uv run claritymed-vision-yolo-forge train   --model <spec>
uv run claritymed-vision-yolo-forge tune    --model <spec> --trials 20
uv run claritymed-vision-yolo-forge deploy  --model <spec>

# One-off re-eval at chosen (conf, iou):
uv run claritymed-vision-yolo-forge eval --model <spec> --split val --conf 0.3 --iou 0.5
```

### Per-phase outputs (RSNA YOLOv8n example)

Under
`~/.claritymed/models/vision/rsna_pneumonia_detection/run/rsna_pneumonia_yolov8n_v1_<ts>/`:

* `search/trial_NNN/` — Ultralytics' per-trial training output dir;
  Optuna fitness logged per trial as MLflow steps under `search/...`
* `best_hparams.json` — winning hparams + merged train-config blob
* `train/` — full-training Ultralytics dir (`weights/best.pt`,
  `weights/last.pt`, `results.csv` — replayed into MLflow once at
  phase end as `train/<col>` per epoch)
* `tune/trial_NNN/` — per-trial `model.val()` output dirs
* `eval_metrics.json` — final test metrics at tuned `(conf, iou)`,
  plus `val_metrics_at_best` + `tuned_inference_params`; read by deploy
* `metadata.json` — pipeline + task_id + dataset + hparams summary

On deploy pass:
`~/.claritymed/models/vision/rsna_pneumonia_detection/rsna_pneumonia_yolov8n_v1_v1.pt`
(the stable weights path) + one row appended to
`~/.claritymed/models/vision/rsna_pneumonia_detection/LATEST.jsonl`
with `pipeline: "yolo_forge"`, `tuned_inference_params`, and the test
metrics blob.

### Dataset adapter layout — `rsna_pneumonia_yolo/`

The classification adapter (`rsna_pneumonia/`) stays untouched. The
detection adapter adds, as a sibling:

* `dataset.py::prepare_rsna_pneumonia_yolo()` — re-parses
  `stage_2_train_labels.csv` *keeping* the bbox rows, reuses the
  shared DICOM→PNG cache via the classification adapter's `discover()`,
  writes one YOLO `.txt` per image (empty for normals; one line per
  bbox for positives, normalised against the actual PNG dimensions),
  symlinks images into `<raw_root>/yolo/images/{train,val,test}/`,
  emits `data.yaml`. Idempotent — second run skips work that's already
  on disk.
* `dataset_spec.py::RSNA_PNEUMONIA_YOLO_DATASET` —
  `DetectionDatasetSpec(dataset_id="rsna_pneumonia_detection",
  disease_id="chest_xray_pneumonia", class_names=("pneumonia",), ...)`.
  Note the suffixed `dataset_id` (keeps the artifact root distinct
  from the classification adapter's), but `disease_id` matches the
  classification side exactly — that's what shares the MLflow
  experiment between the two pipelines.
* `models/yolov8n_v1.py::RSNA_YOLOV8N_V1` — `YoloModelSpec` with
  hparam_space (lr / momentum / weight_decay / mosaic), inference_space
  (conf / iou), and `eval_thresholds={"image_recall": 0.80, "mAP50":
  0.30}`. Bump the filename to `yolov8s_v1.py` etc. when adding
  larger variants — never mutate an existing version in place.

### Adding a new detection dataset

Mirrors the classification "Adding a new dataset" recipe with two
adapter files + a model spec, but uses the detection types:

1. **Dataset module** at `ingest/vision/<dataset>_yolo/` (sibling to any
   existing classification adapter for the same archive). Required:
   `dataset.py::prepare_<dataset>_yolo()` (returns `DetectionSplits`),
   `dataset_spec.py::<DATASET>_YOLO_DATASET` (a
   `DetectionDatasetSpec` instance).
2. **Model module** at `<dataset>_yolo/models/<arch>_v<N>.py` —
   instantiate `YoloModelSpec` with `base_weights` (an Ultralytics
   weight id, e.g. `"yolov8n.pt"`), `train_hparams` overrides,
   `hparam_space` / `inference_space` (re-using the SearchSpace types
   from `yolo_forge.spec`), and `eval_thresholds`.
3. **No new console script** — `claritymed-vision-yolo-forge` resolves
   the dotted path via importlib, so a new spec becomes reachable the
   moment the file exists.

If the raw archive ships DICOMs you can reuse the existing
classification adapter's `discover()` for DICOM→PNG conversion (as
RSNA does) — that keeps the on-disk PNG cache shared.

## Cross-dataset drift bench

A separate, **local-only** bench harness lives at
`tests/benchmarks/cross_dataset_drift/`. It loads trained model
artifacts (`manifest.json` + `weights.pt`) and runs each model over a
*different* dataset's test split — the gap between in-distribution
self-eval and cross-eval is the drift signal.

```bash
make bench-drift PAIR=breast_us    # ships today
make bench-drift PAIR=chest_xray   # requires Kermany training + RSNA download
```

The bench is **read-only** against forge artifacts; it never touches
`LATEST.jsonl`, `configs/vision.yaml`, MLflow, or Optuna stores. Bench
failures cannot contaminate the regression gate.

Output lands under `docs/benchmarks/cross_dataset_drift/<date>-<pair>.{json,md}`.
The whole `docs/` tree is gitignored as a tripwire so results stay
local to each developer's machine; the directory's `README.md`
documents the metric collapse, threshold semantics, and how to add
new pair entries to the registry.

Eval-only datasets that have no trained model in the project ship as
a `DatasetSpec` only — no `models/` subdir. (RSNA Pneumonia was the
original such case but now ships its own `models/resnet50_v1.py` and
trains as a sibling to Kermany; the eval-only pattern below stays the
recipe for new drift-bench targets.) They become reachable as soon as
their `dataset_spec.py` exports a `DatasetSpec` instance with the
matching `disease_id` and `accepted_modality`. To add an eval-only
target:

1. Create `ingest/vision/<source_dataset>/` with `download.py` (Kaggle
   datasets CLI or competitions CLI) + `dataset.py` (discover + splits +
   torch `Dataset`) + `dataset_spec.py`.
2. Match the `disease_id` of the trained model you want to evaluate
   against — that's how the bench's binary clinical task collapse
   stays apples-to-apples across the cross-eval pair.
3. Add a `BenchEntry` for each `(model_artifact, eval_dataset)` cell
   to `tests/benchmarks/cross_dataset_drift/registry.py` (include
   both self-eval baseline and cross-eval drift rows).

See `tests/benchmarks/cross_dataset_drift/README.md` (auto-generated
on first run) for output-format details and how to interpret a result
table.
