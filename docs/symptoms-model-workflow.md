# Symptoms Model: Training & Deployment Workflow

End-to-end pipeline from hyperparameter search to serving.

```
Optuna search  →  Train  →  Joint tune (maxstep × temp)  →  Deploy  →  Start
                     ↓              ↓
                  MLflow         MLflow
```

All training and tuning steps log to MLflow. Start the UI before running
any step to observe metrics live:

```bash
uv run mlflow ui --backend-store-uri sqlite:///$HOME/.claritymed/tracking/mlflow.db --port 5000
# → http://localhost:5000  (experiment: claritymed-symptoms-ddxplus)
```

The tracking DB is shared across every feature + dataset; experiments
are disambiguated by name (`claritymed-<feature>-<dataset_id>`). Old
per-dataset DBs under `models/<feature>/<dataset>/run/mlflow.db` have
been moved into the shared DB.

---

## Step 0: Prepare data

Place the DDXPlus release files under `CLARITYMED_HOME/data/symptoms/ddxplus/`:

```
release_evidences.json
release_conditions.json
release_train_patients.zip
release_validate_patients.zip
release_test_patients.zip
```

`CLARITYMED_HOME` defaults to `~/.claritymed`.

---

## Step 1: Hyperparameter search (one-time)

Searches `hidden` layer width and learning rate. Uses early stopping on the
validate split; never touches the test split.

```bash
uv run claritymed-symptoms-hparam-ddxplus \
    --data-dir ~/.claritymed/data/symptoms/ddxplus \
    --n-trials 20
```

Trials persist automatically to the shared
`~/.claritymed/tracking/optuna.db` (study `typed_basd_hparam`); the
search can be interrupted and resumed with the same command.

Monitor live in another terminal:

```bash
uv run optuna-dashboard sqlite:///$HOME/.claritymed/tracking/optuna.db
# → http://localhost:8080  (pick study: typed_basd_hparam)
```

The script prints a summary table at the end and copies the best
checkpoint to `~/.claritymed/models/symptoms/ddxplus/run/hparam_best.pt`.

**Key flags**

| Flag | Default | Notes |
|------|---------|-------|
| `--n-trials` | 20 | Total Optuna trials |
| `--max-epochs` | 40 | Per-trial ceiling; early stopping fires sooner |
| `--patience` | 4 | Epochs without val improvement before early stop |
| `--eval-n-val` | 2000 | Validate patients per eval (smaller = faster, noisier) |
| `--storage` | none | SQLite URL for persistence + dashboard |

---

## Step 2: Train production checkpoint

Use the best `hidden` and `lr` from Step 1. This run uses the full
training set and evaluates on the test split.

```bash
uv run claritymed-symptoms-train-ddxplus \
    --data-dir ~/.claritymed/data/symptoms/ddxplus \
    --hidden <best_hidden> \
    --lr <best_lr> \
    --epochs 30
```

On success the script writes:

```
~/.claritymed/models/symptoms/ddxplus/run/typed_basd_v2_<timestamp>/
    weights.pt
    manifest.json          # sha256 + eval numbers + train_params
```

`run/` is a staging area. Promote to a stable path after evaluation (Step 4a);
the server reads from the stable path, not from `run/`.

The run is logged to MLflow under experiment `claritymed-symptoms-ddxplus`
(`run_type=train`). Per-epoch val metrics (DDF1, DSR, losses) and final
test metrics are visible in the UI as the run progresses.

**`--patho-temp`** (optional): leave at default (1.0) for the first
training run and pick the operating point in Step 3 instead.

---

## Step 3: Joint tune (maxstep × patho_temp)

`patho_temp` directly affects which diseases enter the differential and
therefore affects DSR — the two dimensions cannot be ablated independently.
The tune command sweeps all `(maxstep, temp)` combinations in one pass,
logging each as a nested MLflow child run under a single parent.

```bash
uv run claritymed-symptoms-tune-ddxplus \
    --data-dir ~/.claritymed/data/symptoms/ddxplus \
    --weights ~/.claritymed/models/symptoms/ddxplus/run/typed_basd_v2_candidate/weights.pt \
    --maxsteps 6,8,10,12,14,16,18,24,30 \
    --temps 1.0,0.7,0.5 \
    --games 1000
```

The script prints a `(T × maxstep)` results matrix and recommends the pair
using the **IL saturation elbow**: the smallest maxstep where the next step's
marginal IL gain/step drops below 0.3, indicating the stop gate is already
terminating most games naturally. Among qualifying (DSR ≥ 92) points at or
beyond the elbow, it picks the smallest maxstep and breaks ties by DDF1.
Rows passing the DSR floor are marked `*`; the elbow row is marked `←elbow`.

In the MLflow UI (experiment `claritymed-symptoms-ddxplus`, filter
`run_type=tune_child`) you can plot DDF1 and IL across all child runs to
visualise the Pareto frontier before committing.

**Selection criteria (in order):**

| Priority | Criterion |
|---|---|
| Hard constraint | DSR ≥ 92.0 |
| Gate | maxstep ≥ IL saturation elbow (marginal IL/step < 0.3) |
| Primary | `score = DDF1 − IL_PENALTY × IL` maximum (default `IL_PENALTY = 1.0`) |
| Tiebreaker | smaller maxstep |

`IL_PENALTY` controls the tradeoff: 1.0 means one extra question must buy at
least 1 DDF1 point to be worthwhile. The table printed by the tune command
shows the `score` column so you can inspect the frontier before accepting the
recommendation. Adjust `IL_PENALTY` in `tune.py` if your product weighting differs.

> Why the elbow gate matters: below the elbow the stop gate never fires — the
> agent is truncated every game, IL ≈ maxstep, and DDF1 is artificially
> depressed. The gate ensures the score comparison is only made among points
> where the agent is genuinely in control of conversation length.

Once you have the recommended `(T, maxstep)`, re-train with `--patho-temp
<T>` so the temperature is baked into the checkpoint:

```bash
uv run claritymed-symptoms-train-ddxplus \
    --data-dir ~/.claritymed/data/symptoms/ddxplus \
    --hidden <best_hidden> --lr <best_lr> \
    --patho-temp <chosen_T>
```

**Quick single-T maxstep scan** (no MLflow, stdout only):

```bash
uv run claritymed-symptoms-ablate-ddxplus \
    --data-dir ~/.claritymed/data/symptoms/ddxplus \
    --weights ~/.claritymed/models/symptoms/ddxplus/run/typed_basd_v2_candidate/weights.pt \
    --maxsteps 6,8,10,12,18,24,30 --games 1000 --patho-temp 0.7
```

---

## Step 4: Deploy

### 4a. Promote candidate to production

Train writes to `run/typed_basd_v2_<timestamp>/`. After Step 3 evaluation
passes, copy out of `run/` to the stable path the server reads:

```bash
RUN=~/.claritymed/models/symptoms/ddxplus/run
STABLE=~/.claritymed/models/symptoms/ddxplus
cp -r $RUN/typed_basd_v2_<timestamp> $STABLE/typed_basd_v2
```

### 4b. Update `configs/symptoms.yaml`

Get the manifest SHA-256:

```bash
python -c "import hashlib,sys; h=hashlib.sha256(); \
  [h.update(c) for c in iter(lambda: open(sys.argv[1],'rb').read(1<<20),b'')]; \
  print(h.hexdigest())" \
  ~/.claritymed/models/symptoms/ddxplus/typed_basd_v2/manifest.json
```

Then in `configs/symptoms.yaml`:

```yaml
datasets:
  - id: ddxplus
    enabled: true                  # was false
    maxstep: 18                    # from Step 3 tune output

models:
  - id: typed_basd_v2
    manifest_sha256: "<paste SHA-256 here>"
```

### 4b. Start the symptoms server

```bash
uv sync --extra symptoms-server
uv run claritymed-symptoms-server
```

The server verifies the two-level integrity chain (config SHA → manifest
SHA → weights SHA) at startup and refuses to start on any mismatch.

---

## Step 5: Start

```bash
uv run claritymed tui
```

The symptoms plugin is enabled automatically when `configs/symptoms.yaml`
has `enabled: true` and the server is reachable.

---

## Re-train checklist

When re-running after a new DDXPlus release or architecture change:

- [ ] Step 1 if search space changed (new hidden choices, lr range)
- [ ] Step 2 with updated `--hidden` / `--lr` (leave `--patho-temp` at 1.0)
- [ ] Step 3 joint tune to re-pick `(maxstep, patho_temp)` — always re-run after new weights
- [ ] Step 2 again with `--patho-temp <chosen_T>` to bake temperature into checkpoint
- [ ] Step 4a to promote and update `manifest_sha256` in `configs/symptoms.yaml`
- [ ] Commit `configs/symptoms.yaml` so the SHA is version-controlled

---

## Subset training (single-domain models alongside the full 49-class)

Beside the full 49-class model you can ship domain-scoped models that
handle only a hand-picked slice of diseases — e.g. a "Pneumonia +
Influenza" 2-class model for respiratory-infection triage. Both models
live side-by-side under `configs/symptoms.yaml` as separate `datasets:`
entries; the LLM routes with `dataset_hint`.

### When (and when NOT) to subset

Use case | Fit
---|---
Product needs faster/cheaper inference on a narrow domain | ✓ subset is fine
You want a "focus" model with less noise from unrelated diseases | ✓ subset is fine
You want the model to *rule out* diseases outside the whitelist | ✗ subset **cannot** do this
Subset has ≥ 8-13 diseases with meaningful clinical overlap | ✓ `--target differential` works
Subset has ≤ 3-4 diseases | ⚠ differential target degenerates to near-one-hot; use `--target pathology`

**Subset-conditional semantics** — a subset model outputs `P(disease |
disease ∈ trained_subset)`. It does NOT distinguish "the patient has
Pneumonia" from "the patient has Bronchitis but I don't have that class,
so I'll say Pneumonia with 92% confidence". The LLM must gate scope
BEFORE trusting the ranking; this caveat lives in the dataset's
`domain_description` and gets injected into the tool description at
runtime.

### Step S1: Pick the disease list (optional confusion-based selection)

For a "manually specified" subset, skip to S2 with your names in hand.

For a confusion-driven pick around a target disease, run the DDXPlus
subset selector against a trained full-corpus checkpoint:

```bash
uv run python scripts/select_ddxplus_subset.py \
    --data-dir ~/.claritymed/data/symptoms/ddxplus \
    --weights ~/.claritymed/models/symptoms/ddxplus/typed_basd_v2/weights.pt \
    --target Pneumonia \
    --out configs/symptoms/pneumonia_subset.yaml \
    --top-confused-n 8
```

Emits a YAML with target + top-confused + cannot-miss lists. Flatten
those name lists into the comma-separated `--diseases` string for
Step S2. See `docs/symptoms-pneumonia-subset-selection.md` for the
selection methodology.

### Step S2: Train the subset model

```bash
uv run claritymed-symptoms-train-ddxplus \
    --data-dir ~/.claritymed/data/symptoms/ddxplus \
    --diseases "Pneumonia,Influenza" \
    --episodes 1_500_000 \
    --hidden 2048 --lr 1e-4 \
    --patho-temp 1.0
```

Auto-defaults when `--diseases` is set:
- `--target` → `pathology` (differential degenerates for small subsets)
- `--out-subpath` → `ddxplus/run/typed_basd_subset_<slug>_<ts>/`

Names must match `release_conditions.json` exactly (case + punctuation)
— typos fail loud with the unrecognized names listed.

**Watch the drop ratio.** Under a 2-disease whitelist, DDXPlus drops
~95% of raw rows (only 4.9% of patients have PATHOLOGY in
{Pneumonia, Influenza}). `--episodes` reads N raw rows BEFORE the
filter, so bump it accordingly:

Whitelist size | kept ratio (approx) | `--episodes` for ~75k kept
---|---|---
2 diseases | 5% | 1.5M
8 diseases | 20% | 400k
15 diseases | 40% | 200k

The load log line `[load_patients train] filtered X of Y rows (Z%)`
tells you the exact ratio for your slice.

**DSR gate note.** If none of your whitelisted diseases have
`severity < 3` (per `release_conditions.json`), DSR eval is NaN and
train.py falls back to DDF1 as the improvement signal — expected, not
a failure.

The manifest records `diseases_trained` (in class-idx order),
`training_target`, and `train_pats_kept`. The class-idx contract is
what the adapter cross-checks against `disease_whitelist` at server
load time.

### Step S3: Joint tune (optional)

Same `claritymed-symptoms-tune-ddxplus` command as Step 3 works on
subset checkpoints. The DSR-based selection may fall back to DDF1 for
subsets without severe classes — inspect the printed table before
accepting the recommendation.

### Step S4: Promote and wire into config

Promote same as Step 4a:

```bash
RUN=~/.claritymed/models/symptoms/ddxplus/run
STABLE=~/.claritymed/models/symptoms/ddxplus
cp -r $RUN/typed_basd_subset_influ_pneumo_<ts> $STABLE/typed_basd_pneumonia_flu_v1
```

Compute the manifest SHA (same helper as Step 4b) and paste into
`configs/symptoms.yaml`. Uncomment the example blocks (`datasets:
- id: ddxplus_pneumonia_flu` and its matching `models:` entry), flip
`enabled: true`, and set `disease_whitelist:` to the same names you
trained with:

```yaml
datasets:
  - id: ddxplus_pneumonia_flu
    enabled: true
    model_ids: [typed_basd_pneumonia_flu_v1]
    disease_whitelist:
      - Pneumonia
      - Influenza
    domain_description:
      en: |
        DDXPlus subset — Pneumonia + Influenza. Returned probabilities
        are subset-conditional; cannot rule out Bronchitis, URTI, TB, or
        overlapping respiratory presentations.
      zh: |
        DDXPlus 子集 — 肺炎 + 流感。返回的概率是子集条件下的排序，不能
        排除支气管炎、上呼吸道感染、肺结核等重叠表现。

models:
  - id: typed_basd_pneumonia_flu_v1
    algorithm_module: typed_basd
    weights_subpath: ddxplus/typed_basd_pneumonia_flu_v1
    manifest_sha256: "<paste SHA-256 here>"
    maxstep: 18
    patho_temp: 1.0
    stop_thres: 0.1
```

The adapter enforces `disease_whitelist == manifest.diseases_trained`
at startup — a mismatch fails loud rather than silently misaligning
class indices in production.

### What auto-updates, what doesn't

- **Tool description** — `symptoms_plugin._build_tool_description()`
  iterates every enabled dataset and joins their `domain_description["en"]`
  into the `{covered_conditions}` placeholder. Adding the second dataset
  entry is enough; no plugin code changes.
- **LLM routing** — the tool exposes a `dataset_hint` param. The LLM
  reads both `domain_description` blocks and picks. If your full-corpus
  `domain_description` and subset `domain_description` don't clearly
  delineate scope, the LLM will route randomly — the caveat text
  matters.
- **Multi-card renderer / differential formatter / severity gate** —
  all key off the per-dataset `condition_names` (built from the
  subset-shrunk pidx). No code changes.

### Subset re-train checklist

- [ ] Verify disease names match `release_conditions.json` exactly
- [ ] `--episodes` sized for the drop ratio (see table above)
- [ ] Optional `hparam-ddxplus` sweep if the subset is very different in scale
- [ ] `train-ddxplus --diseases "..."` — auto-defaults `--target pathology`
- [ ] `tune-ddxplus` on the subset checkpoint
- [ ] Promote to stable path, compute manifest SHA
- [ ] Uncomment / update the second dataset entry in `configs/symptoms.yaml`
- [ ] `disease_whitelist` MUST match `manifest.diseases_trained` — startup will refuse otherwise
- [ ] Commit `configs/symptoms.yaml`

## Step 6: XGBoost algorithm (sibling to typed-BASD)

`algorithm_module: xgb` is a second implementation of the same
`train / tune / promote / serve` shape. Same CLI verbs, same manifest
chain, same subset-training flag — the underlying model is a pair of
gradient-boosted trees plus an information-gain question policy. Ships
CPU-native (no MPS/CUDA at train or serve time) and produces
human-readable feature importances.

### Why XGBoost as an alternative

typed-BASD's `next_action` is greedy argmax over the `sym` head — it
asks the evidence it thinks the patient probably has, NOT the evidence
that discriminates diseases. Under a subset like Pneumonia + Influenza
where classes share ~80% of common symptoms, this shows up as
non-discriminative questions (rash, pain location, travel history)
while the classifier reaches 100% accuracy through aggregate state +
init_matcher seeding alone. XGBoost's IG policy asks the evidence that
maximally reduces class entropy — hits P(class) = 0.998 after ONE
question on the same task, and feature importances are auditable
against clinical intuition.

### Train

```bash
uv run --extra symptoms-server claritymed-symptoms-xgb-train-ddxplus \
    --data-dir ~/.claritymed/data/symptoms/ddxplus \
    --diseases "Pneumonia,Influenza" \
    --n-train 300000 \
    --max-depth 6 --n-estimators 400 --lr 0.05 \
    --calibration platt \
    --mask-policy random
```

Writes `weights.pkl` (joblib) + `manifest.json` under
`ddxplus/run/xgb_subset_<slug>_<ts>/`. Manifest carries the same
`sha256 / diseases_trained / training_target` fields as typed-BASD plus
XGBoost-specific `feature_columns`, `feature_importance_top_k`, and
`algorithm_module_version`.

Key knobs beyond the typed-BASD defaults:

- `--calibration platt|isotonic|none` — posterior-calibration method.
  Platt (sigmoid) is default; isotonic overfits on small validate splits.
- `--mask-policy random|full` — `random` fits an ev-marginals regressor
  on (masked, full) pairs to inform IG at partial states. `full` skips
  it and falls back to global column means — smaller checkpoint,
  less-informed IG picks.
- `--keep-rate-lo / --keep-rate-hi` — random-mask keep-rate range
  (default 0.3, 0.7 — matches typed-BASD's `train_step` masking density).

### Tune

```bash
uv run --extra symptoms-server claritymed-symptoms-xgb-tune-ddxplus \
    --data-dir ~/.claritymed/data/symptoms/ddxplus \
    --weights ~/.claritymed/models/symptoms/ddxplus/run/xgb_subset_.../weights.pkl \
    --maxsteps 4,6,8,10,12 \
    --stop-thres 0.85,0.90,0.95,0.99 \
    --ig-smoothings 0.0,0.05,0.10
```

Different sweep space than typed-BASD (patho_temp is meaningless for
XGBoost — calibration is applied at training time). Same DSR-floor + IL
elbow selection. `stop_thres` interpretation flips: XGBoost stops when
max class prob **rises above** the threshold; typed-BASD stops when max
symptom prob **drops below** it. Same YAML field, opposite direction —
see `ModelSpec.stop_thres` docstring.

### Deploy

Identical to Step 4, but:

1. `weights.pkl` instead of `weights.pt` under the promoted dir.
2. `configs/symptoms.yaml` model entry uses `algorithm_module: xgb`.
3. Adapter cross-checks `manifest.feature_columns` against the live
   DDXPlus schema at load time — a column-order drift fails loud.

```yaml
- id: xgb_pneumonia_flu_v1
  algorithm_module: xgb
  weights_subpath: ddxplus/xgb_pneumonia_flu_v1
  manifest_sha256: "<sha>"
  maxstep: 6
  patho_temp: 1.0        # ignored by xgb
  stop_thres: 0.90       # confidence threshold, NOT symptom-prob floor
```

Point a dataset at both by listing `[typed_basd_pneumonia_flu_v1,
xgb_pneumonia_flu_v1]` under `model_ids` and choosing
`model_selection: round_robin` for A/B or `first` for single-primary.
