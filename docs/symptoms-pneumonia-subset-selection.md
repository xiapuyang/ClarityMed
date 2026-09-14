# Pneumonia-focused DDXPlus 疾病子集筛选标准

**目标**：从 DDXPlus 49 个 pathology 里，选出一个 pneumonia-centric 的子集用于第二版训练，直接优化 Pneumonia 的下游 recall / precision / F1。

**核心方法论**：**baseline 全 49 类跑一遍 → 取 Pneumonia 的 confusion 行列 → 模型实际混淆什么就选什么。** 症状 Jaccard / IDF 加权 / ICD 层级都是代理指标，会被"发烧、乏力"这类通用宪法性症状污染（例：SLE / HIV / Anaphylaxis Jaccard 排前列但临床上完全不是 pneumonia 鉴别诊断）。confusion matrix 是**模型真实决策边界**的直接证据，其他都是先验猜测。

---

## 数据流

```
Phase A: 全 49 类 baseline
    claritymed-symptoms-train-ddxplus  →  weights.pt (49 类)
                    ↓
Phase B: 抽 Pneumonia 混淆
    scripts/select_ddxplus_subset.py  →  pneumonia_subset.yaml
                    ↓
Phase C: 重训子集 baseline
    load_pidx + subset filter  →  train.py (~15 类) →  评估 Pneumonia F1
```

---

## Phase B 筛选算法

### 输入

- 一个训完的 49 类 baseline checkpoint（`weights.pt` + train.py 产出的 manifest）
- `release_validate_patients.zip`（**不要用 test split，留给最终评估**）
- 目标类：默认 `Pneumonia`（脚本可换）
- Top-K 混淆数：默认 `8`
- Cannot-miss 白名单：见下

### Step 1 — 跑 baseline 拿 per-patient predictions

在 `release_validate_patients.zip` 上跑 `interactive_eval` 的同款推理循环（`initialize_state → reveal loop → diagnose`），但**收集** `(true_disease_id, pred_disease_id)` 对，聚合成 49×49 confusion matrix `C[i][j] = count(true=i, pred=j)`。

**样本量**：默认 `--eval-n 20_000`，够让每类至少 ~50 个样本（Pneumonia 有 ~530 在 20k 中，能给出稳定的 confusion 行）。低于 5k 时 confusion 噪声太大，脚本会 warn。

### Step 2 — 提 target 疾病的双向混淆

设 `t = target_id`。计算两个向量：

- **Recall-row**：`recall[j] = C[t][j] / sum_j C[t][j]`
  语义：真实是 Pneumonia 的病人，模型有多少比例把它预测为 `j`？→ **模型眼里 Pneumonia 长得像什么**
- **Precision-column**：`precision[j] = C[j][t] / sum_i C[i][t]`
  语义：模型预测为 Pneumonia 的病人里，有多少比例真的是 `j`？→ **什么疾病会被误诊为 Pneumonia**

两条方向都重要——一条负责 Pneumonia 的 recall（不漏），一条负责 precision（不错报）。**只看单向会漏一半**（例：如果 PE 常被错报为 Pneumonia 但 Pneumonia 很少被错报为 PE，只看 recall-row 会把 PE 漏掉）。

### Step 3 — combined score 排序

`combined[j] = recall[j] + precision[j]`（j ≠ t），从高到低排，取 top-K（默认 K=8）。

排序后剔除 `combined < 0.01` 的（噪声阈值——低于此的类被混淆的病例可能是 5 个人以内，不构成学习信号）。

### Step 4 — 合并 cannot-miss 白名单

有些疾病 **临床上必须能鉴别，即使 confusion 矩阵里没排上**（因为 baseline 模型在 49 类里已经能分对这些，但一旦子集训练把它们剔掉，第二版模型就"从未见过"这些疾病，遇到时会直接标 Pneumonia）。

默认硬编码集合（**基于临床鉴别诊断，不是数据**）：

```
Pulmonary embolism                     # 抗凝药 vs 抗生素，路径分岔
Tuberculosis                           # 慢性肺炎 mimic，公卫报告义务
Acute COPD exacerbation / infection    # SOB + cough，激素路径
Bronchospasm / acute asthma exacerbation  # 支气管扩张剂路径
Spontaneous pneumothorax               # 急性胸痛 + SOB，胸腔穿刺
Acute pulmonary edema                  # 心源性 vs 感染性，利尿剂 vs 抗生素
```

**这个列表是硬编码的，改动需要临床理由和 code review**。不要因为某次 confusion 排名把它们去掉。

### Step 5 — 样本量守卫

**下界**：train split 里 <`MIN_SAMPLES_TRAIN` (默认 5000) 的类**丢弃**，记录到 `dropped_low_sample`。理由：DDXPlus 的 `Bronchiolitis` (261) / `Croup` (2852) 样本太少，会被模型死记住或拉低 minority-class F1；若必须保留，改从其他数据源（如 pediatric-only corpus）单独训。

**上界**：train split 里样本数 >`Pneumonia × MAX_SAMPLES_RATIO`（默认 2×，即 ~50k）的类**设 sampling_cap** 为该上界，写入 YAML。理由：DDXPlus 里 `URTI` (64k) / `Viral pharyngitis` (62k) 各占 6%+，不 cap 会主导 cross-entropy loss，把 Pneumonia 的梯度信号淹没。

**Pneumonia 本身不 cap，也不上采样**（25k 已经够）。

### Step 6 — 输出 YAML

见下方 schema。三层分组：`target` / `from_confusion` / `cannot_miss`，加上 `dropped_low_sample` 保留 audit 痕迹。

---

## 输出 YAML schema

```yaml
version: 1
target: Pneumonia
generated_at: "2026-07-05T10:23:15+00:00"
source:
  baseline_weights: /Users/.../typed_basd_v2/weights.pt
  data_dir: /Users/.../ddxplus
  n_eval_patients: 20000
  eval_split: validate

selection_params:
  target_disease: Pneumonia
  top_confused_n: 8
  min_samples_train: 5000
  max_samples_ratio: 2.0
  combined_score_floor: 0.01

diseases:
  target:
    - name: Pneumonia
      train_samples: 25761
      sampling_cap: null

  from_confusion:
    - name: Bronchitis
      train_samples: 26400
      recall_share: 0.153       # P(pred=Bronchitis | true=Pneumonia)
      precision_share: 0.121    # P(true=Bronchitis | pred=Pneumonia)
      combined_score: 0.274
      sampling_cap: null
    # ... 其余 top-K

  cannot_miss:
    - name: Pulmonary embolism
      train_samples: 27468
      recall_share: 0.042
      precision_share: 0.018
      combined_score: 0.060
      sampling_cap: null
      reason: clinical_anticoagulation_pathway

dropped_low_sample:
  - name: Bronchiolitis
    train_samples: 261
    reason: below_min_samples_train

capped_high_sample:
  - name: URTI
    train_samples: 64368
    sampling_cap: 51522         # = 25761 × 2
    reason: exceeds_max_samples_ratio
```

---

## Phase C 训练修改

第二版训练读这份 YAML，在 `schema.load_pidx` 之后做子集过滤：

```python
whitelist = yaml.safe_load(open("pneumonia_subset.yaml"))
allowed = {d["name"] for tier in ("target", "from_confusion", "cannot_miss")
           for d in whitelist["diseases"][tier]}
sample_caps = {d["name"]: d["sampling_cap"]
               for tier in whitelist["diseases"].values() for d in tier
               if d.get("sampling_cap")}

# 过滤 patients — 只保留 pathology ∈ allowed
train_pats = [p for p in train_pats
              if id_to_name[p["d"]] in allowed]

# 应用 sampling_cap
for name, cap in sample_caps.items():
    class_pats = [p for p in train_pats if id_to_name[p["d"]] == name]
    if len(class_pats) > cap:
        # random sub-sample
        keep_ids = set(np.random.choice(len(class_pats), cap, replace=False))
        # ...
```

**重要**：子集训练**不改** `release_evidences.json` 里的 evidence set（还是全量 ~200+ evidence），只筛 pathology。这样第二版模型能问同样的问题，但只在这 ~15 类里判决。

---

## 评估：Pneumonia-focused metrics

`typed_basd.interactive_eval` 现在返回宏平均。评估第二版子集 baseline 时**改看 target-class per-metric**：

- **主指标**：Pneumonia 的 per-class recall / precision / F1（top-1）
- **次指标**：restricted DDF1，只统计 `env.disease == pneumonia_id` 的行
- **对照**：Pneumonia 被错分到 top-3 目标的比率（追 confusion 矩阵变化）
- **Bootstrap 95% CI**：F1 差异用 1000× resample 打置信区间，别拿单点数字下结论
- **DSR**：Pneumonia 若在 severity < 3 里（应该是），继续保留

`scripts/select_ddxplus_subset.py` 输出的 markdown 报告已经把这些数字打好了 baseline 对照（Phase A 阶段），Phase C 训完新模型后再跑同一份数据做前后对比。

---

## 快速命令

```bash
# Phase A: 训 baseline (已有流程)
uv run claritymed-symptoms-train-ddxplus \
    --data-dir ~/.claritymed/data/symptoms/ddxplus \
    --episodes 200000

# Phase B: 抽 confusion + 生成子集 YAML
uv run python scripts/select_ddxplus_subset.py \
    --data-dir ~/.claritymed/data/symptoms/ddxplus \
    --weights ~/.claritymed/models/symptoms/ddxplus/typed_basd_v2/weights.pt \
    --out configs/symptoms/pneumonia_subset.yaml \
    --top-confused-n 8

# 输出：configs/symptoms/pneumonia_subset.yaml + stdout markdown 报告

# Phase C: 用子集重训（train.py 直接读疾病名，不吃 YAML）
# 从 Phase B 输出的 pneumonia_subset.yaml 里 diseases.target + from_confusion +
# cannot_miss 手工拼成一个逗号分隔的字符串传给 --diseases。
uv run claritymed-symptoms-train-ddxplus \
    --data-dir ~/.claritymed/data/symptoms/ddxplus \
    --diseases "Pneumonia,Bronchitis,URTI,Tuberculosis,Bronchiectasis,..." \
    --out-subpath ddxplus/pneumonia_v1
```
