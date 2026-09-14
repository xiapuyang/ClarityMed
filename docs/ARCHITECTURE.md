# ClarityMed 架构文档

> 多模态、双语、本地优先的患者向医疗问答系统
> 研究项目（SYDE 660）· 核心价值 = 结构化不确定性传播 + 评测严谨性
> 状态：架构冻结（v1 范围）· 本文档为团队共享基线

---

## 0. 一句话定位

一个**单编排器 + 工具**的医疗问答系统：用户用配置语言（默认英语，支持中文）输入**文字咨询 / 化验报告 / 皮肤照 / 胸片**，系统经确定性安全闸与路由分发到对应模块，**让 LLM 只做受约束的自然语言层、绝不触碰原始数值与参考范围**，最终输出带**引用、校准置信度、来源、红旗提示**的答案。它**不是** multi-agent，**不是**自治 agent；唯一的自主循环关在 RAG 的自纠正里。

---

## 1. 冻结决策清单（团队基线）

| 编号 | 决策 | 取值 |
|---|---|---|
| A1 | 项目性质 | 研究项目，重评测严谨性 + 不确定性叙事 |
| A2 | 目标用户 | 面向患者（术语降级、强红旗分流、"何时就医"引导） |
| A3 | 语言 | 双语可配置，**默认英语**；回复**一定使用用户选择的语言** |
| A4 | MVP 范围 | 文本 RAG 竖切 + 评测台先行；化验单次之；影像最后 |
| A5 | 模型部署 | 本地优先，可切换云模型（**云路径含 PHI 红线**） |
| A6 | 病史写回 | v1 **只读**档案 + 确定性注入；事实抽取回写放 v2 |
| G1 | 知识库与语言 | **一套英文权威知识库（RAG 数据复用）+ 双语言交付层**；不建两份知识索引 |
| G2 | 语言/证据不一致 | 用配置语言作答 + **始终标注证据来源**（含语言/日期），不拒答、不硬翻证据 |
| G3 | 术语对齐 | v1 **必需**（UMLS + CMeKG + LOINC，中英映射） |
| G4 | 论文核心论点 | 结构化不确定性传播 + 校准/conformal 在多模态医疗 QA 上的可靠性 |
| G5 | 评测套件 | **全套**：MCQA 地板线 / RAGAS / Rubric(HealthBench+CMB-Clin) / 影像 VQA / 自建标注；**各语言分别评** |
| G6 | 影像范围 | 选定两种疾病：**常见皮疹**（皮肤）+ **ChestX-ray14**（胸片）；per-disease 模型**动态注册 + 热加载** |
| 其他-1 | 日志 | 结构化 logging + **审计日志独立**（含检索内容/置信度/是否转人工） |
| 其他-2 | 多语言 | 横切；一套知识库 + 双语外壳（红旗/免责/UI/术语双语等价） |
| 其他-3 | Prompt 管理 | **统一注册表 + 轻量管理界面**（版本化 YAML + 简单 web 表单），按语言加载 |
| 其他-4 | base 模型切换 | 声明式配置 + provider 抽象（运行时可换本地/云） |
| 其他-5 | OCR 模型切换 | 接口 + 适配器（PaddleOCR-VL / MinerU / 云医疗 OCR） |
| 其他-6 | vision 模型注册 | 注册表 + 热加载运行时，统一 `DiseaseVisionModel` 契约 |
| 其他-7 | 个人/病史维护 | 档案 + 纵向记录 CRUD（admin 入口）；v1 只读注入 |

---

## 2. 核心原则（贯穿全系统，违反即设计错误）

1. **最小自主性**：确定性优先；每多一个自主决策点 = 多一个不可评测、难追责的失败点。唯一 agentic 循环 = text_rag 的自纠正。
2. **LLM 不碰数值**：化验数值、参考范围、剂量一律走确定性代码；LLM 只做受约束的解释层。让 LLM 读/判数值或编造参考范围 = 唯一会出人命的错。
3. **个人信息不入 RAG**：结构化档案确定性注入 + 条件化知识检索；个人信息靠向量"召回"会漏致命事实（如过敏）。知识库与个人库**物理隔离**。
4. **不确定性按来源/类型分解传播**：区分偶然（aleatoric→要求补输入）与认知（epistemic→给候选集/转人工）；不揉成一个标量、不跨模态相乘。
5. **安全先于功能**：红旗闸是第一道闸，绕过 RAG/LLM，优先级最高。
6. **核心可替换、前端可移植**：编排是最薄、最该可替换的一层（将来换 LangGraph/加多 agent 只动 `orchestrator/`）；前端走 AG-UI 开放协议（框架无关）。

---

## 3. 整体架构（数据流）

```
用户输入（文字 / 化验单 / 皮肤照 / 胸片 + language 配置）
   │
   ▼
[1] 红旗安全闸（确定性规则，双语）──命中──▶ 急症/就医指引（绕过下游）
   │ 未命中
   ▼
[2] 确定性路由（规则 + 轻分类器：模态/意图）   ──含 PHI 守卫：云模型路径禁传 PHI
   │
   ├─ 文字 ─▶ text_rag（跨语言 query → 混合检索 → rerank → [自纠正循环] → faithfulness）
   ├─ 化验单 ─▶ lab_pipeline（L1 OCR → L2 归一/LOINC → L3 规则判读 → 自洽校验）
   ├─ 皮肤照/胸片 ─▶ vision（注册表取模型 → 校准概率 → conformal 预测集 → OOD/质量门）
   └─ 追问 ─▶ 带病史的多轮
   │
   ▼
[3] 上下文装配（分区，标来源/时效/置信）
     【患者事实（档案，确定性注入，勿改）】|【检索知识（带引用+日期）】|【既往记录】
   │
   ▼
[4] 编排器 / Pydantic AI Agent（受约束生成 → GroundedAnswer；输出强制符合 schema）
   │
   ▼
[5] 不确定性融合 + 策略（校准 → conformal → 语义熵 → 晚融合；矛盾升不确定、弱环节封顶）
   │   ├─ 偶然高 ─▶ 要求补输入（重拍/追问）
   │   └─ 认知高 ─▶ 给候选集 / 转人工（弃答）
   ▼
[6] 安全/组合层（按 language 的免责措辞、副作用审批门、审计日志）
   │
   ▼
输出（答案 + 引用 + 置信档位&原因 + 红旗 + 来源日期，全部用 language）
```

横切：**i18n 外壳 · 可观测(OTel) · 审计 · 评测子系统 · per-user PHI 隔离** 贯穿全程。

---

## 4. 技术选型

| 层 | 选型 | 备注 |
|---|---|---|
| 编排 | **Pydantic AI** + 自写确定性路由 | 薄；唯一 agentic 循环在 RAG；类型化输出承载 `UncertaintyResult` |
| 文本模型 | Qwen3-14B / 30B-A3B，经 Ollama / LM Studio（OpenAI 兼容） | 本地优先；开发期可云模型兜底编排（受 PHI 守卫约束） |
| 视觉基座 | Lingshu-7B / MedGemma-4B，经 mlx-vlm；+ per-disease 专用模型 | 作为工具调用；专用模型热加载 |
| Embedding | bge-m3（dense + sparse + ColBERT） | MPS；跨语言检索（中文 query 检英文库） |
| 向量库 | Qdrant（hybrid + payload 过滤） | 单一英文知识库；payload 做来源/日期/per-user 过滤 |
| Reranker | bge-reranker-v2-m3 | cross-encoder 精排 |
| 化验 OCR | PaddleOCR-VL / MinerU（适配器可切换） | 两阶段抗幻觉；数值自洽校验 |
| 档案/纵向 | SQLite（SQLModel），类 FHIR | per-user；**不入 RAG** |
| 聊天记忆 | LanceDB + SQLite | 独立库；v1 只读 |
| 校准 | temperature scaling + ECE/Brier（netcal） | 可靠性图 |
| Conformal | MAPIE / crepes | 预测集 + 弃答；皮疹做肤色分层 |
| 语义熵 | 自写（采样 + 嵌入聚类） | 文本侧幻觉/不确定 |
| CLI | Typer + Rich | 入口；置信档位/红旗渲染 |
| Web（后续） | FastAPI + 原生 AG-UI + CopilotKit(React) | 类 Codex 交互：工具步骤可见 + 审批门 |
| 管理界面 | 轻量（版本化 YAML + 简单 web 表单） | prompt/档案/病史/模型切换 |
| 可观测/评测 | OTel + Phoenix/Langfuse + RAGAS + pydantic-evals | 评测一等公民 |
| 术语 | LOINC + UMLS + CMeKG | 中英对齐（G3 必需） |

---

## 5. 模块拆分（按职责，各自可独立测试）

- **core/orchestrator**：红旗闸（safety）、确定性路由（router）、Pydantic AI agent、PHI 守卫（phi_guard）
- **core/schemas**：全部 Pydantic 契约（`UncertaintyResult` / `Patient` / `LabPanel` / `DiseaseVisionModel` 接口 / `GroundedAnswer`）—— **系统骨架**
- **core/prompts**：prompt 注册表（版本化、按语言）
- **core/i18n**：语言外壳（按 `language` 取文案 / 决定生成语言）
- **core/context**：分区上下文装配
- **core/uncertainty**：calibration / conformal / entropy / fusion / policy
- **core/stores**：knowledge(Qdrant) / profile(SQLite, CRUD) / chat_memory(LanceDB)，适配器隔离
- **core/observability**：logging（结构化）/ audit（独立审计）/ otel
- **tools/text_rag**：跨语言 query → 混合检索 → rerank → 自纠正 → faithfulness
- **tools/lab_pipeline**：L1 extract（OCR 适配器）/ L2 normalize（LOINC、单位）/ L3 flag（报告自带区间）/ interpret（规则）/ 自洽校验
- **tools/vision**：registry（注册+热加载）/ base（`DiseaseVisionModel`）/ models/{rash, chestxray14}
- **tools/memory**：聊天检索（v1 只读）/ 事实抽取（v2）
- **ingest（离线）**：语料加载 / bge-m3 建索引 / LOINC 表
- **cli / api / admin**：入口与管理界面
- **evals（一等公民）**：检索指标 / RAGAS / rubric / 校准 / conformal，各语言分别评

---

## 6. 横切关注点设计

### 6.1 多语言（i18n）
- **一套英文知识库**；中文 query 经 bge-m3 跨语言检索（需测召回）。
- `language` 配置贯穿：query 处理、生成语言、红旗规则、免责措辞、UI 文案、术语表。
- 红旗/免责**双语两套且语义等价**，放 `configs/i18n/{en,zh}.yaml`。
- 证据始终标注来源（含语言/日期）；语言不一致时用配置语言作答（G2）。
- **评测各语言分别做**（英文 MIRAGE/HealthBench，中文 CMB-Clin）。

### 6.2 Prompt 管理
- prompt 不散在代码，集中为**版本化注册表**（`core/prompts/store/*.yaml`），按语言加载。
- v1 管理界面 = admin 里的 YAML 编辑 + 简单 web 表单（版本可见）；A/B、灰度放 v2。
- prompt 作为评测变量接入 pydantic-evals。

### 6.3 可切换 / 可注册（统一模式：接口 + 注册表 + 配置）
- **base 模型**：`configs/models.yaml` + Pydantic AI provider 抽象；运行时切本地/云。
- **OCR 模型**：`tools/lab_pipeline/extractors/` 适配器 + `configs/ocr.yaml`。
- **vision 模型**：`tools/vision/registry.py` + `configs/vision_registry.yaml`；详见 §7。

### 6.4 PHI 守卫（A5 的硬约束）
- 云模型路径**禁传 PHI** 或按用户显式开关 + 提示；规则进 `configs/safety.yaml` 与 `orchestrator/phi_guard.py`。
- per-user 强隔离；用户数据加密；不进训练。撞 PIPEDA + PIPL，本地优先把风险压到最低。

### 6.5 日志与审计
- **结构化 logging**（运行调试）与**审计日志**（每个答案可追溯：检索了什么、置信度、是否触发审批/转人工）**分离**；审计复用不确定性 provenance。

---

## 7. Vision 注册表设计（G6 核心）

> 实现细节：`docs/plans/2026-06-14-001-feat-vision-detection-plan.md`。

### 7.1 v1 契约 `DiseaseVisionModel`
每个 per-disease 模型实现同一接口，注册即生效，无需改编排：
- `preprocess(image) -> tensor`
- `predict(tensor) -> raw_scores`
- `calibrate(raw_scores) -> dict[label, prob]`（温度缩放等）
- `segment(tensor) -> SegmentationResult | None`（仅分割任务）
- `quality_gate(image) -> InputQuality`（分辨率、模态匹配等）
- `spec: ModelSpec`：疾病、模态、权重路径、版本、`manifest_sha256`

**v2 deferred**：`conformal_set(calibrated_probs, alpha)` 与 `ood_score(image)` 在 v1 没有落地——校准 softmax + `confidence_tier` 已经能驱动 `clinical_action` 的分支（KTD-V10 把低置信度强制覆盖为 `inconclusive_review`）。conformal/OOD 会在 v2 跟随多疾病铺开一起加。**肤色分层校准** 同样 deferred：v1 上线的 `breast_cancer_ultrasound` 不依赖肤色；当 `skin_cancer_dermoscopy` 落地（v1.1）时，再按 Fitzpatrick17k 做分层。

### 7.2 注册 + 启动时锁定
- `core/vision/registry.py::VisionRegistry` 读 `configs/vision.yaml`，启动时 `bootstrap()` 调每个 server 的 `/v1/catalog`，按 `model_id` 交叉校验 `manifest_sha`、`accepted_modality`、`cancer_class`；不一致就 fail-loud。
- 路由完全靠 config（`ModelSpec.server_id` 指明谁托管哪个模型），**没有**运行时热加载、**没有**定时刷新——运维换模型 = 改 YAML + 重启 orchestrator（KTD-V2）。多 server / 周期刷新留给 v1.x。
- Server 自己读 `manifest.json` 时跑两层 sha256 链（KTD-V7）：`configs/vision.yaml::models[i].manifest_sha256` ↔ manifest 自身 sha256 ↔ `manifest.sha256_weights` ↔ 权重文件 sha256。任一层不一致拒绝启动。

### 7.3 v1 首发疾病
| 疾病 | 数据集 | 模型 | 关键设计 |
|---|---|---|---|
| **breast_cancer_ultrasound** | BUSI（Kaggle Dataset_BUSI_with_GT；良/恶/正三类 + GT 掩码） | U-Net backbone 同时驱动分类头 + 分割头 | 唯一覆盖「分类 + 分割」的完整 schema cell；`cancer_class=true` 强制 manifest 携带 `cancer_status_mapping` + `clinical_action_mapping`。详见 `docs/vision-model-workflow.md`。|

> **未来疾病**（`skin_cancer_dermoscopy` / `lung_cancer_ct` / `chest_disease_ct`）走「registry 加一行 + 训练跑一遍」的零代码路径——分阶段铺开的动机在 [vision plan §"Deferred"](plans/2026-06-14-001-feat-vision-detection-plan.md#deferred-to-separate-tasks) 里说明。

### 7.4 两条独立的 reply-tone 轴（KTD-V1）
症状与影像是两条平行的紧急度语义，**不要**共用：

- 症状工具（`predict_disease_from_symptoms`）→ `severity_tier ∈ {Critical, Urgent, Moderate, Mild}`，急性分诊用，配合 `symptoms.safety_keywords.<tier>` 审计。
- 影像工具（`detect_disease_from_image`，癌性）→ `clinical_action ∈ {urgent_specialist, soon_specialist, routine_followup, no_action, inconclusive_review}`，配合 `vision.specialist_keywords.<action>` 审计。

恶性发现需要的是「几天内看专科」而不是「打 120」；把 `malignant` 映射进 `severity_tier=2/Urgent` 会继承 ER 急救措辞，临床上是错的。两套 enum、两个 reply prompt（`symptoms_final_reply.yaml` / `vision_final_reply.yaml`）、两组 i18n keys 独立维护。

---

## 8. 不确定性设计（论文核心，G4）

沿管线传播，每条边带**类型**（aleatoric/epistemic）与**来源**（自述/实测/检索/影像）：

- **文字**：检索分/支持文档一致性、RAGAS faithfulness/context-recall、自洽采样/语义熵。
- **化验**：每字段 OCR 置信 + MCV/MCH/MCHC 自洽校验 + 参考区间是否存在；判读本身确定性（应高且不确定性已局部化）。
- **影像**：校准 softmax + conformal 预测集 + OOD/质量门 + 集成方差；皮疹按肤色分层。
- **融合**：晚融合按可靠性加权；**矛盾升不确定、弱环节封顶**，不天真相乘。
- **策略**：偶然高→要求补输入；认知高→给候选集/转人工（selective prediction）。
- **沟通**：用户侧给**校准档位（低/中/高）+ 原因**，不给假精确小数；学术侧报 ECE/Brier/可靠性图 + 风险-覆盖率曲线（**论文主结果**）。

---

## 9. 评测子系统（G5，一等公民）

分层评，各语言分别做：

1. **检索层**：recall@k、nDCG、MRR、context precision/recall。
2. **接地层**：RAGAS faithfulness / context recall / answer relevancy（无参考，RAG 改善的主信号）。
3. **开放式质量层**：英文 HealthBench / MIRAGE 子集，中文 CMB-Clin（rubric 多点判定）。
4. **影像层**：VQA-RAD/SLAKE 等公开基准 + 少量自建标注；皮疹/胸片各自的校准/conformal 指标。
5. **地板线**：MedQA/CMB-Exam 选择题——只验证 RAG 没损坏原有知识（回归测试），不当主指标。
6. **可靠性**：ECE/Brier、可靠性图、风险-覆盖率曲线（**贯穿所有模态，论文核心图**）。

> 金标来源：英文复用 MIRAGE/HealthBench，中文用 CMB-Clin，影像用公开 VQA + 自建标注（几百条，尽早定规模——时间主要消耗点）。

---

## 10. 项目目录结构（适配现有 ClarityMed 单包 src/ 布局）

```
ClarityMed/
├── CLAUDE.md  ·  README.md  ·  LICENSE
├── pyproject.toml              # uv 单包；[project.scripts] 暴露 CLI 入口
├── uv.lock
├── docs/
│   ├── ARCHITECTURE.md         # 本文档
│   └── decisions/              # ADR（可选，记录后续变更）
├── configs/                    # 声明式配置（阈值/规则外置，不写死在代码）
│   ├── models.yaml             # base 模型可切换（本地/云）
│   ├── ocr.yaml                # OCR 模型可切换
│   ├── vision_registry.yaml    # 已注册 per-disease 模型清单
│   ├── retrieval.yaml          # hybrid 权重/top-k/rerank-k
│   ├── uncertainty.yaml        # 校准参数/conformal α/弃答阈值
│   ├── safety.yaml             # 红旗规则/免责/转人工/PHI 红线（双语）
│   └── i18n/  en.yaml · zh.yaml
├── data/                       # gitignore 大文件
│   ├── knowledge/              # 单一英文权威语料（MedCorp 子集等）
│   ├── qdrant/                 # 向量索引
│   ├── users/                  # PHI，加密，per-user 隔离
│   └── vision_models/          # 热加载权重 <disease>/
├── src/claritymed/
│   ├── core/
│   │   ├── orchestrator/       # router.py · agent.py · safety.py · phi_guard.py
│   │   ├── schemas/            # uncertainty.py · patient.py · lab.py · vision.py · answer.py
│   │   ├── prompts/            # registry.py · store/*.yaml（版本化、按语言）
│   │   ├── i18n/               # loader.py
│   │   ├── context/            # assembler.py（分区注入）
│   │   ├── uncertainty/        # calibration · conformal · entropy · fusion · policy
│   │   ├── stores/             # knowledge.py · profile.py(CRUD) · chat_memory.py
│   │   └── observability/      # logging.py · audit.py · otel.py
│   ├── tools/
│   │   ├── text_rag/           # retriever · reranker · self_correct · faithfulness
│   │   ├── lab_pipeline/
│   │   │   ├── extract.py · normalize.py · flag.py · interpret.py · selfcheck.py
│   │   │   └── extractors/     # base.py · paddleocr_vl.py · mineru.py（OCR 适配器）
│   │   ├── vision/
│   │   │   ├── registry.py     # 注册 + 热加载
│   │   │   ├── base.py         # DiseaseVisionModel 契约
│   │   │   └── models/  rash/ · chestxray14/
│   │   └── memory/             # retrieve.py（v1）· fact_extract.py（v2）
│   ├── ingest/                 # 离线：corpus/ · embed_index.py · loinc.py
│   ├── cli/                    # Typer+Rich：main.py(ask/upload-lab/upload-image/profile/history) · render.py
│   ├── api/                    # FastAPI+AG-UI（web 阶段）：main.py · ag_ui.py · auth.py · sessions.py
│   └── admin/                  # 轻量：prompt/档案/病史/模型·OCR·vision 切换
├── web/                        # CopilotKit React（web 阶段，独立前端，走 AG-UI）
├── evals/                      # 全套评测
│   ├── datasets/               # MIRAGE 子集 · CMB-Clin · 影像 VQA · 自建标注
│   └── retrieval_eval.py · ragas_eval.py · rubric_eval.py · calibration_eval.py · conformal_eval.py
└── tests/                      # 每个工具独立单测
```

设计要点：**契约(schemas) 与 评测(evals) 是一等公民；编排薄、可替换；存储/模型/OCR/vision 用"接口+注册表+配置"统一模式；阈值/规则/参考区间外置到 configs/；离线 ingest 与在线服务分离；CLI/api/admin 共用同一个 `claritymed` 包。**

---

## 11. 构建顺序（MVP 竖切，每片可独立评测）

1. **契约 + 评测台先行**：定死 `schemas/`（尤其 `uncertainty.py`、`vision.py` 契约）；搭 RAGAS + 检索指标管线。**没有评测，后续无法判断改进。**
2. **文本 RAG 竖切**：跨语言检索 + rerank + 自纠正 + faithfulness（双语各评）。
3. **化验单管线**：L1-L4 + 自洽校验（确定性、高价值、低风险）。
4. **影像竖切**：vision 注册表 + 两个疾病模型（皮疹肤色分层 / 胸片多标签）+ conformal。
5. **个人存储 + 上下文装配**：档案 CRUD、纵向记录、分区注入。
6. **不确定性融合 + 安全层**：校准/conformal/语义熵、红旗、PHI 守卫、审计、双语免责。
7. **管理界面（admin）** + **Web 阶段**：FastAPI + AG-UI + CopilotKit。

CLI 从第 1 步起即为薄壳；每个工具脱离编排器单测。

---

## 12. 待跟踪风险

- **本地小模型工具调用不稳**：靠 Pydantic AI 结构化校验+重试兜底；或用 Qwen3-30B-A3B；开发期云模型兜底（受 PHI 守卫）。**最可能卡住处。**
- **自建标注集规模**：研究项目最大时间黑洞，尽早定。
- **皮疹肤色偏移**：不做分层校准会得到对深肤色不可靠却"自信"的模型——安全与论文双重风险。
- **双语评测翻倍**：G3/G5 使工作量上升，排期要留出。
- **PHI 出机风险**：云切换的红线必须在 v1 就堵死，不留口子。
- **跨语言检索召回**：中文 query 检英文库的召回要实测，不达标则加 query 翻译。
```
