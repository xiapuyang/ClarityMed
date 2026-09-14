# 视觉模型服务 — 简化协议（中文）

本文档是给**外部对接方**看的最小协议规范，只覆盖一件事：**送一张图片，拿回一个疾病分类结果**。

---

## 1. 通信约定

- **协议**：HTTP/1.1，请求体与响应体都是 JSON（UTF-8）。
- **绑定**：参考实现**只监听回环地址**（`127.0.0.1`）。患者图像会流经该服务，
  暴露到公网即是安全回退。要跨主机部署，必须自己保证两端处于同一信任域
  （Unix socket 包装、内网 VPC 等）。
- **认证**：v1 没有任何认证；回环约束就是认证。客户端不得携带 `Authorization`。
- **版本**：URL 前缀带主版本号（`/v1/...`）。破坏性变更走 `/v2/...`，可在过渡期同时供应两个。
- **图片编码**：图片字节用 base64 包在 JSON 里；服务端会对解码后的字节重新算 SHA-256 校验。

## 2. 请求追踪头

每个请求**可以**带 `X-Request-ID`。服务端：

1. 把它作为追踪关联键；
2. 在**所有**响应（包括 4xx/5xx）里原样回显；
3. 如果客户端没传，自己生成一个（建议 UUIDv4）。

`/v1/detect` 的请求体里也带一份 `request_id` 字段，二者会保持一致。

---

## 3. 三个接口

| 方法 | 路径 | 用途 | 涉及 PHI？ |
|---|---|---|---|
| `GET`  | `/health`     | 进程是否就绪 + 加载的模型摘要 | 否 |
| `GET`  | `/v1/catalog` | 服务端的模型清单 | 否 |
| `POST` | `/v1/detect`  | 跑一张图、返回分类结果 | **是** |

没有批量接口，也没有流式接口。**一张图一个请求**。

---

### 3.1 `GET /health`

**请求**：无 body。

**200 响应**：

```json
{
  "status": "ok",
  "models_loaded": [
    { "disease_id": "breast_cancer_ultrasound", "model_id": "breast_busi_unet_v1" }
  ],
  "uptime_s": 412
}
```

- `status` ∈ `{"ok", "loading"}`。`loading` 表示模型还在加载，客户端应该重试。
- `models_loaded` 只列**成功加载**的模型。被 `enabled: false` 关掉的疾病不会出现。
- `uptime_s` 是非负整数。

---

### 3.2 `GET /v1/catalog`

返回服务端当前真正加载了哪些模型，用于客户端做配置一致性校验。

**请求**：无 body。

**200 响应**：

```json
{
  "models": [
    {
      "disease_id": "breast_cancer_ultrasound",
      "model_id": "breast_busi_unet_v1",
      "model_version": "v1.0.0",
      "framework": "pytorch",
      "task": "classification",
      "labels": ["benign", "malignant", "normal"],
      "cancer_class": true,
      "accepted_modality": "ultrasound",
      "expected_ms": 800
    }
  ]
}
```

逐字段说明：

- `framework` ∈ `{"pytorch", "onnx", "ultralytics"}`。
- `task` ∈ `{"classification", "detection"}`。
- `labels` 是**有序**的类别列表。`/v1/detect` 返回的概率向量与此顺序一一对应。
- `cancer_class: true` 表示模型自带"癌性/良性"映射，`/v1/detect` 响应会带
  `cancer_status` 字段；`false` 表示仅做普通分类。
- `accepted_modality` 是**硬门槛**：图片的影像模态必须等于这个值，否则
  服务端会 422 拒绝（防御纵深）。常见值：`ultrasound` / `ct` / `xray` /
  `dermoscopy` / `photo` / `document`。
- `expected_ms` 是模型在该机器上**单次推理的经验耗时**（毫秒），用作客户端
  超时预算的参考。

---

### 3.3 `POST /v1/detect`

跑一张图、拿回分类结果。

**请求**：

```json
{
  "request_id": "req_8c4f0a3b9d6e1f5a",
  "disease_id": "breast_cancer_ultrasound",
  "model_id": "breast_busi_unet_v1",
  "image": {
    "sha256": "8c4f0a3b... (64 位小写十六进制)",
    "data_b64": "iVBORw0KGgo..."
  },
  "language": "zh"
}
```

字段规则：

- `request_id`：1–64 字符，必填。
- `disease_id`：必须是 catalog 里 `enabled: true` 的疾病，否则 404。
- `model_id`：可选。缺省时服务端用 `disease.primary_model_id`。
- `image.sha256`：64 位小写十六进制。服务端解码 base64 后会重新算 SHA-256，
  不匹配就 400 拒绝。这一条阻止客户端绕过上游影像模态标签去偷换图片字节。
- `image.data_b64`：标准 base64，不能分块。
- `language` ∈ `{"en", "zh"}`：用于 `warnings` 等文本的本地化。

**200 响应**：

```json
{
  "request_id": "req_8c4f0a3b9d6e1f5a",
  "disease_id": "breast_cancer_ultrasound",
  "model_id": "breast_busi_unet_v1",
  "model_version": "v1.0.0",
  "elapsed_ms": 612,
  "input_quality": {
    "passed": true,
    "checks": [
      { "name": "min_resolution", "score": 512, "passed": true },
      { "name": "modality_match", "score": 1.0, "passed": true }
    ]
  },
  "classification": {
    "labels": ["benign", "malignant", "normal"],
    "probabilities": [0.12, 0.84, 0.04],
    "top1": "malignant",
    "top1_prob": 0.84,
    "confidence_tier": "high"
  },
  "cancer_status": "malignant",
  "clinical_action": "urgent_specialist",
  "labels_meta": {
    "benign":    { "description": "非癌性病灶，通常按常规随访处理。", "cancer_status": "benign",    "clinical_action": "routine_followup" },
    "malignant": { "description": "可疑癌性病变，建议尽快请专科医生复核。", "cancer_status": "malignant", "clinical_action": "urgent_specialist" },
    "normal":    { "description": "未发现病灶，仅就本图无需处理。", "cancer_status": "normal",    "clinical_action": "no_action" }
  },
  "warnings": [],
  "model_card_url": null
}
```

逐字段说明：

- `elapsed_ms`：服务端测量的端到端耗时（毫秒），整数 ≥ 0。
- `input_quality.passed`：质量门控（最小分辨率、模态置信度等）。失败会触发
  下面的"临床动作降级"。
- `classification.probabilities` 与 `classification.labels` 一一对应，
  概率之和 ≈ 1.0。`top1` 一定出现在 `labels` 里。
- `confidence_tier` ∈ `{"low", "medium", "high"}`，由模型清单里的
  `confidence_thresholds` 决定（或采用适配器默认值）。`low` 会触发
  "临床动作降级"。
- `cancer_status`：非 cancer-class 模型不返回。取值
  `{"benign", "malignant", "normal", "unknown"}`。
- `clinical_action` ∈ `{"urgent_specialist", "soon_specialist",
  "routine_followup", "no_action", "inconclusive_review"}`。
- `labels_meta`：每个类别的本地化描述与对应的临床动作；前端/LLM 可以原样展示给用户。
- `warnings`：字符串数组，详见下一节。
- `model_card_url`：可选，模型卡片链接。

**错误码**：见第 5 节。

### 临床动作降级（重要！）

在返回响应之前，服务端会做一次最终覆写：

```
if not input_quality.passed:
    clinical_action = "inconclusive_review"
    warnings.append("质量门控未通过…")
elif classification.confidence_tier == "low":
    clinical_action = "inconclusive_review"
    warnings.append("置信度偏低…")
```

- 质量门控的覆写**先于**置信度覆写——"重拍一张"比"模型不确定"更可操作。
- 覆写对**所有**模型生效，无论是否 cancer-class。
- `cancer_status` **不**会被覆写，只覆写 `clinical_action`。

---

## 4. 图片字段编码

```
ImagePayload {
  sha256:   "<64 位小写十六进制>"
  data_b64: "<标准 RFC-4648 base64，不分块，长度 ≥ 1>"
}
```

服务端处理顺序：

1. 严格 base64 解码（`validate=True`）；失败 → `400 image_decode_failed`。
2. 对解码后的字节算 SHA-256，必须等于 `sha256`；不匹配 →
   `400 image_hash_mismatch`，`details` 里带 `claimed` 与 `actual`。
3. 字节交给适配器做 `preprocess()`。

哈希校验不是安全控制，是**篡改信号**——上游附件管道把
`(sha256, modality)` 绑定在一起，允许客户端换图等于绕过模态硬门槛。

---

## 5. 错误格式

所有 4xx/5xx 响应统一长这样：

```json
{
  "error": {
    "code": "modality_mismatch",
    "message": "human-readable message",
    "request_id": "req_…",
    "details": {
      "model_accepts": "ultrasound",
      "image_modality": "xray"
    }
  }
}
```

- `code` 是**稳定的机器可读键**，客户端只能基于它分支，不能基于 `message`。
- `details` 的字段依 `code` 而定。

标准错误码（参考实现的映射）：

| HTTP 状态 | `code` | 触发场景 |
|---|---|---|
| 400 | `bad_request` | 请求体未通过 Pydantic 校验，且没有更具体的码 |
| 400 | `image_decode_failed` | `data_b64` 不是合法 base64 |
| 400 | `image_hash_mismatch` | 解码后字节的 SHA-256 与 `image.sha256` 不一致 |
| 404 | `unknown_disease` | `disease_id` 不在 catalog 里，或被禁用 |
| 404 | `unknown_model` | `model_id` 不在该疾病的 flow 里 |
| 422 | `modality_mismatch` | 防御纵深：图片模态 ≠ `model.accepted_modality` |
| 500 | `inference_failed` | 适配器抛了异常。`message` 是 `f"{exc类型}: {exc消息}"` |
| 503 | `service_unavailable` | 模型还在加载，或目标模型未加载到本进程 |

服务端可以扩展自己的私有 `code`，但需要在自家文档里说明。

---

## 6. 对接方自检清单

一个外部服务想成为 ClarityMed 兼容的视觉服务，只需做到：

- [ ] 默认绑定回环；
- [ ] 实现 `GET /health`（响应 shape 见 §3.1）；
- [ ] 实现 `GET /v1/catalog`（响应 shape 见 §3.2）；
- [ ] 实现 `POST /v1/detect`（请求/响应 shape 见 §3.3）；
- [ ] 在所有响应上回显 `X-Request-ID`（§2）；
- [ ] 强制图片 SHA-256 校验（§4）；
- [ ] 返回符合 §5 的错误信封；
- [ ] 实现 §3.3 末尾描述的临床动作降级逻辑。

做到这些以后，客户端只需要把 `configs/vision.yaml::servers[*].base_url`
指向你的服务即可，无需改一行 Python。

---

## 7. v1 故意不做的事

- **批量**：一张图一次请求，没商量。延迟预算（`tool.total_budget_ms`）
  是单次调用维度。
- **流式**：v1 不支持 `chunked` 响应。
- **异步任务队列**：detect 是同步的，客户端阻塞等待。
- **模型热替换**：模型在 lifespan 启动时一次性加载完，新增/替换模型需要重启服务。
- **跨服务路由**：catalog 是单服务视图，多服务之间的路由由客户端负责。
- **回环之外的认证**：v2 可能加 bearer token，v1 没有。
