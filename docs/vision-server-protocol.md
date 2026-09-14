# Vision Model Server — Protocol Specification

This document is the language-agnostic interface contract for the
ClarityMed vision model server. The reference implementation lives in
`src/claritymed/servers/vision/`, but **anything** speaking this
protocol is plug-compatible: drop a new `base_url` into
`configs/vision.yaml::servers[*]` and the orchestrator routes traffic
to it without code changes.

The authoritative wire schemas are the Pydantic models in
[`src/claritymed/core/vision/wire.py`](../src/claritymed/core/vision/wire.py)
and [`src/claritymed/core/vision/schemas.py`](../src/claritymed/core/vision/schemas.py).
This doc reproduces shapes and rules for human readers — when in doubt,
the Pydantic models win.

The executable conformance test set is
[`tests/servers/vision/`](../tests/servers/vision/). Implementations
that pass those tests are considered compatible.

---

## 1. Posture

- **Transport:** HTTP/1.1, JSON request and response bodies, UTF-8.
- **Network:** the reference server **must bind loopback only**
  (`127.0.0.1`). PHI traverses this server — exposing it on a routable
  interface is a security regression (KTD-V8). Hosted alternatives are
  permitted only when the server runs inside the same trust boundary
  as the orchestrator (e.g. a sidecar on the same host, a Unix domain
  socket wrapper, or a private VPC peered into the same machine).
- **Authentication:** none. The loopback constraint *is* the auth
  story. Adding bearer tokens to a future revision is permitted; v1
  has none and clients must not send `Authorization`.
- **Encoding:** image bytes travel as base64 inside JSON. The server
  re-hashes the decoded bytes (see §5) — a tampered payload is
  rejected before any model touches it.

## 2. Versioning

- The URL prefix carries the major version: `/v1/...`.
- Breaking changes (request/response shape, semantic of a field) ship
  under a new prefix (`/v2/...`); the server may serve both during a
  deprecation window.
- The error envelope and `request_id` semantics are part of the
  versioned contract.
- `Manifest.model_version` (per-model semver) is **independent** of
  the protocol version; bumping a checkpoint never breaks wire compat.

## 3. Common header — `X-Request-ID`

Every request **may** carry `X-Request-ID`. The server:

1. Uses the incoming value as the trace correlation key.
2. Echoes the value back in the response headers on **every** path,
   including 4xx/5xx envelopes.
3. When absent, derives a value (UUIDv4 is recommended) and echoes the
   derived value.

The same id flows into request/response bodies as `request_id` for
`/v1/detect` so the audit log can pivot on a single key
(orchestrator → tool → server → manifest hash chain).

## 4. Endpoints

| Method | Path | Purpose | PHI? |
|---|---|---|---|
| `GET`  | `/health`     | Per-process readiness summary | No  |
| `GET`  | `/v1/catalog` | Server's truth about every loaded model | No  |
| `POST` | `/v1/detect`  | Run one image through one model | **Yes** |

There is **no batch** endpoint and **no streaming** endpoint in v1.
One image, one request.

---

### 4.1 `GET /health`

Returns process readiness and a summary of loaded models. Used by
operators and by the orchestrator's per-server health-check loop.

**Request:** no body.

**Response (`200`):**

```json
{
  "status": "ok",
  "models_loaded": [
    { "disease_id": "breast_cancer_ultrasound", "model_id": "breast_busi_unet_v1" }
  ],
  "uptime_s": 412
}
```

- `status` ∈ `{"ok", "loading"}`. `loading` means the lifespan hasn't
  finished verifying the manifest chain (see §6); clients should
  retry. The server never returns `status: "ok"` if any configured
  model failed integrity verification — it crashes on boot instead.
- `models_loaded` lists only **successfully verified** models. A
  disabled disease (`configs/vision.yaml::diseases[*].enabled: false`)
  contributes nothing here.
- `uptime_s` is a non-negative integer.

**Response (`503`):** when the lifespan handler crashed but the
process is still up (uncommon — most failures crash the server). Body
follows the error envelope (§7).

---

### 4.2 `GET /v1/catalog`

The server's truth about every loaded model. The orchestrator hits this
at boot and cross-references field-by-field against
`configs/vision.yaml::models` (KTD-V2). Any disagreement aborts the
client startup — this is the contract that lets the orchestrator
refuse to serve a tampered or stale server.

**Request:** no body.

**Response (`200`):**

```json
{
  "models": [
    {
      "disease_id": "breast_cancer_ultrasound",
      "model_id": "breast_busi_unet_v1",
      "model_version": "v1.0.0",
      "framework": "pytorch",
      "task": "classification+segmentation",
      "labels": ["benign", "malignant", "normal"],
      "cancer_class": true,
      "accepted_modality": "ultrasound",
      "manifest_sha": "0123abcd... (64 hex chars)",
      "expected_ms": 800,
      "supports_saliency": false,
      "supports_tta": false
    }
  ]
}
```

Per-field rules:

- `framework` ∈ `{"pytorch", "onnx", "ultralytics"}`. Adding a value
  here is a `ModelFramework` Literal update plus one adapter file.
- `task` ∈ `{"classification", "classification+segmentation", "detection"}`.
  Controls whether `/v1/detect` may return a `segmentation` block.
- `labels` is the **ordered** class list. Probabilities returned by
  `/v1/detect` are aligned to this order.
- `cancer_class: true` means the model carries
  `cancer_status_mapping` + `clinical_action_mapping` in its manifest;
  detection responses then include `cancer_status` and use the model's
  mapping for `clinical_action`. `false` means classification-only,
  `cancer_status` is omitted, `clinical_action` defaults to
  `"routine_followup"` until KTD-V10 fires.
- `accepted_modality` is the **hard gate** (KTD-V3). The orchestrator
  refuses to invoke a model whose `accepted_modality` doesn't match
  the image's tagged modality; the server itself enforces the same
  rule as defence-in-depth (see §4.3 status 422).
- `manifest_sha` is the lowercase-hex SHA-256 of `manifest.json` on
  disk. This is the **config-side root** of the integrity chain (§6).
- `expected_ms` is the soft per-call latency budget; the
  orchestrator's fallback-skip logic reads this.
- `supports_saliency` / `supports_tta` — capability hints. See §8.

---

### 4.3 `POST /v1/detect`

Run one image through one model. The body of the response is the
**richest** payload in the protocol; the orchestrator strips it down
to `LLMDetectionPayload` before handing anything to the LLM.

**Request:**

```json
{
  "request_id": "req_8c4f0a3b9d6e1f5a",
  "disease_id": "breast_cancer_ultrasound",
  "model_id": "breast_busi_unet_v1",
  "image": {
    "sha256": "8c4f0a3b...  (64 hex chars)",
    "data_b64": "iVBORw0KGgo..."
  },
  "language": "en",
  "options": {
    "return_segmentation": true,
    "return_saliency": false,
    "tta": false
  }
}
```

Field rules:

- `request_id` — 1–64 chars. Required. The server echoes it in the
  response body and in `X-Request-ID`.
- `disease_id` — must be `enabled: true` in the catalog. Unknown or
  disabled disease → `404`.
- `model_id` — optional. When absent, the server resolves
  `disease.primary_model_id`. When present, the value must appear in
  `disease.flow`; otherwise `404`.
- `image.sha256` — 64-char lowercase hex. The server SHA-256s the
  decoded bytes and rejects a mismatch (`400 image_hash_mismatch`).
  This blocks a client from bypassing the attachment-pipeline modality
  tag by swapping bytes after the tag was assigned.
- `image.data_b64` — standard base64, no chunking. The server enforces
  strict mode (`validate=True`).
- `language` ∈ `{"en", "zh"}`. The server uses this to localize
  warnings + `labels_meta.description`. (v1 BUSI ships EN baked in;
  ZH localization happens at the orchestrator's i18n layer in the
  current implementation. The field is reserved so a future server
  can ship its own translation.)
- `options.return_segmentation` — when `true` and the model's `task`
  includes segmentation, the response includes a `segmentation`
  block. Models without a segmentation head silently ignore the flag
  (§8).
- `options.return_saliency` — when `true` and the model's
  `supports_saliency` is `true`, the response includes
  `saliency_b64`. Otherwise no-op + warning (§8).
- `options.tta` — when `true` and the model's `supports_tta` is
  `true`, the server runs test-time augmentation (configurable per
  adapter). Otherwise no-op + warning.

**Response (`200`):**

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
  "segmentation": {
    "mask_png_b64": "iVBORw0KGgo...",
    "bbox": [42, 51, 187, 203],
    "area_ratio": 0.094
  },
  "saliency_b64": null,
  "labels_meta": {
    "benign":    { "description": "Non-cancerous lesion. Routine follow-up is usually appropriate.", "cancer_status": "benign", "clinical_action": "routine_followup" },
    "malignant": { "description": "Suspicious for cancer. A breast specialist should review the image.", "cancer_status": "malignant", "clinical_action": "urgent_specialist" },
    "normal":    { "description": "No lesion identified. No action required from this image alone.", "cancer_status": "normal", "clinical_action": "no_action" }
  },
  "warnings": [],
  "model_card_url": null
}
```

Per-field rules:

- `request_id`, `disease_id`, `model_id`, `model_version` round-trip
  from the request + catalog. Clients verify these match what they
  asked for.
- `elapsed_ms` is the server-measured wall-clock from request entry to
  response build, integer, ≥ 0.
- `input_quality.passed` reflects the per-adapter quality gate
  (min-resolution, modality confidence, etc.). `false` triggers the
  KTD-V10 override (§9).
- `classification.probabilities` aligns 1:1 with `classification.labels`
  and sums to ~1.0 (calibrated). `top1` must appear in `labels`.
- `confidence_tier` ∈ `{"low", "medium", "high"}` — comes from the
  manifest's tuned `confidence_thresholds` (when present) or a
  per-adapter default. `"low"` triggers KTD-V10 (§9).
- `cancer_status` is omitted for non-cancer-class models. Values:
  `{"benign", "malignant", "normal", "unknown"}`.
- `clinical_action` ∈ `{"urgent_specialist", "soon_specialist",
  "routine_followup", "no_action", "inconclusive_review"}`. The KTD-V10
  override rewrites this to `inconclusive_review` regardless of the
  manifest mapping when quality fails or confidence is low.
- `segmentation` is `null` when `options.return_segmentation` is
  `false`, when the model is classification-only, or when the
  predicted mask covers essentially nothing (server may return `null`
  rather than an all-zero mask).
- `segmentation.bbox` is `[x_min, y_min, x_max, y_max]` in image
  pixels.
- `segmentation.area_ratio` is `(mask pixels) / (total pixels)` ∈
  `[0.0, 1.0]`.
- `saliency_b64` is reserved; servers without saliency support return
  `null`.
- `labels_meta` is carried verbatim from the manifest — the LLM may
  surface `description` when the user asks "what does benign mean
  here?".
- `warnings` — see §8 + §9. Strings, never structured.
- `model_card_url` — optional URL to a published model card.

**Errors:** see §7 for status-code mapping + envelope.

---

## 5. Image payload encoding

```
ImagePayload {
  sha256:   "<64 lowercase hex chars>"
  data_b64: "<standard RFC-4648 base64; no chunking; min length 1>"
}
```

Server processing order:

1. Decode `data_b64` (strict, `validate=True`). Decode failure →
   `400 image_decode_failed`.
2. SHA-256 the decoded bytes. Result must equal `sha256`. Mismatch →
   `400 image_hash_mismatch` with `claimed` and `actual` in
   `details`.
3. Hand the bytes to the adapter for `preprocess()`.

The hash check is **not** a security control — it's a tamper signal.
The attachment pipeline tags `(sha256, modality)` together at ingest
time; allowing a different image to ride that tag would break the
modality hard gate.

## 6. Integrity chain (two-level SHA-256)

The reference loader (`src/claritymed/servers/vision/loader.py`)
enforces a two-level chain. Compatible servers **must** implement an
equivalent check or document explicitly that they don't (in which
case the orchestrator should refuse to load them in
production-equivalent posture).

```
configs/vision.yaml::models[i].manifest_sha256   ←─ root (committed config)
            │
            ▼
       manifest.json    ←─ on-disk per-model contract
            │           (contains sha256_weights pointing at the checkpoint)
            ▼
       weights.pt       ←─ the actual model file
```

1. Server hashes `manifest.json`'s bytes; refuses to start when
   the digest differs from `configs/vision.yaml::models[i].manifest_sha256`.
2. Server parses the manifest; refuses to start when
   `framework` or `accepted_modality` disagrees with the catalog
   entry's claim (drift catch).
3. Server hashes `weights.pt`'s bytes; refuses to start when the
   digest differs from `manifest.sha256_weights`.

Mismatch at any step **aborts startup**. The orchestrator will see
the server as unreachable (no health response) — that's the
intended failure mode. Servers must not start up "best effort" with
the unverified model.

## 7. Error envelope

Every 4xx/5xx response — `HTTPException`, validation errors,
unexpected exceptions — uses this shape:

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

- `code` is the **stable, machine-readable** key. Clients dispatch on
  this; never on `message` text.
- `message` is for logs and operators, not for code paths.
- `request_id` is echoed when the server can recover it from the
  request (header or body); omitted when the request was malformed
  enough that the field couldn't be parsed.
- `details` is an object whose keys depend on `code`. Documented per
  code below.

Standard codes (reference implementation maps them as shown; see
`src/claritymed/servers/vision/app.py::_default_code`):

| Status | `code` | When |
|---|---|---|
| 400 | `bad_request` | Pydantic validation failure not caught by a more specific code |
| 400 | `image_decode_failed` | `data_b64` is not valid base64 |
| 400 | `image_hash_mismatch` | SHA-256 of decoded bytes ≠ `image.sha256` |
| 404 | `unknown_disease` | `disease_id` not in catalog, or catalog entry is disabled |
| 404 | `unknown_model` | `model_id` not in `disease.flow` |
| 413 | `payload_too_large` | Request body exceeds the server's configured ceiling (reference: none in v1; reserved) |
| 422 | `modality_mismatch` | Defence-in-depth catch: image modality ≠ `model.accepted_modality` |
| 500 | `inference_failed` | Adapter raised. Body's `message` carries `f"{type(exc).__name__}: {exc!s}"` |
| 503 | `service_unavailable` | Lifespan didn't finish loading (`status=loading` on `/health`) or a configured model isn't loaded on this server |

Any `code` not in this table is implementation-specific and should be
documented by the server that emits it.

## 8. Capability negotiation

Models declare capabilities in their manifest:

- `supports_saliency: bool`
- `supports_tta: bool`

When a client sets `options.return_saliency: true` against a model
with `supports_saliency: false`, the server **does not** 4xx. It runs
the request normally, leaves `saliency_b64: null`, and adds a single
entry to `warnings[]`:

```
"options.return_saliency=true but model.supports_saliency=false — no-op; saliency_b64 stays null"
```

Same rule for `options.tta` vs `supports_tta`. Rationale: capability
negotiation is an **optional** request hint. A strict 4xx would make
client code brittle for an optional feature.

`options.return_segmentation` follows the same shape: when the model's
`task` is `"classification"` only, the response's `segmentation`
field is `null`. No warning, because the catalog declares the
task — the client already knows.

## 9. KTD-V10 — Clinical action override

After the model emits its classification and the manifest's mapping
produces a `clinical_action`, the server applies one final override:

```
if not input_quality.passed:
    clinical_action = "inconclusive_review"
    warnings.append("quality_gate.passed=False (failed: …) — clinical_action overridden to inconclusive_review (KTD-V10)")
elif classification.confidence_tier == "low":
    clinical_action = "inconclusive_review"
    warnings.append("classification.confidence_tier=low (top1_prob=…) — clinical_action overridden to inconclusive_review (KTD-V10)")
```

- The quality-gate check fires **before** the confidence check —
  "retake the photo" is more actionable than "the model isn't sure".
- The override applies to **every** model, cancer-class or not.
- `cancer_status` is **not** overridden; only `clinical_action`.

The LLM-side reply prompt branches on `clinical_action ==
"inconclusive_review"` to compose the right user-facing message
("the model couldn't read this clearly, here's what to try").

## 10. Implementing a compatible server — checklist

A new server (any language, any framework) is compatible if it:

- [ ] Binds loopback by default and documents how to relax (§1).
- [ ] Implements `GET /health` with the response shape in §4.1.
- [ ] Implements `GET /v1/catalog` with the response shape in §4.2.
- [ ] Implements `POST /v1/detect` with the request + response shape in §4.3.
- [ ] Echoes `X-Request-ID` on every response (§3).
- [ ] Enforces the image hash check (§5).
- [ ] Enforces the two-level SHA-256 integrity chain at boot (§6) —
      or explicitly documents the divergence.
- [ ] Returns the error envelope shape in §7 with the stable `code`
      vocabulary.
- [ ] Implements capability negotiation as silent no-op + warning (§8).
- [ ] Implements the KTD-V10 override (§9).
- [ ] Passes the conformance test suite in
      [`tests/servers/vision/`](../tests/servers/vision/). The
      reference client (`core/vision/client.py`) can be pointed at any
      `base_url` via `configs/vision.yaml::servers[*].base_url` —
      pass the suite with that swap and the orchestrator will treat
      the new server as a drop-in.

## 11. What this protocol intentionally leaves out

- **Batching.** One image per request, full stop. The latency budget
  in `configs/vision.yaml::tool.total_budget_ms` is per-call;
  batching mid-tool is out of scope.
- **Streaming.** No `Transfer-Encoding: chunked` semantics in v1.
- **Async / job queues.** Detect is synchronous; the client blocks.
- **Model hot-swap.** Lifespan loads every enabled model once at
  boot. Adding or replacing a model is a server restart.
- **Cross-server routing.** The catalog endpoint is a per-server
  view, not a fleet view. Orchestrator-side routing across multiple
  servers is the client's job.
- **Authentication beyond loopback.** v2 may add bearer-token
  support; v1 has none.

## 12. Change log

- **v1 (this document).** Initial spec. Three endpoints. KTD-V10
  override applied server-side. Two-level SHA-256 integrity chain.
  Capability negotiation as silent no-op.
