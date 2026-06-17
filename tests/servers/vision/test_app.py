"""FastAPI app — /health, /v1/catalog, /v1/detect (Unit 4).

Uses ``fastapi.TestClient`` so the app runs in-process — no socket bind,
no subprocess. ``CLARITYMED_VISION_SKIP_LOAD=1`` bypasses the lifespan
model load; each test stages ``app._state["resources"]`` and
``app._state["diseases"]`` directly via the conftest fixture.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any

import pytest

from claritymed.core.vision.schemas import DiseaseSpec
from claritymed.servers.vision import app as app_mod
from claritymed.servers.vision.app import HOST, app
from claritymed.servers.vision.inference import InferenceResources
from claritymed.servers.vision.loader import verify_manifest_chain


# --- fixtures ------------------------------------------------------------


@pytest.fixture(autouse=True)
def _skip_load(monkeypatch: pytest.MonkeyPatch):
    """Bypass lifespan model loading; isolate module state per-test."""
    monkeypatch.setenv("CLARITYMED_VISION_SKIP_LOAD", "1")
    app_mod._state["resources"] = {}
    app_mod._state["diseases"] = {}
    app_mod._state["started_at"] = 0.0
    app_mod._state["config_loaded"] = False
    yield
    app_mod._state["resources"] = {}
    app_mod._state["diseases"] = {}


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


@pytest.fixture
def stage_resources(make_vision_artifact, vision_models_root: Path, monkeypatch):
    """Stage a (DiseaseSpec, InferenceResources) pair into module state.

    Patches ``load_vision_config`` so the catalog endpoint's
    ``_spec_for_resources`` lookup succeeds without a real configs/ file.
    Returns the spec for assertions.
    """

    def _stage(**artifact_kwargs):
        spec, manifest_path, weights_path = make_vision_artifact(**artifact_kwargs)
        manifest = verify_manifest_chain(spec, manifest_path.parent)

        from claritymed.servers.vision.adapters.torch_adapter import TorchAdapter

        adapter = TorchAdapter(
            spec=spec, manifest=manifest, weights_path=weights_path, device="cpu"
        )
        disease = DiseaseSpec(
            id=spec.disease_id,
            enabled=True,
            primary_model_id=spec.id,
            # flow is fallbacks-only; primary auto-prepended via effective_flow
            flow=[],
            cancer_class=manifest.cancer_class,
            intent_hints_i18n_key=f"vision.intent.{spec.disease_id}",
        )
        app_mod._state["diseases"][disease.id] = disease
        app_mod._state["resources"][spec.id] = InferenceResources(
            spec_id=spec.id,
            disease_id=disease.id,
            model=adapter,
            manifest=manifest,
        )
        app_mod._state["config_loaded"] = True

        # _spec_for_resources re-reads configs/vision.yaml; patch it to
        # return a minimal config containing this spec so the catalog
        # endpoint doesn't trip on the real (file-system) config.
        class _FakeConfig:
            models = [spec]

        monkeypatch.setattr(app_mod, "load_vision_config", lambda: _FakeConfig())
        return spec, disease, manifest

    return _stage


def _detect_payload(
    *,
    image_bytes: bytes,
    disease_id: str,
    model_id: str | None = None,
    request_id: str = "req_42",
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sha = hashlib.sha256(image_bytes).hexdigest()
    body: dict[str, Any] = {
        "request_id": request_id,
        "disease_id": disease_id,
        "image": {
            "sha256": sha,
            "data_b64": base64.b64encode(image_bytes).decode("ascii"),
        },
        "language": "en",
        "options": options or {},
    }
    if model_id is not None:
        body["model_id"] = model_id
    return body


def _png_bytes() -> bytes:
    """64×64 white PNG — passes the Torch adapter's min_resolution gate."""
    from PIL import Image
    import io

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color="white").save(buf, format="PNG")
    return buf.getvalue()


# --- loopback assertion --------------------------------------------------


def test_host_constant_is_loopback() -> None:
    """KTD-V8 — vision server must refuse non-loopback binds at constant level."""
    assert HOST == "127.0.0.1"


# --- /health -------------------------------------------------------------


def test_health_loading_when_lifespan_skipped(client) -> None:
    # The SKIP_LOAD lifespan still flips config_loaded=True so the
    # FastAPI app reports ready. To exercise the "loading" branch we
    # reset the flag *after* TestClient's lifespan ran.
    app_mod._state["config_loaded"] = False
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "loading"
    assert body["models_loaded"] == []


def test_health_ok_with_loaded_model(client, stage_resources) -> None:
    stage_resources()
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] == [
        {
            "disease_id": "breast_cancer_ultrasound",
            "model_id": "breast_busi_unet_v1",
        }
    ]


# --- /v1/catalog ---------------------------------------------------------


def test_catalog_returns_full_model_metadata(client, stage_resources) -> None:
    spec, _, manifest = stage_resources()
    resp = client.get("/v1/catalog")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["models"]) == 1
    entry = body["models"][0]
    assert entry["disease_id"] == spec.disease_id
    assert entry["model_id"] == spec.id
    assert entry["model_version"] == manifest.model_version
    assert entry["framework"] == manifest.framework
    assert entry["task"] == manifest.task
    assert entry["labels"] == list(manifest.labels)
    assert entry["cancer_class"] is True
    assert entry["accepted_modality"] == "ultrasound"
    assert entry["manifest_sha"] == spec.manifest_sha256
    assert entry["supports_saliency"] is False
    assert entry["supports_tta"] is False


def test_catalog_is_empty_when_no_models_loaded(client) -> None:
    resp = client.get("/v1/catalog")
    assert resp.status_code == 200
    assert resp.json() == {"models": []}


# --- /v1/detect happy path ------------------------------------------------


def test_detect_returns_raw_detection_with_echoed_request_id(
    client, stage_resources
) -> None:
    stage_resources()
    image_bytes = _png_bytes()
    resp = client.post(
        "/v1/detect",
        json=_detect_payload(
            image_bytes=image_bytes,
            disease_id="breast_cancer_ultrasound",
        ),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["request_id"] == "req_42"
    # X-Request-ID echoed in the header per plan §"Test scenarios".
    assert resp.headers.get("x-request-id") == "req_42"
    # Wire shape matches RawDetection — spot-check the load-bearing keys.
    assert "classification" in body
    assert "clinical_action" in body
    assert "labels_meta" in body
    assert "model_version" in body


def test_detect_omits_model_id_falls_back_to_primary(client, stage_resources) -> None:
    spec, _, _ = stage_resources()
    image_bytes = _png_bytes()
    payload = _detect_payload(
        image_bytes=image_bytes, disease_id=spec.disease_id, model_id=None
    )
    resp = client.post("/v1/detect", json=payload)
    assert resp.status_code == 200
    assert resp.json()["model_id"] == spec.id


# --- /v1/detect error paths ----------------------------------------------


def test_detect_503_when_lifespan_did_not_complete(client) -> None:
    # Same trick as the /health test — flip back to False after the
    # TestClient ran the SKIP_LOAD lifespan, to exercise the not-ready
    # branch of /v1/detect.
    app_mod._state["config_loaded"] = False
    image_bytes = _png_bytes()
    resp = client.post(
        "/v1/detect",
        json=_detect_payload(
            image_bytes=image_bytes,
            disease_id="breast_cancer_ultrasound",
        ),
    )
    assert resp.status_code == 503
    err = resp.json()["error"]
    assert err["code"] == "service_unavailable"


def test_detect_404_for_unknown_disease(client, stage_resources) -> None:
    stage_resources()
    image_bytes = _png_bytes()
    resp = client.post(
        "/v1/detect",
        json=_detect_payload(image_bytes=image_bytes, disease_id="nope"),
    )
    assert resp.status_code == 404
    err = resp.json()["error"]
    assert err["code"] == "unknown_disease"
    assert "available" in err["details"]


def test_detect_404_for_disabled_disease(client, stage_resources) -> None:
    spec, disease, _ = stage_resources()
    # Manually disable so the 404 branch fires.
    app_mod._state["diseases"][disease.id] = disease.model_copy(
        update={"enabled": False}
    )
    image_bytes = _png_bytes()
    resp = client.post(
        "/v1/detect",
        json=_detect_payload(image_bytes=image_bytes, disease_id=disease.id),
    )
    assert resp.status_code == 404
    err = resp.json()["error"]
    assert err["code"] == "unknown_disease"


def test_detect_404_for_unknown_model_id_under_known_disease(
    client, stage_resources
) -> None:
    stage_resources()
    image_bytes = _png_bytes()
    resp = client.post(
        "/v1/detect",
        json=_detect_payload(
            image_bytes=image_bytes,
            disease_id="breast_cancer_ultrasound",
            model_id="no_such_model_v9",
        ),
    )
    assert resp.status_code == 404
    err = resp.json()["error"]
    assert err["code"] == "unknown_model"
    assert "breast_busi_unet_v1" in err["details"]["available"]


def test_detect_400_for_image_hash_mismatch(client, stage_resources) -> None:
    stage_resources()
    image_bytes = _png_bytes()
    payload = _detect_payload(
        image_bytes=image_bytes, disease_id="breast_cancer_ultrasound"
    )
    # Substitute a wrong sha — the server re-hashes and refuses.
    payload["image"]["sha256"] = "0" * 64
    resp = client.post("/v1/detect", json=payload)
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["code"] == "image_hash_mismatch"


def test_detect_400_for_corrupted_base64(client, stage_resources) -> None:
    stage_resources()
    payload = {
        "request_id": "req_42",
        "disease_id": "breast_cancer_ultrasound",
        "image": {
            "sha256": "0" * 64,
            "data_b64": "not-base64-!!!",
        },
        "language": "en",
    }
    resp = client.post("/v1/detect", json=payload)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "image_decode_failed"


def test_detect_500_when_adapter_raises(client, stage_resources) -> None:
    spec, disease, manifest = stage_resources()

    # Swap the adapter for one that raises in predict — exercises the
    # generic 500 envelope.
    class _BoomAdapter:
        spec = None

        def preprocess(self, image_bytes):
            from PIL import Image
            import io

            return Image.open(io.BytesIO(image_bytes)).convert("RGB")

        def predict(self, x):
            raise RuntimeError("synthetic boom")

        def calibrate(self, raw):
            return {}

        def segment(self, x):
            return None

        def quality_gate(self, image):
            from claritymed.core.vision.schemas import InputQuality, QualityCheck

            return InputQuality(
                passed=True,
                checks=[QualityCheck(name="min_resolution", score=64, passed=True)],
            )

    app_mod._state["resources"][spec.id] = InferenceResources(
        spec_id=spec.id, disease_id=disease.id, model=_BoomAdapter(), manifest=manifest
    )
    image_bytes = _png_bytes()
    resp = client.post(
        "/v1/detect",
        json=_detect_payload(
            image_bytes=image_bytes, disease_id="breast_cancer_ultrasound"
        ),
    )
    assert resp.status_code == 500
    err = resp.json()["error"]
    assert err["code"] == "inference_failed"
    assert "synthetic boom" in err["message"]
