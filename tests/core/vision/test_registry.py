"""VisionRegistry — boot-time catalog cross-check + routing (Unit 5).

The registry is a sync routing surface with one async hook
(``bootstrap``) that talks to each configured server's ``/v1/catalog``.
Tests inject a ``MockTransport`` so the cross-check runs without an
open socket.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from claritymed.core.vision.client import VisionHttpClient
from claritymed.core.vision.registry import VisionRegistry
from claritymed.core.vision.schemas import (
    DiseaseSpec,
    ModelSpec,
    ServerSpec,
    ToolConfig,
    VisionConfig,
    OcrReportConfig,
)
from claritymed.errors import (
    UnknownDiseaseError,
    VisionCatalogMismatchError,
    VisionServerUnreachableError,
)


_HEX_A = "a" * 64
_HEX_B = "b" * 64


# --- fixture builders ----------------------------------------------------


def _make_config(
    *,
    server_id: str = "local_default",
    model_id: str = "breast_busi_unet_v1",
    second_model_id: str | None = None,
    second_server_id: str | None = None,
    manifest_sha: str = _HEX_A,
    enabled: bool = True,
) -> VisionConfig:
    """Build a VisionConfig matching the production layout."""
    models = [
        ModelSpec(
            id=model_id,
            disease_id="breast_cancer_ultrasound",
            server_id=server_id,
            framework="pytorch",
            accepted_modality="ultrasound",
            weights_subpath=f"vision/breast_cancer_ultrasound/{model_id}",
            manifest_sha256=manifest_sha,
            expected_ms=800,
        )
    ]
    # `flow` is fallbacks-only — the primary model is auto-prepended via
    # DiseaseSpec.effective_flow. The first ModelSpec stays the primary;
    # the optional second one joins as the lone fallback.
    flow: list[str] = []
    if second_model_id:
        models.append(
            ModelSpec(
                id=second_model_id,
                disease_id="breast_cancer_ultrasound",
                server_id=second_server_id or server_id,
                framework="pytorch",
                accepted_modality="ultrasound",
                weights_subpath=f"vision/breast_cancer_ultrasound/{second_model_id}",
                manifest_sha256=_HEX_B,
                expected_ms=400,
            )
        )
        flow.append(second_model_id)
    servers = [
        ServerSpec(id=server_id, base_url="http://127.0.0.1:8085", expected_ms=800)
    ]
    if second_server_id and second_server_id != server_id:
        servers.append(
            ServerSpec(
                id=second_server_id, base_url="http://127.0.0.1:8086", expected_ms=300
            )
        )
    diseases = [
        DiseaseSpec(
            id="breast_cancer_ultrasound",
            enabled=enabled,
            primary_model_id=model_id,
            flow=flow,
            cancer_class=True,
            intent_hints_i18n_key="vision.intent.breast_cancer_ultrasound",
        )
    ]
    return VisionConfig(
        diseases=diseases,
        servers=servers,
        models=models,
        tool=ToolConfig(),
        ocr_report=OcrReportConfig(
            min_chars=200,
            markers={"en": ["findings"], "zh": ["所见"]},
        ),
    )


def _catalog_payload(*, model_id: str, manifest_sha: str) -> dict[str, Any]:
    return {
        "models": [
            {
                "disease_id": "breast_cancer_ultrasound",
                "model_id": model_id,
                "model_version": "v1.0.0",
                "framework": "pytorch",
                "task": "classification+segmentation",
                "labels": ["benign", "malignant", "normal"],
                "cancer_class": True,
                "accepted_modality": "ultrasound",
                "manifest_sha": manifest_sha,
                "expected_ms": 800,
                "supports_saliency": False,
                "supports_tta": False,
            }
        ]
    }


def _client_with_catalog(payload: dict[str, Any]) -> VisionHttpClient:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/catalog"
        return httpx.Response(200, json=payload)

    return VisionHttpClient(
        "http://127.0.0.1:8085", transport=httpx.MockTransport(handler)
    )


# --- bootstrap + cross-check --------------------------------------------


async def test_bootstrap_succeeds_when_catalog_matches_config() -> None:
    cfg = _make_config(manifest_sha=_HEX_A)
    client = _client_with_catalog(
        _catalog_payload(model_id="breast_busi_unet_v1", manifest_sha=_HEX_A)
    )
    registry = VisionRegistry(cfg, clients={"local_default": client})
    await registry.bootstrap()
    # Idempotent — second call should be a no-op (we re-use the same
    # MockTransport which would assert on the URL again).
    await registry.bootstrap()
    await registry.aclose()


async def test_bootstrap_fails_on_manifest_sha_drift() -> None:
    cfg = _make_config(manifest_sha=_HEX_A)
    # Server advertises a different sha than the config pins.
    client = _client_with_catalog(
        _catalog_payload(model_id="breast_busi_unet_v1", manifest_sha=_HEX_B)
    )
    registry = VisionRegistry(cfg, clients={"local_default": client})
    with pytest.raises(VisionCatalogMismatchError) as ei:
        await registry.bootstrap()
    assert "manifest sha drift" in str(ei.value)
    await registry.aclose()


async def test_bootstrap_fails_when_server_advertises_unknown_model() -> None:
    cfg = _make_config()
    client = _client_with_catalog(
        _catalog_payload(model_id="surprise_model_v1", manifest_sha=_HEX_A)
    )
    registry = VisionRegistry(cfg, clients={"local_default": client})
    with pytest.raises(VisionCatalogMismatchError) as ei:
        await registry.bootstrap()
    assert "surprise_model_v1" in str(ei.value)
    await registry.aclose()


async def test_bootstrap_fails_when_server_missing_a_configured_model() -> None:
    cfg = _make_config()

    # Server returns empty catalog.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": []})

    client = VisionHttpClient(
        "http://127.0.0.1:8085", transport=httpx.MockTransport(handler)
    )
    registry = VisionRegistry(cfg, clients={"local_default": client})
    with pytest.raises(VisionCatalogMismatchError) as ei:
        await registry.bootstrap()
    assert "breast_busi_unet_v1" in str(ei.value)
    await registry.aclose()


async def test_bootstrap_ignores_disabled_disease_and_flow_omitted_models() -> None:
    """Models present in config but outside any enabled flow are
    intentionally not loaded by the server (disabled-disease scaffolds,
    future fallback placeholders). The registry tolerates the gap so
    ``configs/vision.yaml`` can carry "ready to flip on" entries.
    """
    cfg = _make_config(manifest_sha=_HEX_A)
    # Append a disabled-disease scaffold + a flow-omitted fallback model;
    # neither should appear in the served set the catalog is checked
    # against.
    cfg = cfg.model_copy(
        update={
            "diseases": [
                *cfg.diseases,
                DiseaseSpec(
                    id="skin_cancer_dermoscopy",
                    enabled=False,
                    primary_model_id="skin_isic_resnet50_v1",
                    # Fallbacks-only; primary auto-prepended via effective_flow.
                    flow=[],
                    cancer_class=True,
                    intent_hints_i18n_key="vision.intent.skin_cancer_dermoscopy",
                ),
            ],
            "models": [
                *cfg.models,
                ModelSpec(
                    id="skin_isic_resnet50_v1",
                    disease_id="skin_cancer_dermoscopy",
                    server_id="local_default",
                    framework="pytorch",
                    accepted_modality="dermoscopy",
                    weights_subpath="vision/skin_cancer_dermoscopy/placeholder",
                    manifest_sha256="0" * 64,
                    expected_ms=600,
                ),
                ModelSpec(
                    id="breast_us_kaggle_resnet50_v1",
                    disease_id="breast_cancer_ultrasound",
                    server_id="local_default",
                    framework="pytorch",
                    accepted_modality="ultrasound",
                    weights_subpath="vision/breast_cancer_ultrasound/placeholder",
                    manifest_sha256="0" * 64,
                    expected_ms=700,
                ),
            ],
        }
    )
    client = _client_with_catalog(
        _catalog_payload(model_id="breast_busi_unet_v1", manifest_sha=_HEX_A)
    )
    registry = VisionRegistry(cfg, clients={"local_default": client})
    await registry.bootstrap()
    await registry.aclose()


async def test_bootstrap_propagates_unreachable() -> None:
    cfg = _make_config()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("server down")

    client = VisionHttpClient(
        "http://127.0.0.1:8085", transport=httpx.MockTransport(handler)
    )
    registry = VisionRegistry(cfg, clients={"local_default": client})
    with pytest.raises(VisionServerUnreachableError):
        await registry.bootstrap()
    await registry.aclose()


# --- routing -------------------------------------------------------------


def test_route_falls_back_to_primary_when_no_hint() -> None:
    cfg = _make_config()
    registry = VisionRegistry(
        cfg,
        clients={
            "local_default": _client_with_catalog(
                _catalog_payload(model_id="breast_busi_unet_v1", manifest_sha=_HEX_A)
            )
        },
    )
    server, model = registry.route("breast_cancer_ultrasound")
    assert server.id == "local_default"
    assert model.id == "breast_busi_unet_v1"


def test_route_honors_hint_when_in_flow() -> None:
    cfg = _make_config(
        second_model_id="breast_busi_unet_v2",
        second_server_id="local_default",
    )
    registry = VisionRegistry(
        cfg,
        clients={
            "local_default": _client_with_catalog(
                _catalog_payload(model_id="breast_busi_unet_v1", manifest_sha=_HEX_A)
            )
        },
    )
    server, model = registry.route(
        "breast_cancer_ultrasound", model_id_hint="breast_busi_unet_v2"
    )
    assert model.id == "breast_busi_unet_v2"


def test_route_falls_back_to_primary_when_hint_not_in_flow() -> None:
    """An out-of-flow hint is a soft drop-through — LLM may pass speculative ids."""
    cfg = _make_config()
    registry = VisionRegistry(
        cfg,
        clients={
            "local_default": _client_with_catalog(
                _catalog_payload(model_id="breast_busi_unet_v1", manifest_sha=_HEX_A)
            )
        },
    )
    server, model = registry.route(
        "breast_cancer_ultrasound", model_id_hint="nonexistent_v9"
    )
    assert model.id == "breast_busi_unet_v1"


def test_route_unknown_disease_raises_with_available_list() -> None:
    cfg = _make_config(enabled=True)
    registry = VisionRegistry(
        cfg,
        clients={
            "local_default": _client_with_catalog(
                _catalog_payload(model_id="breast_busi_unet_v1", manifest_sha=_HEX_A)
            )
        },
    )
    with pytest.raises(UnknownDiseaseError) as ei:
        registry.route("not_a_real_disease")
    assert ei.value.disease_id == "not_a_real_disease"
    assert "breast_cancer_ultrasound" in ei.value.available


def test_route_treats_disabled_disease_as_unknown() -> None:
    """A disabled disease is invisible to routing — same as if it didn't exist."""
    cfg = _make_config(enabled=False)
    registry = VisionRegistry(
        cfg,
        clients={
            "local_default": _client_with_catalog(
                _catalog_payload(model_id="breast_busi_unet_v1", manifest_sha=_HEX_A)
            )
        },
    )
    with pytest.raises(UnknownDiseaseError):
        registry.route("breast_cancer_ultrasound")


def test_enabled_disease_ids_filters_disabled() -> None:
    cfg = _make_config(enabled=False)
    registry = VisionRegistry(
        cfg,
        clients={
            "local_default": _client_with_catalog(
                _catalog_payload(model_id="breast_busi_unet_v1", manifest_sha=_HEX_A)
            )
        },
    )
    assert registry.enabled_disease_ids() == []


def test_client_for_returns_cached_instance() -> None:
    cfg = _make_config()
    client = _client_with_catalog(
        _catalog_payload(model_id="breast_busi_unet_v1", manifest_sha=_HEX_A)
    )
    registry = VisionRegistry(cfg, clients={"local_default": client})
    server = cfg.servers[0]
    assert registry.client_for(server) is client
