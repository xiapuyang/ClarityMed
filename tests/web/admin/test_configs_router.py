"""Tests for ``/api/v1/admin/configs``."""

from __future__ import annotations

import pytest
import yaml

from claritymed import config as _cfg


@pytest.fixture
def tmp_configs_dir(tmp_path, monkeypatch):
    """Redirect CONFIGS_DIR so test writes don't stomp the real repo."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "app.yaml").write_text(
        yaml.safe_dump(
            {
                "i18n": {"default_lang": "en"},
                "tracing": {"enabled": False},
                "upload": {"max_text_chars": 5000},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (cfg_dir / "retrieval.yaml").write_text(
        yaml.safe_dump({"top_k_default": 5, "rerank": {"enabled": True}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(_cfg, "CONFIGS_DIR", cfg_dir)
    _cfg.load_yaml.cache_clear()
    yield cfg_dir
    _cfg.load_yaml.cache_clear()


@pytest.mark.asyncio
async def test_list_configs_accessible_to_any_user(web_client, non_admin_cookies):
    response = await web_client.get("/api/v1/admin/configs", cookies=non_admin_cookies)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_list_configs_returns_catalog(web_client, admin_cookies):
    response = await web_client.get("/api/v1/admin/configs", cookies=admin_cookies)
    body = response.json()
    assert "app.yaml" in body["configs"]
    assert "i18n.default_lang" in body["editable_keys"]["app.yaml"]


@pytest.mark.asyncio
async def test_read_config(web_client, admin_cookies, tmp_configs_dir):
    response = await web_client.get(
        "/api/v1/admin/configs/app.yaml", cookies=admin_cookies
    )
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["i18n"]["default_lang"] == "en"


@pytest.mark.asyncio
async def test_read_unknown_config_rejected(web_client, admin_cookies):
    response = await web_client.get(
        "/api/v1/admin/configs/not_allowed.yaml", cookies=admin_cookies
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_patch_known_key(web_client, admin_cookies, tmp_configs_dir):
    response = await web_client.patch(
        "/api/v1/admin/configs/app.yaml",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        json={"path": "i18n.default_lang", "value": "zh"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["i18n"]["default_lang"] == "zh"
    # Verify on disk.
    on_disk = yaml.safe_load((tmp_configs_dir / "app.yaml").read_text())
    assert on_disk["i18n"]["default_lang"] == "zh"


@pytest.mark.asyncio
async def test_patch_unknown_key_rejected(web_client, admin_cookies, tmp_configs_dir):
    response = await web_client.patch(
        "/api/v1/admin/configs/app.yaml",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        json={"path": "secrets.api_key", "value": "leaked"},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_patch_unknown_config_rejected(web_client, admin_cookies):
    response = await web_client.patch(
        "/api/v1/admin/configs/passwd",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        json={"path": "x", "value": "y"},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_patch_boolean_value(web_client, admin_cookies, tmp_configs_dir):
    response = await web_client.patch(
        "/api/v1/admin/configs/app.yaml",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        json={"path": "tracing.enabled", "value": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["tracing"]["enabled"] is True


@pytest.mark.asyncio
async def test_read_config_missing_file_returns_404(
    web_client, admin_cookies, tmp_configs_dir, monkeypatch
):
    """When load_yaml raises FileNotFoundError the router returns 404."""

    def _raise(name):
        raise FileNotFoundError(f"{name} not found")

    _raise.cache_clear = lambda: None  # satisfy teardown's cache_clear call
    monkeypatch.setattr(_cfg, "load_yaml", _raise)
    response = await web_client.get(
        "/api/v1/admin/configs/safety.yaml", cookies=admin_cookies
    )
    assert response.status_code == 404
    assert "safety.yaml" in response.json()["detail"]


@pytest.mark.asyncio
async def test_patch_config_missing_file_returns_404(
    web_client, admin_cookies, tmp_configs_dir, monkeypatch
):
    """When load_yaml raises FileNotFoundError on PATCH the router returns 404."""

    def _raise(name):
        raise FileNotFoundError(f"{name} not found")

    _raise.cache_clear = lambda: None
    monkeypatch.setattr(_cfg, "load_yaml", _raise)
    response = await web_client.patch(
        "/api/v1/admin/configs/safety.yaml",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        json={"path": "phi.privacy_filter.enabled", "value": True},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_patch_nested_key_creates_intermediate_dict(
    web_client, admin_cookies, tmp_configs_dir
):
    """Patching a key whose parent dict doesn't exist creates it on write."""
    response = await web_client.patch(
        "/api/v1/admin/configs/app.yaml",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        json={"path": "paste.max_text_chars", "value": 2000},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["paste"]["max_text_chars"] == 2000
    on_disk = yaml.safe_load((tmp_configs_dir / "app.yaml").read_text())
    assert on_disk["paste"]["max_text_chars"] == 2000


@pytest.mark.asyncio
async def test_patch_config_save_yaml_error_returns_400(
    web_client, admin_cookies, tmp_configs_dir, monkeypatch
):
    """When save_yaml raises ValueError the router surfaces it as a 400."""
    from claritymed.web.routers.admin import configs as configs_router

    def _raise(*_a, **_kw):
        raise ValueError("disk allowlist")

    monkeypatch.setattr(configs_router, "save_yaml", _raise)
    response = await web_client.patch(
        "/api/v1/admin/configs/app.yaml",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        json={"path": "i18n.default_lang", "value": "zh"},
    )
    assert response.status_code == 400
    assert "disk allowlist" in response.json()["detail"]
