"""Tests for ``/api/v1/admin/models``."""

from __future__ import annotations

import os
import stat
from unittest.mock import patch

import pytest
import yaml

from claritymed import config as _cfg


@pytest.fixture
def tmp_catalogs(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "models.yaml").write_text(
        yaml.safe_dump(
            {
                "default_provider": "openai-gpt-4o",
                "providers": [
                    {
                        "id": "openai-gpt-4o",
                        "kind": "cloud",
                        "model": "openai:gpt-4o",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (cfg_dir / "vision.yaml").write_text(
        yaml.safe_dump(
            {
                "enabled": True,
                "diseases": [
                    {
                        "id": "pneumonia",
                        "primary_model_id": "torch:v1",
                        "flow": ["torch:v1", "torch:v0"],
                        "enabled": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(_cfg, "CONFIGS_DIR", cfg_dir)
    _cfg.load_yaml.cache_clear()
    yield cfg_dir
    _cfg.load_yaml.cache_clear()


@pytest.fixture
def tmp_env_home(monkeypatch):
    """Use the current CLARITYMED_HOME (already redirected by parent conftest).

    Reloading the config module would invalidate the test_user fixture's
    on-disk state (which lives under the same home), so we just operate
    on whatever path _cfg.CLARITYMED_HOME already points at. We monkey-
    delete the manifest's keys from os.environ so the masked_view test
    sees a clean missing state.
    """
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    # Drop any lingering .env from a prior test in the same home.
    env_path = _cfg.CLARITYMED_HOME / ".env"
    if env_path.exists():
        env_path.unlink()
    yield _cfg.CLARITYMED_HOME
    if env_path.exists():
        env_path.unlink()


@pytest.mark.asyncio
async def test_catalogs_accessible_to_any_user(web_client, non_admin_cookies):
    response = await web_client.get(
        "/api/v1/admin/models/catalogs", cookies=non_admin_cookies
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_list_catalogs(web_client, admin_cookies):
    response = await web_client.get(
        "/api/v1/admin/models/catalogs", cookies=admin_cookies
    )
    body = response.json()
    assert "models.yaml" in body["catalogs"]
    assert "vision.yaml" in body["catalogs"]


@pytest.mark.asyncio
async def test_read_catalog(web_client, admin_cookies, tmp_catalogs):
    response = await web_client.get(
        "/api/v1/admin/models/catalogs/models.yaml", cookies=admin_cookies
    )
    body = response.json()
    assert body["data"]["default_provider"] == "openai-gpt-4o"


@pytest.mark.asyncio
async def test_read_unknown_catalog_rejected(web_client, admin_cookies):
    response = await web_client.get(
        "/api/v1/admin/models/catalogs/private.yaml", cookies=admin_cookies
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_patch_catalog_full_overwrite(web_client, admin_cookies, tmp_catalogs):
    next_data = {
        "default_provider": "openai-gpt-4o",
        "providers": [
            {"id": "openai-gpt-4o", "kind": "cloud", "model": "openai:gpt-4o"},
            {
                "id": "anthropic",
                "kind": "cloud",
                "model": "anthropic:claude-sonnet-4-5",
            },
        ],
    }
    response = await web_client.patch(
        "/api/v1/admin/models/catalogs/models.yaml",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        json={"data": next_data},
    )
    assert response.status_code == 200
    on_disk = yaml.safe_load((tmp_catalogs / "models.yaml").read_text())
    assert len(on_disk["providers"]) == 2


@pytest.mark.asyncio
async def test_secrets_returns_manifest_and_status(
    web_client, admin_cookies, tmp_env_home, monkeypatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = await web_client.get(
        "/api/v1/admin/models/secrets", cookies=admin_cookies
    )
    body = response.json()
    keys = {row["key"] for row in body["status"]}
    assert "OPENAI_API_KEY" in keys
    openai_row = next(r for r in body["status"] if r["key"] == "OPENAI_API_KEY")
    assert openai_row["is_set"] is False


@pytest.mark.asyncio
async def test_secrets_patch_writes_file(web_client, admin_cookies, tmp_env_home):
    response = await web_client.patch(
        "/api/v1/admin/models/secrets",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        json={"updates": {"OPENAI_API_KEY": "sk-test-123"}},
    )
    body = response.json()
    assert body["restart_required"] is True
    assert "OPENAI_API_KEY" in body["keys_changed"]
    env_file = tmp_env_home / ".env"
    assert env_file.exists()
    mode = stat.S_IMODE(os.stat(env_file).st_mode)
    assert mode == 0o600


@pytest.mark.asyncio
async def test_secrets_patch_unknown_key_rejected(
    web_client, admin_cookies, tmp_env_home
):
    response = await web_client.patch(
        "/api/v1/admin/models/secrets",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        json={"updates": {"NOT_IN_MANIFEST": "x"}},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_read_catalog_missing_file_returns_404(
    web_client, admin_cookies, tmp_catalogs, monkeypatch
):
    """Reading an editable catalog that isn't on disk → 404."""

    def _raise(name):
        raise FileNotFoundError(f"{name} not found")

    _raise.cache_clear = lambda: None
    monkeypatch.setattr(_cfg, "load_yaml", _raise)
    response = await web_client.get(
        "/api/v1/admin/models/catalogs/models.yaml", cookies=admin_cookies
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_patch_catalog_not_allowlisted_returns_400(web_client, admin_cookies):
    """Patching a catalog not in the editable list → 400."""
    response = await web_client.patch(
        "/api/v1/admin/models/catalogs/private.yaml",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        json={"data": {"key": "value"}},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_patch_catalog_file_missing_succeeds_with_empty_old(
    web_client, admin_cookies, tmp_catalogs
):
    """Patching an editable catalog whose file is absent still succeeds (old={})."""
    (tmp_catalogs / "models.yaml").unlink()
    new_data = {
        "default_provider": "openai-gpt-4o",
        "providers": [
            {"id": "openai-gpt-4o", "kind": "cloud", "model": "openai:gpt-4o"}
        ],
    }
    response = await web_client.patch(
        "/api/v1/admin/models/catalogs/models.yaml",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        json={"data": new_data},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["default_provider"] == "openai-gpt-4o"


@pytest.mark.asyncio
async def test_secrets_patch_empty_updates_returns_no_op(
    web_client, admin_cookies, tmp_env_home
):
    """PATCH /models/secrets with empty updates dict → no keys changed, no restart."""
    response = await web_client.patch(
        "/api/v1/admin/models/secrets",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        json={"updates": {}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["keys_changed"] == []
    assert body["restart_required"] is False


@pytest.mark.asyncio
async def test_patch_vision_catalog_triggers_schema_validation_path(
    web_client, admin_cookies, tmp_catalogs
):
    """Patching vision.yaml exercises the VisionConfig import + validate path.

    Minimal data missing required sub-fields triggers the Exception handler.
    Covers both the VisionConfig import (lines 163-165) and the exception
    re-raise (lines 172-176) in _validate_catalog.
    """
    response = await web_client.patch(
        "/api/v1/admin/models/catalogs/vision.yaml",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        json={"data": {"diseases": [{"id": "x", "primary_model_id": "m", "flow": []}]}},
    )
    assert response.status_code == 422


def test_validate_catalog_non_dict_raises_422():
    """_validate_catalog with a non-dict body raises 422 HTTPException."""
    from fastapi import HTTPException

    from claritymed.web.routers.admin.models import _validate_catalog

    with pytest.raises(HTTPException) as exc_info:
        _validate_catalog("models.yaml", ["not", "a", "dict"])
    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_patch_catalog_load_yaml_raises_file_not_found(
    web_client, admin_cookies, tmp_catalogs, monkeypatch
):
    """When load_yaml raises FileNotFoundError in patch_catalog → old={}, proceeds."""

    def _raise(name):
        raise FileNotFoundError(f"{name} not found")

    _raise.cache_clear = lambda: None
    monkeypatch.setattr(_cfg, "load_yaml", _raise)
    new_data = {
        "default_provider": "openai-gpt-4o",
        "providers": [
            {"id": "openai-gpt-4o", "kind": "cloud", "model": "openai:gpt-4o"}
        ],
    }
    response = await web_client.patch(
        "/api/v1/admin/models/catalogs/models.yaml",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        json={"data": new_data},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["default_provider"] == "openai-gpt-4o"


def test_validate_catalog_import_error_is_noop():
    """_validate_catalog with ImportError for schema module returns silently."""

    from claritymed.web.routers.admin.models import _validate_catalog

    with patch.dict("sys.modules", {"claritymed.core.vision.schemas": None}):
        _validate_catalog("vision.yaml", {"key": "value"})
