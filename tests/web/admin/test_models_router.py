"""Tests for ``/api/v1/admin/models``."""

from __future__ import annotations

import os
import stat

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
async def test_catalogs_requires_admin(web_client, non_admin_cookies):
    response = await web_client.get(
        "/api/v1/admin/models/catalogs", cookies=non_admin_cookies
    )
    assert response.status_code == 403


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
