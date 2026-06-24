"""Tests for ``/api/v1/admin/rag``."""

from __future__ import annotations

import pytest
import yaml

from claritymed import config as _cfg


@pytest.fixture
def tmp_retrieval(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "retrieval.yaml").write_text(
        yaml.safe_dump(
            {
                "system_rag": {
                    "collections": [
                        {
                            "name": "statpearls",
                            "language": "en",
                            "authority_tier": 1,
                            "topics": ["clinical"],
                        },
                        {
                            "name": "textbooks",
                            "language": "en",
                            "authority_tier": 2,
                            "topics": ["pediatrics"],
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(_cfg, "CONFIGS_DIR", cfg_dir)
    _cfg.load_yaml.cache_clear()
    yield cfg_dir
    _cfg.load_yaml.cache_clear()


@pytest.mark.asyncio
async def test_rag_collections_requires_admin(web_client, non_admin_cookies):
    response = await web_client.get(
        "/api/v1/admin/rag/collections", cookies=non_admin_cookies
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_list_rag_collections(web_client, admin_cookies, tmp_retrieval):
    response = await web_client.get(
        "/api/v1/admin/rag/collections", cookies=admin_cookies
    )
    assert response.status_code == 200
    body = response.json()
    names = [c["name"] for c in body["items"]]
    assert "statpearls" in names
    assert "textbooks" in names


@pytest.mark.asyncio
async def test_inspect_collection_404_on_unknown(
    web_client, admin_cookies, tmp_retrieval
):
    response = await web_client.get(
        "/api/v1/admin/rag/collections/not_declared", cookies=admin_cookies
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_inspect_declared_collection(web_client, admin_cookies, tmp_retrieval):
    response = await web_client.get(
        "/api/v1/admin/rag/collections/statpearls", cookies=admin_cookies
    )
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "statpearls"
    assert body["metadata"]["authority_tier"] == 1


@pytest.mark.asyncio
async def test_bootstrap_creates_job(web_client, admin_cookies, tmp_retrieval):
    response = await web_client.post(
        "/api/v1/admin/rag/bootstrap",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        json={"skip_existing": True},
    )
    assert response.status_code == 200
    spec = response.json()
    assert spec["kind"] == "rag_bootstrap"
    assert spec["state"] in ("queued", "running", "done", "failed")


@pytest.mark.asyncio
async def test_ingest_validates_required_fields(web_client, admin_cookies):
    response = await web_client.post(
        "/api/v1/admin/rag/ingest",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        json={},
    )
    assert response.status_code == 422
