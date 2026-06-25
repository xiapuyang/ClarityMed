"""Tests for ``/api/v1/admin/rag``."""

from __future__ import annotations

import json

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
async def test_upsert_rejects_invalid_name(web_client, admin_cookies, tmp_retrieval):
    """``name`` must match the [a-z][a-z0-9_]+ Qdrant collection convention."""
    response = await web_client.post(
        "/api/v1/admin/rag/collections/upsert",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        data={"metadata": json.dumps({"name": "BadName"})},
        files={"files": ("note.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_upsert_rejects_empty_files(web_client, admin_cookies, tmp_retrieval):
    """Multipart with no file parts -> 422 from FastAPI's parser."""
    response = await web_client.post(
        "/api/v1/admin/rag/collections/upsert",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        data={"metadata": json.dumps({"name": "good_name"})},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_upsert_rejects_empty_file_body(web_client, admin_cookies, tmp_retrieval):
    """A file part with zero bytes is rejected at the boundary."""
    response = await web_client.post(
        "/api/v1/admin/rag/collections/upsert",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        data={"metadata": json.dumps({"name": "good_name"})},
        files={"files": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_upsert_rejects_invalid_metadata_json(
    web_client, admin_cookies, tmp_retrieval
):
    """Malformed JSON in the metadata field surfaces a 422 with a clear hint."""
    response = await web_client.post(
        "/api/v1/admin/rag/collections/upsert",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        data={"metadata": "{not-json"},
        files={"files": ("note.txt", b"x", "text/plain")},
    )
    assert response.status_code == 422
    assert "json" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_upsert_requires_admin(web_client, non_admin_cookies, tmp_retrieval):
    response = await web_client.post(
        "/api/v1/admin/rag/collections/upsert",
        cookies=non_admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        data={"metadata": json.dumps({"name": "good_name"})},
        files={"files": ("note.txt", b"x", "text/plain")},
    )
    assert response.status_code == 403
