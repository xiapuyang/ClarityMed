"""Tests for ``/api/v1/admin/rag``."""

from __future__ import annotations

import io
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from claritymed import config as _cfg


# --- _save_uploads unit tests ------------------------------------------


def _make_upload(filename: str, data: bytes):
    """Create a minimal UploadFile-like object for _save_uploads tests."""
    from fastapi import UploadFile

    return UploadFile(filename=filename, file=io.BytesIO(data))


def test_save_uploads_empty_filename_raises_422(tmp_path):
    """An upload part with no filename raises 422."""
    from fastapi import HTTPException

    from claritymed.web.routers.admin.rag import _save_uploads

    upload = _make_upload("", b"data")
    with pytest.raises(HTTPException) as exc_info:
        _save_uploads([upload], tmp_path)
    assert exc_info.value.status_code == 422
    assert "filename" in exc_info.value.detail


def test_save_uploads_duplicate_filenames_are_disambiguated(tmp_path):
    """Two uploads with the same name are stored as file.txt and file-2.txt."""
    from claritymed.web.routers.admin.rag import _save_uploads

    files = [
        _make_upload("dup.txt", b"first"),
        _make_upload("dup.txt", b"second"),
    ]
    saved = _save_uploads(files, tmp_path)
    assert len(saved) == 2
    names = {p.name for p in saved}
    assert "dup.txt" in names
    assert "dup-2.txt" in names


def test_save_uploads_three_way_collision_increments_to_three(tmp_path):
    """Three files with identical names get suffixes: file.txt, file-2.txt, file-3.txt."""
    from claritymed.web.routers.admin.rag import _save_uploads

    files = [
        _make_upload("triple.txt", b"one"),
        _make_upload("triple.txt", b"two"),
        _make_upload("triple.txt", b"three"),
    ]
    saved = _save_uploads(files, tmp_path)
    assert len(saved) == 3
    names = {p.name for p in saved}
    assert names == {"triple.txt", "triple-2.txt", "triple-3.txt"}


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


@pytest.mark.asyncio
async def test_upsert_too_many_files_returns_422(
    web_client, admin_cookies, tmp_retrieval
):
    """More than _MAX_UPLOAD_FILES (100) in a single request → 422."""
    from claritymed.web.routers.admin.rag import _MAX_UPLOAD_FILES

    files = [
        ("files", (f"f{i}.txt", b"x", "text/plain"))
        for i in range(_MAX_UPLOAD_FILES + 1)
    ]
    response = await web_client.post(
        "/api/v1/admin/rag/collections/upsert",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        data={"metadata": json.dumps({"name": "good_name"})},
        files=files,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_upsert_file_too_large_returns_413(
    web_client, admin_cookies, tmp_retrieval
):
    """A single file exceeding 25 MiB is rejected with 413."""
    from claritymed.web.routers.admin.rag import _MAX_UPLOAD_BYTES

    big_content = b"x" * (_MAX_UPLOAD_BYTES + 1)
    response = await web_client.post(
        "/api/v1/admin/rag/collections/upsert",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        data={"metadata": json.dumps({"name": "good_name"})},
        files={"files": ("big.txt", big_content, "text/plain")},
    )
    assert response.status_code == 413
    assert "big.txt" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upsert_503_when_no_runner(web_client, admin_cookies, tmp_retrieval):
    """If the rag_ingest runner is not registered, the endpoint returns 503."""
    app = web_client._transport.app  # type: ignore[attr-defined]
    # Temporarily remove the runner.
    original = app.state.jobs._runners.pop("rag_ingest", None)
    try:
        response = await web_client.post(
            "/api/v1/admin/rag/collections/upsert",
            cookies=admin_cookies,
            headers={"X-CSRF-Token": "csrf-test-token"},
            data={"metadata": json.dumps({"name": "good_name"})},
            files={"files": ("note.txt", b"hello world", "text/plain")},
        )
    finally:
        if original is not None:
            app.state.jobs.register_runner("rag_ingest", original)
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_upsert_success_returns_job_spec(
    web_client, admin_cookies, tmp_retrieval, monkeypatch
):
    """Happy path: returns a job spec immediately; ingest runs in background."""
    from unittest.mock import AsyncMock, patch

    from claritymed.ingest.corpus.base import IngestStats
    from claritymed.ingest.system_rag import SystemRagIngestResult

    fake_result = SystemRagIngestResult(
        stats=IngestStats(
            source="test",
            docs_processed=1,
            parents_written=2,
            children_written=3,
            docs_skipped=0,
        ),
        yaml_snippet="name: good_name\n",
        is_new_collection=False,
        centroid_refreshed=False,
    )

    with patch(
        "claritymed.web.admin.job_runners.rag_ingest.ingest_system_rag",
        new=AsyncMock(return_value=fake_result),
    ):
        response = await web_client.post(
            "/api/v1/admin/rag/collections/upsert",
            cookies=admin_cookies,
            headers={"X-CSRF-Token": "csrf-test-token"},
            data={"metadata": json.dumps({"name": "good_name"})},
            files={"files": ("doc.txt", b"medical content " * 20, "text/plain")},
        )

    assert response.status_code == 200
    body = response.json()
    assert "id" in body
    assert body["kind"] == "rag_ingest"


@pytest.mark.asyncio
async def test_delete_collection_success(web_client, admin_cookies, tmp_retrieval):
    """DELETE on a declared collection calls RagCollectionStore.delete_collection().

    RagCollectionStore is not yet in claritymed.stores.knowledge, so we
    inject it via patch(create=True) to exercise the happy path (204 + audit).
    """
    from unittest.mock import patch

    deleted: dict = {}

    class FakeStore:
        def __init__(self, name):
            self.name = name

        def delete_collection(self):
            deleted["name"] = self.name

    with patch(
        "claritymed.stores.knowledge.RagCollectionStore", FakeStore, create=True
    ):
        response = await web_client.delete(
            "/api/v1/admin/rag/collections/statpearls",
            cookies=admin_cookies,
            headers={"X-CSRF-Token": "csrf-test-token"},
        )
    assert response.status_code == 204
    assert deleted.get("name") == "statpearls"


@pytest.mark.asyncio
async def test_delete_collection_404_on_undeclared(
    web_client, admin_cookies, tmp_retrieval
):
    """DELETE on a collection not in retrieval.yaml returns 404."""
    response = await web_client.delete(
        "/api/v1/admin/rag/collections/does_not_exist",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_collection_501_when_store_has_no_delete(
    web_client, admin_cookies, tmp_retrieval
):
    """If RagCollectionStore has no delete_collection method, endpoint returns 501."""
    from unittest.mock import patch

    class NoDeleteStore:
        def __init__(self, name):
            pass

        # delete_collection is intentionally absent — accessing it raises AttributeError.

    with patch(
        "claritymed.stores.knowledge.RagCollectionStore", NoDeleteStore, create=True
    ):
        response = await web_client.delete(
            "/api/v1/admin/rag/collections/statpearls",
            cookies=admin_cookies,
            headers={"X-CSRF-Token": "csrf-test-token"},
        )
    assert response.status_code == 501
    assert "not implemented" in response.json()["detail"]


@pytest.mark.asyncio
async def test_delete_collection_500_on_store_error(
    web_client, admin_cookies, tmp_retrieval
):
    """Store.delete_collection() raising an unexpected error returns 500."""
    from unittest.mock import patch

    class ExplodingStore:
        def __init__(self, name):
            pass

        def delete_collection(self):
            raise RuntimeError("qdrant unavailable")

    with patch(
        "claritymed.stores.knowledge.RagCollectionStore", ExplodingStore, create=True
    ):
        response = await web_client.delete(
            "/api/v1/admin/rag/collections/statpearls",
            cookies=admin_cookies,
            headers={"X-CSRF-Token": "csrf-test-token"},
        )
    assert response.status_code == 500
    assert "delete failed" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upsert_metadata_pydantic_error_returns_422(
    web_client, admin_cookies, tmp_retrieval
):
    """Metadata with unknown fields or wrong types raises 422 (Pydantic extra='forbid')."""
    response = await web_client.post(
        "/api/v1/admin/rag/collections/upsert",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        data={"metadata": json.dumps({"name": "good_name", "unknown_field": "boom"})},
        files={"files": ("note.txt", b"x", "text/plain")},
    )
    assert response.status_code == 422


# --- _chunk_count unit tests -------------------------------------------


@pytest.mark.asyncio
async def test_chunk_count_collection_not_exists_returns_none():
    """collection_exists() → False means pre-bootstrap → None."""
    from claritymed.web.routers.admin.rag import _chunk_count

    aclient = MagicMock()
    aclient.collection_exists = AsyncMock(return_value=False)
    result = await _chunk_count(aclient, "no-such-col")
    assert result is None


@pytest.mark.asyncio
async def test_chunk_count_returns_int_on_success():
    """Happy path: collection exists + count call returns info."""
    from claritymed.web.routers.admin.rag import _chunk_count

    aclient = MagicMock()
    aclient.collection_exists = AsyncMock(return_value=True)
    info = MagicMock()
    info.count = 42
    aclient.count = AsyncMock(return_value=info)
    result = await _chunk_count(aclient, "my-col")
    assert result == 42


@pytest.mark.asyncio
async def test_chunk_count_exception_returns_none():
    """Any exception from qdrant is swallowed → None."""
    from claritymed.web.routers.admin.rag import _chunk_count

    aclient = MagicMock()
    aclient.collection_exists = AsyncMock(side_effect=RuntimeError("qdrant gone"))
    result = await _chunk_count(aclient, "my-col")
    assert result is None


@pytest.mark.asyncio
async def test_list_collections_with_nameless_entry_skipped(
    web_client, admin_cookies, tmp_retrieval, monkeypatch
):
    """An entry with no 'name' key is silently skipped."""
    from claritymed.web.routers.admin import rag as rag_router

    monkeypatch.setattr(
        rag_router, "_load_system_entries", lambda: [{"language": "en"}]
    )
    response = await web_client.get(
        "/api/v1/admin/rag/collections", cookies=admin_cookies
    )
    assert response.status_code == 200
    assert response.json()["total_count"] == 0


@pytest.mark.asyncio
async def test_list_collections_with_qdrant_aclient(
    web_client, admin_cookies, tmp_retrieval, monkeypatch
):
    """When _open_qdrant_aclient returns a mock, close() is called after."""
    from unittest.mock import AsyncMock, MagicMock

    from claritymed.web.routers.admin import rag as rag_router

    mock_aclient = MagicMock()
    mock_aclient.collection_exists = AsyncMock(return_value=False)
    mock_aclient.close = AsyncMock()
    monkeypatch.setattr(rag_router, "_open_qdrant_aclient", lambda: mock_aclient)

    response = await web_client.get(
        "/api/v1/admin/rag/collections", cookies=admin_cookies
    )
    assert response.status_code == 200
    mock_aclient.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_inspect_collection_with_qdrant_aclient(
    web_client, admin_cookies, tmp_retrieval, monkeypatch
):
    """inspect_collection closes aclient after call."""
    from unittest.mock import AsyncMock, MagicMock

    from claritymed.web.routers.admin import rag as rag_router

    mock_aclient = MagicMock()
    mock_aclient.collection_exists = AsyncMock(return_value=False)
    mock_aclient.close = AsyncMock()
    monkeypatch.setattr(rag_router, "_open_qdrant_aclient", lambda: mock_aclient)

    response = await web_client.get(
        "/api/v1/admin/rag/collections/statpearls", cookies=admin_cookies
    )
    assert response.status_code == 200
    mock_aclient.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_upsert_duplicate_filenames_are_disambiguated(
    web_client, admin_cookies, tmp_retrieval, monkeypatch
):
    """Two files with the same name are stored as file.txt and file-2.txt."""
    from unittest.mock import AsyncMock, patch

    from claritymed.ingest.corpus.base import IngestStats
    from claritymed.ingest.system_rag import SystemRagIngestResult

    fake_result = SystemRagIngestResult(
        stats=IngestStats(
            source="t",
            docs_processed=2,
            parents_written=4,
            children_written=6,
            docs_skipped=0,
        ),
        yaml_snippet="",
        is_new_collection=False,
        centroid_refreshed=False,
    )

    with patch(
        "claritymed.web.admin.job_runners.rag_ingest.ingest_system_rag",
        new=AsyncMock(return_value=fake_result),
    ):
        response = await web_client.post(
            "/api/v1/admin/rag/collections/upsert",
            cookies=admin_cookies,
            headers={"X-CSRF-Token": "csrf-test-token"},
            data={"metadata": json.dumps({"name": "good_name"})},
            files=[
                ("files", ("dup.txt", b"first content", "text/plain")),
                ("files", ("dup.txt", b"second content", "text/plain")),
            ],
        )
    assert response.status_code == 200
