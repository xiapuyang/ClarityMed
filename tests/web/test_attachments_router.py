"""``POST /api/v1/sessions/{id}/attachments`` integration tests.

The OCR worker build is heavy and depends on a working provider catalog;
these tests force ``app.state.ocr_worker = None`` so the routes exercise
the upload + session_attachments paths without spinning up real OCR.
The text fast-path is exercised directly via .txt uploads and asserts
the OCR sentinel is written inline.
"""

from __future__ import annotations

import pytest

from claritymed.orchestrator.services import ChatSession
from claritymed.stores.account import init_user
from claritymed.stores.blob_store import BlobStore
from claritymed.stores.session_attachments import SessionAttachments
from claritymed.web.csrf import HEADER_CSRF_TOKEN
from claritymed.web.jwt import create_token
from claritymed.web.middleware import COOKIE_ACCESS_TOKEN


@pytest.fixture
def auth_cookies(test_user):
    token = create_token(test_user.user_id, test_user.language)
    return {COOKIE_ACCESS_TOKEN: token, "csrf_token": "csrf-test"}


def _csrf() -> dict[str, str]:
    return {HEADER_CSRF_TOKEN: "csrf-test"}


@pytest.fixture(autouse=True)
def disable_ocr_worker(web_app):
    """Force the worker build to skip — keeps tests hermetic."""
    web_app.state.ocr_worker = None
    yield


async def _new_session(web_client, auth_cookies) -> str:
    resp = await web_client.post(
        "/api/v1/sessions", cookies=auth_cookies, headers=_csrf()
    )
    return resp.json()["session_id"]


# --- text fast-path ----------------------------------------------------


async def test_upload_text_runs_fast_path(web_client, test_user, auth_cookies):
    sid = await _new_session(web_client, auth_cookies)
    resp = await web_client.post(
        f"/api/v1/sessions/{sid}/attachments",
        files={"files": ("notes.txt", b"hello world", "text/plain")},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["attachments"]) == 1
    att = body["attachments"][0]
    assert att["filename"] == "notes.txt"
    assert att["kind"] == "text"
    assert att["ocr_status"] == "done"
    # Sentinel landed on disk:
    blob_store = BlobStore(test_user.user_id)
    assert blob_store.ocr_done(att["id"])
    # Session tray sees the row:
    rows = SessionAttachments(test_user.user_id, sid).list()
    assert any(r.sha256 == att["id"] for r in rows)


async def test_upload_image_no_worker_left_pending(web_client, test_user, auth_cookies):
    sid = await _new_session(web_client, auth_cookies)
    # 1x1 transparent PNG header bytes — enough to hash, not a real PNG.
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    resp = await web_client.post(
        f"/api/v1/sessions/{sid}/attachments",
        files={"files": ("scan.png", png_bytes, "image/png")},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    # Even with the worker disabled the upload itself must succeed —
    # the attachment is registered and the next stream call surfaces
    # ocr_status="pending" inline. Failing the upload would leave the
    # user with no way to attach images at all.
    assert resp.status_code == 200, resp.text
    att = resp.json()["attachments"][0]
    assert att["kind"] == "image"
    assert att["ocr_status"] == "pending"


# --- validation --------------------------------------------------------


async def test_upload_empty_file_returns_422(web_client, test_user, auth_cookies):  # noqa: ARG001
    sid = await _new_session(web_client, auth_cookies)
    resp = await web_client.post(
        f"/api/v1/sessions/{sid}/attachments",
        files={"files": ("empty.txt", b"", "text/plain")},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    assert resp.status_code == 422


async def test_upload_unsupported_extension_returns_415(
    web_client, test_user, auth_cookies
):  # noqa: ARG001
    sid = await _new_session(web_client, auth_cookies)
    resp = await web_client.post(
        f"/api/v1/sessions/{sid}/attachments",
        files={"files": ("evil.exe", b"\x00\x00", "application/octet-stream")},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    assert resp.status_code == 415


async def test_upload_too_many_files_returns_422(web_client, test_user, auth_cookies):  # noqa: ARG001
    sid = await _new_session(web_client, auth_cookies)
    files = [("files", (f"f{i}.txt", b"data", "text/plain")) for i in range(9)]
    resp = await web_client.post(
        f"/api/v1/sessions/{sid}/attachments",
        files=files,
        cookies=auth_cookies,
        headers=_csrf(),
    )
    assert resp.status_code == 422


# --- ownership / csrf --------------------------------------------------


async def test_upload_against_other_user_session_returns_403(
    web_client, test_user, auth_cookies
):  # noqa: ARG001
    other = init_user("other", display_name="Other")
    other_session = ChatSession.new(other.user_id)
    other_session.append_system("init", kind="start")
    resp = await web_client.post(
        f"/api/v1/sessions/{other_session.session_id}/attachments",
        files={"files": ("notes.txt", b"hello", "text/plain")},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    assert resp.status_code == 403


async def test_upload_requires_csrf(web_client, test_user, auth_cookies):
    sid = await _new_session(web_client, auth_cookies)
    resp = await web_client.post(
        f"/api/v1/sessions/{sid}/attachments",
        files={"files": ("notes.txt", b"hello", "text/plain")},
        cookies=auth_cookies,
    )
    assert resp.status_code == 403


# --- GET /attachments/{sha} -------------------------------------------


async def test_get_attachment_returns_bytes(web_client, test_user, auth_cookies):  # noqa: ARG001
    sid = await _new_session(web_client, auth_cookies)
    png = b"\x89PNG\r\n\x1a\n" + b"\xab" * 32
    upload = await web_client.post(
        f"/api/v1/sessions/{sid}/attachments",
        files={"files": ("scan.png", png, "image/png")},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    sha_full = upload.json()["attachments"][0]["id"]

    resp = await web_client.get(
        f"/api/v1/sessions/{sid}/attachments/{sha_full}",
        cookies=auth_cookies,
    )
    assert resp.status_code == 200, resp.text
    assert resp.content == png
    assert resp.headers["content-type"].startswith("image/png")
    assert "scan.png" in resp.headers["content-disposition"]


async def test_get_attachment_short_prefix(web_client, test_user, auth_cookies):  # noqa: ARG001
    sid = await _new_session(web_client, auth_cookies)
    png = b"\x89PNG\r\n\x1a\n" + b"\xcd" * 32
    upload = await web_client.post(
        f"/api/v1/sessions/{sid}/attachments",
        files={"files": ("scan.png", png, "image/png")},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    sha_full = upload.json()["attachments"][0]["id"]
    prefix = sha_full[:8]

    resp = await web_client.get(
        f"/api/v1/sessions/{sid}/attachments/{prefix}",
        cookies=auth_cookies,
    )
    assert resp.status_code == 200, resp.text
    assert resp.content == png


async def test_get_attachment_non_image_returns_415(
    web_client,
    test_user,
    auth_cookies,  # noqa: ARG001
):
    sid = await _new_session(web_client, auth_cookies)
    upload = await web_client.post(
        f"/api/v1/sessions/{sid}/attachments",
        files={"files": ("notes.txt", b"hello world", "text/plain")},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    sha_full = upload.json()["attachments"][0]["id"]
    resp = await web_client.get(
        f"/api/v1/sessions/{sid}/attachments/{sha_full}",
        cookies=auth_cookies,
    )
    assert resp.status_code == 415


async def test_get_attachment_not_found_returns_404(web_client, auth_cookies):
    sid = await _new_session(web_client, auth_cookies)
    resp = await web_client.get(
        f"/api/v1/sessions/{sid}/attachments/{'a' * 8}",
        cookies=auth_cookies,
    )
    assert resp.status_code == 404


async def test_get_attachment_invalid_sha_returns_400(web_client, auth_cookies):
    sid = await _new_session(web_client, auth_cookies)
    # 7 chars — below the 8-char minimum
    resp = await web_client.get(
        f"/api/v1/sessions/{sid}/attachments/abcdef0",
        cookies=auth_cookies,
    )
    assert resp.status_code == 400


async def test_get_attachment_prefix_collision_returns_409(
    web_client, test_user, auth_cookies
):
    sid = await _new_session(web_client, auth_cookies)
    # Two synthetic rows sharing an 8-char prefix; rely on SessionAttachments
    # directly so we don't have to engineer real sha collisions.
    prefix = "abcd1234"
    tail_a = "0" * 56
    tail_b = "1" * 56
    tray = SessionAttachments(test_user.user_id, sid)
    tray.add(sha256=prefix + tail_a, filename="a.png", mime="image/png", size=1)
    tray.add(sha256=prefix + tail_b, filename="b.png", mime="image/png", size=1)

    resp = await web_client.get(
        f"/api/v1/sessions/{sid}/attachments/{prefix}",
        cookies=auth_cookies,
    )
    assert resp.status_code == 409


async def test_get_attachment_other_user_session_returns_403(
    web_client,
    test_user,
    auth_cookies,  # noqa: ARG001
):
    other = init_user("other", display_name="Other")
    other_session = ChatSession.new(other.user_id)
    other_session.append_system("init", kind="start")
    resp = await web_client.get(
        f"/api/v1/sessions/{other_session.session_id}/attachments/{'a' * 8}",
        cookies=auth_cookies,
    )
    assert resp.status_code == 403


# --- GET /attachments/meta --------------------------------------------


async def test_meta_batch_resolves_mixed_results(web_client, test_user, auth_cookies):
    sid = await _new_session(web_client, auth_cookies)
    upload = await web_client.post(
        f"/api/v1/sessions/{sid}/attachments",
        files={"files": ("notes.txt", b"hello world", "text/plain")},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    sha_full = upload.json()["attachments"][0]["id"]

    # Plant a colliding pair: both rows share the 8-char prefix "deadbeef".
    SessionAttachments(test_user.user_id, sid).add(
        sha256="deadbeef" + "0" * 56, filename="a.png", mime="image/png", size=1
    )
    SessionAttachments(test_user.user_id, sid).add(
        sha256="deadbeef" + "1" * 56, filename="b.png", mime="image/png", size=1
    )

    resp = await web_client.get(
        f"/api/v1/sessions/{sid}/attachments/meta",
        params=[
            ("sha", sha_full[:8]),  # short prefix, unique → resolved
            ("sha", "ffff0000"),  # not found
            ("sha", "dead"),  # 4 chars → invalid
            ("sha", "deadbeef"),  # 8 chars, matches both planted rows
        ],
        cookies=auth_cookies,
    )
    assert resp.status_code == 200, resp.text
    items = resp.json()["items"]
    assert len(items) == 4

    by_req = {it["requested"]: it for it in items}
    hit = by_req[sha_full[:8]]
    assert hit["error"] is None
    assert hit["resolved"]["id"] == sha_full
    assert hit["resolved"]["filename"] == "notes.txt"
    assert by_req["ffff0000"]["error"] == "not_found"
    assert by_req["dead"]["error"] == "invalid"
    assert by_req["deadbeef"]["error"] == "ambiguous"


async def test_meta_batch_requires_sha_param(web_client, auth_cookies):
    sid = await _new_session(web_client, auth_cookies)
    resp = await web_client.get(
        f"/api/v1/sessions/{sid}/attachments/meta",
        cookies=auth_cookies,
    )
    assert resp.status_code == 422


async def test_meta_batch_over_limit_returns_422(web_client, auth_cookies):
    sid = await _new_session(web_client, auth_cookies)
    params = [("sha", f"{i:08x}") for i in range(51)]
    resp = await web_client.get(
        f"/api/v1/sessions/{sid}/attachments/meta",
        params=params,
        cookies=auth_cookies,
    )
    assert resp.status_code == 422


async def test_meta_batch_other_user_session_returns_403(
    web_client,
    test_user,  # noqa: ARG001
    auth_cookies,
):
    other = init_user("other", display_name="Other")
    other_session = ChatSession.new(other.user_id)
    other_session.append_system("init", kind="start")
    resp = await web_client.get(
        f"/api/v1/sessions/{other_session.session_id}/attachments/meta",
        params=[("sha", "a" * 8)],
        cookies=auth_cookies,
    )
    assert resp.status_code == 403
