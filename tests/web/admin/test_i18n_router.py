"""Tests for ``/api/v1/admin/i18n``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from claritymed import config as _cfg


@pytest.fixture
def tmp_i18n_dir(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    i18n = cfg_dir / "i18n"
    i18n.mkdir()
    (i18n / "en.yaml").write_text(
        yaml.safe_dump(
            {
                "app": {"hello": "Hello"},
                "errors": {"invalid": "Invalid input"},
            }
        ),
        encoding="utf-8",
    )
    (i18n / "zh.yaml").write_text(
        yaml.safe_dump({"app": {"hello": "你好"}}, allow_unicode=True),
        encoding="utf-8",
    )
    monkeypatch.setattr(_cfg, "CONFIGS_DIR", cfg_dir)
    monkeypatch.setattr(_cfg, "I18N_DIR", i18n)
    yield i18n


@pytest.fixture
def tmp_admin_ui_locale(tmp_path, monkeypatch):
    """Redirect the admin_ui locale resolution to a tmp dir so the test
    doesn't write through the live source file. Seeds en.json + zh.json
    with a minimal nav block so the read/patch tests have content to
    interact with.
    """
    locale_dir = tmp_path / "i18n"
    locale_dir.mkdir()
    for lang, seed in (
        ("en", {"nav": {"overview": "Overview"}}),
        ("zh", {"nav": {"overview": "概览"}}),
    ):
        (locale_dir / f"{lang}.json").write_text(
            json.dumps(seed, ensure_ascii=False), encoding="utf-8"
        )

    from claritymed.web.routers.admin import i18n as i18n_router

    def _redirected_path(lang: str) -> Path:
        return locale_dir / f"{lang}.json"

    monkeypatch.setattr(i18n_router, "_admin_ui_path", _redirected_path)
    yield locale_dir


@pytest.mark.asyncio
async def test_i18n_requires_admin(web_client, non_admin_cookies):
    response = await web_client.get(
        "/api/v1/admin/i18n/backend-strings?lang=en", cookies=non_admin_cookies
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_read_backend_strings(web_client, admin_cookies, tmp_i18n_dir):
    response = await web_client.get(
        "/api/v1/admin/i18n/backend-strings?lang=en", cookies=admin_cookies
    )
    body = response.json()
    assert body["data"]["app"]["hello"] == "Hello"


@pytest.mark.asyncio
async def test_read_unsupported_language(web_client, admin_cookies):
    response = await web_client.get(
        "/api/v1/admin/i18n/backend-strings?lang=fr", cookies=admin_cookies
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_patch_backend_strings_deep_merge(
    web_client, admin_cookies, tmp_i18n_dir
):
    response = await web_client.patch(
        "/api/v1/admin/i18n/backend-strings?lang=en",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        json={"updates": {"app": {"hello": "Hi!"}, "newkey": "fresh"}},
    )
    body = response.json()
    assert body["data"]["app"]["hello"] == "Hi!"
    assert body["data"]["errors"]["invalid"] == "Invalid input"
    assert body["data"]["newkey"] == "fresh"


@pytest.mark.asyncio
async def test_admin_strings_read_existing(
    web_client, admin_cookies, tmp_admin_ui_locale
):
    response = await web_client.get(
        "/api/v1/admin/i18n/admin-strings?lang=en", cookies=admin_cookies
    )
    body = response.json()
    assert "nav" in body["data"]


@pytest.mark.asyncio
async def test_admin_strings_patch_needs_rebuild(
    web_client, admin_cookies, tmp_admin_ui_locale
):
    response = await web_client.patch(
        "/api/v1/admin/i18n/admin-strings?lang=en",
        cookies=admin_cookies,
        headers={"X-CSRF-Token": "csrf-test-token"},
        json={"updates": {"nav": {"overview": "Dashboard"}}},
    )
    body = response.json()
    assert body["needs_rebuild"] is True
    # Verify the on-disk JSON was actually updated.
    en_path = tmp_admin_ui_locale / "en.json"
    on_disk = json.loads(en_path.read_text(encoding="utf-8"))
    assert on_disk["nav"]["overview"] == "Dashboard"
