"""Test fixtures shared by all admin-router tests.

Inherits the parent ``tests/web/conftest.py`` fixtures (``web_client``,
``test_user`` — the first user, auto-admin) and adds:

* ``admin_cookies`` — pre-built cookie pair for ``test_user`` so admin
  endpoints can be hit with one line.
* ``non_admin_cookies`` — cookie pair for a second user (role=``user``)
  so 403-path tests don't need to re-issue tokens themselves.
"""

from __future__ import annotations

import pytest

from tests.web.conftest import make_auth_cookies


@pytest.fixture
def admin_cookies(test_user):
    """Cookie dict for the admin (first) user."""
    return make_auth_cookies(test_user.user_id, lang=test_user.language)


@pytest.fixture
def non_admin_cookies(non_admin_user):
    """Cookie dict for a regular (second) user."""
    return make_auth_cookies(non_admin_user.user_id, lang=non_admin_user.language)
