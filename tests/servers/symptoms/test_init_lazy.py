"""Unit tests for the PEP 562 lazy-attribute loader in symptoms/__init__.py.

These tests use mock objects so the ``symptoms-server`` extra (uvicorn,
fastapi, xgboost …) does NOT need to be installed.
"""

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


def _reload_symptoms_init() -> types.ModuleType:
    """Force a fresh import of the package so cached attributes don't bleed."""
    pkg = "claritymed.servers.symptoms"
    # Remove any cached attributes that a previous test may have attached
    # (setattr caches the value on the module to short-circuit future access).
    mod = sys.modules.get(pkg)
    if mod is not None:
        mod.__dict__.pop("app", None)
        mod.__dict__.pop("main", None)
    return importlib.import_module(pkg)


def test_lazy_getattr_app_triggers_import():
    """Accessing .app calls importlib.import_module and returns the result."""
    fake_app = MagicMock(name="fake_app")
    fake_module = MagicMock()
    fake_module.app = fake_app

    mod = _reload_symptoms_init()

    with patch("importlib.import_module", return_value=fake_module) as mock_import:
        result = mod.app

    mock_import.assert_called_once_with("claritymed.servers.symptoms.app")
    assert result is fake_app


def test_lazy_getattr_main_triggers_import():
    """Accessing .main calls importlib.import_module and returns the result."""
    fake_main = MagicMock(name="fake_main")
    fake_module = MagicMock()
    fake_module.main = fake_main

    mod = _reload_symptoms_init()

    with patch("importlib.import_module", return_value=fake_module) as mock_import:
        result = mod.main

    mock_import.assert_called_once_with("claritymed.servers.symptoms.app")
    assert result is fake_main


def test_lazy_getattr_caches_value():
    """Second access returns the cached attribute without re-importing."""
    fake_app = MagicMock(name="cached_app")
    fake_module = MagicMock()
    fake_module.app = fake_app

    mod = _reload_symptoms_init()

    with patch("importlib.import_module", return_value=fake_module) as mock_import:
        first = mod.app
        second = mod.app

    assert mock_import.call_count == 1
    assert first is second is fake_app


def test_lazy_getattr_unknown_raises_attribute_error():
    """Accessing an unknown attribute raises AttributeError."""
    mod = _reload_symptoms_init()
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = mod.does_not_exist
