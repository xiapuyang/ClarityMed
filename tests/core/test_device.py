"""Tests for ``claritymed.core.device.resolve_device``."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from claritymed.core.device import resolve_device


def test_explicit_device_is_returned_unchanged():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("mps") == "mps"


def test_auto_returns_mps_when_available():
    mock_torch = MagicMock()
    mock_torch.backends.mps.is_available.return_value = True
    mock_torch.cuda.is_available.return_value = False

    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = resolve_device("auto")

    assert result == "mps"


def test_auto_returns_cuda_when_mps_unavailable():
    mock_torch = MagicMock()
    mock_torch.backends.mps.is_available.return_value = False
    mock_torch.cuda.is_available.return_value = True

    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = resolve_device("auto")

    assert result == "cuda"


def test_auto_returns_cpu_when_no_accelerator():
    mock_torch = MagicMock()
    mock_torch.backends.mps.is_available.return_value = False
    mock_torch.cuda.is_available.return_value = False

    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = resolve_device("auto")

    assert result == "cpu"


def test_auto_returns_cpu_when_torch_import_fails():
    with patch("builtins.__import__", side_effect=ImportError("no torch")):
        # Can't patch all imports this way — use a lighter approach
        pass

    with patch.dict("sys.modules", {"torch": None}):
        result = resolve_device("auto")

    assert result == "cpu"
