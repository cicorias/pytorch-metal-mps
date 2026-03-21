"""Tests for metalcheck.device module."""

import torch
import pytest

from metalcheck.device import (
    is_mps_available,
    is_mps_built,
    get_device,
    device_info,
)


def test_is_mps_available_returns_bool():
    result = is_mps_available()
    assert isinstance(result, bool)


def test_is_mps_built_returns_bool():
    result = is_mps_built()
    assert isinstance(result, bool)


def test_get_device_returns_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ("mps", "cpu")


def test_get_device_cpu_fallback():
    device = get_device(prefer_mps=False)
    assert device.type == "cpu"


def test_device_info_returns_dict():
    info = device_info()
    assert isinstance(info, dict)
    expected_keys = {
        "pytorch_version",
        "python_version",
        "platform",
        "architecture",
        "mps_built",
        "mps_available",
        "device",
    }
    assert expected_keys.issubset(info.keys())


def test_device_info_consistent_with_helpers():
    info = device_info()
    assert info["mps_available"] == is_mps_available()
    assert info["mps_built"] == is_mps_built()


@pytest.mark.skipif(not is_mps_available(), reason="MPS not available")
def test_mps_tensor_operation():
    """Verify that basic tensor operations work on MPS."""
    device = get_device()
    assert device.type == "mps"
    a = torch.randn(4, 4, device=device)
    b = torch.randn(4, 4, device=device)
    c = a @ b
    assert c.shape == (4, 4)
    assert c.device.type == "mps"
