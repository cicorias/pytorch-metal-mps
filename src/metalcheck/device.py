"""MPS / Metal and CUDA / ROCm device detection and utilities."""

from __future__ import annotations

import torch


def is_mps_available() -> bool:
    """Check if MPS (Metal Performance Shaders) backend is available."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def is_mps_built() -> bool:
    """Check if PyTorch was built with MPS support."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_built()


def is_cuda_available() -> bool:
    """Check if CUDA backend is available (covers both NVIDIA and AMD ROCm)."""
    return torch.cuda.is_available()


def is_rocm_available() -> bool:
    """Check if AMD ROCm (HIP) backend is available."""
    return torch.cuda.is_available() and getattr(torch.version, "hip", None) is not None


def get_device(prefer_mps: bool = True, prefer_cuda: bool = True) -> torch.device:
    """Return the best available device (MPS > CUDA/ROCm > CPU).

    Args:
        prefer_mps: If True (default), prefer MPS when available.
        prefer_cuda: If True (default), prefer CUDA/ROCm when available.

    Returns:
        torch.device for the best available accelerator, otherwise CPU.
    """
    if prefer_mps and is_mps_available():
        return torch.device("mps")
    if prefer_cuda and is_cuda_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_info() -> dict:
    """Return a dictionary of device and system information."""
    import platform

    info = {
        "pytorch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "mps_built": is_mps_built(),
        "mps_available": is_mps_available(),
        "cuda_available": is_cuda_available(),
        "rocm_available": is_rocm_available(),
        "device": str(get_device()),
    }
    hip_version = getattr(torch.version, "hip", None)
    if hip_version:
        info["hip_version"] = hip_version
    if is_cuda_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    return info


def print_device_info() -> None:
    """Print device info in a human-readable format."""
    info = device_info()
    print("=" * 50)
    print("  metalcheck — Device Information")
    print("=" * 50)
    for key, value in info.items():
        label = key.replace("_", " ").title()
        print(f"  {label:<20s}: {value}")
    print("=" * 50)


def main() -> None:
    """Entry point for the check-mps / check-rocm CLI command."""
    print_device_info()

    device = get_device()
    print(f"\nRunning a quick tensor operation on: {device}")
    a = torch.randn(3, 3, device=device)
    b = torch.randn(3, 3, device=device)
    c = a @ b
    print(f"Result (3x3 matmul):\n{c}")

    if device.type == "mps":
        print("\n✅ Metal GPU is working!")
    elif device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        backend = "ROCm" if is_rocm_available() else "CUDA"
        print(f"\n✅ {backend} GPU is working! ({gpu_name})")
    else:
        print("\n⚠️  Using CPU (no GPU backend available)")


if __name__ == "__main__":
    main()
