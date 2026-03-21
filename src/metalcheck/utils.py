"""Common utility helpers for metalcheck."""

from __future__ import annotations

import functools
import platform
import time
from typing import Any, Callable

import torch


def timer(func: Callable) -> Callable:
    """Decorator that prints execution time of a function."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        # Synchronize MPS operations before timing
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        elapsed = time.perf_counter() - start
        print(f"⏱  {func.__name__} took {elapsed:.4f}s")
        return result

    return wrapper


def benchmark_matmul(
    size: int = 1024,
    device: torch.device | None = None,
    iterations: int = 10,
) -> dict:
    """Benchmark matrix multiplication on the given device.

    Args:
        size: Matrix dimension (size x size).
        device: Device to run on. Uses best available if None.
        iterations: Number of iterations to average over.

    Returns:
        Dict with timing results.
    """
    from metalcheck.device import get_device

    if device is None:
        device = get_device()

    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Warmup
    _ = a @ b
    if device.type == "mps":
        torch.mps.synchronize()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = a @ b
        if device.type == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - start)

    return {
        "device": str(device),
        "matrix_size": size,
        "iterations": iterations,
        "mean_time_s": sum(times) / len(times),
        "min_time_s": min(times),
        "max_time_s": max(times),
    }


def system_info() -> dict:
    """Return system information as a dictionary."""
    return {
        "os": platform.system(),
        "os_version": platform.mac_ver()[0] if platform.system() == "Darwin" else platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
    }
