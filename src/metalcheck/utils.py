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
        # Synchronize GPU operations before timing
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
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
    elif device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = a @ b
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return {
        "device": str(device),
        "matrix_size": size,
        "iterations": iterations,
        "mean_time_s": sum(times) / len(times),
        "min_time_s": min(times),
        "max_time_s": max(times),
    }


def benchmark_conv2d(
    spatial_size: int = 128,
    in_channels: int = 64,
    out_channels: int = 64,
    kernel_size: int = 3,
    batch_size: int = 1,
    device: torch.device | None = None,
    iterations: int = 10,
) -> dict:
    """Benchmark Conv2d on the given device.

    Args:
        spatial_size: Height and width of the input feature map.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        batch_size: Batch size.
        device: Device to run on. Uses best available if None.
        iterations: Number of iterations to average over.

    Returns:
        Dict with timing results.
    """
    from metalcheck.device import get_device

    if device is None:
        device = get_device()

    conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2).to(
        device
    )
    x = torch.randn(batch_size, in_channels, spatial_size, spatial_size, device=device)

    # Warmup
    with torch.no_grad():
        _ = conv(x)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        with torch.no_grad():
            _ = conv(x)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return {
        "device": str(device),
        "spatial_size": spatial_size,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "batch_size": batch_size,
        "iterations": iterations,
        "mean_time_s": sum(times) / len(times),
        "min_time_s": min(times),
        "max_time_s": max(times),
    }


def benchmark_elementwise(
    num_elements: int = 10_000_000,
    op: str = "add",
    device: torch.device | None = None,
    iterations: int = 10,
) -> dict:
    """Benchmark an element-wise operation on a large tensor.

    Args:
        num_elements: Number of tensor elements.
        op: One of 'add', 'mul', 'exp', 'sin', 'relu'.
        device: Device to run on. Uses best available if None.
        iterations: Number of iterations to average over.

    Returns:
        Dict with timing results.
    """
    from metalcheck.device import get_device

    if device is None:
        device = get_device()

    a = torch.randn(num_elements, device=device)
    b = torch.randn(num_elements, device=device)

    ops = {
        "add": lambda: torch.add(a, b),
        "mul": lambda: torch.mul(a, b),
        "exp": lambda: torch.exp(a),
        "sin": lambda: torch.sin(a),
        "relu": lambda: torch.nn.functional.relu(a),
    }
    fn = ops[op]

    # Warmup
    _ = fn()
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = fn()
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return {
        "device": str(device),
        "num_elements": num_elements,
        "op": op,
        "iterations": iterations,
        "mean_time_s": sum(times) / len(times),
        "min_time_s": min(times),
        "max_time_s": max(times),
    }


def benchmark_model_inference(
    model_factory: Callable,
    input_shape: tuple,
    device: torch.device | None = None,
    iterations: int = 10,
) -> dict:
    """Benchmark model inference (forward pass only).

    Args:
        model_factory: Callable that returns an ``nn.Module``.
        input_shape: Shape of the input tensor (including batch dim).
        device: Device to run on. Uses best available if None.
        iterations: Number of iterations to average over.

    Returns:
        Dict with timing results.
    """
    from metalcheck.device import get_device

    if device is None:
        device = get_device()

    model = model_factory().to(device).eval()
    x = torch.randn(*input_shape, device=device)

    # Warmup
    with torch.no_grad():
        _ = model(x)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return {
        "device": str(device),
        "input_shape": list(input_shape),
        "iterations": iterations,
        "mean_time_s": sum(times) / len(times),
        "min_time_s": min(times),
        "max_time_s": max(times),
    }


def benchmark_model_training(
    model_factory: Callable,
    input_shape: tuple,
    num_classes: int = 1000,
    device: torch.device | None = None,
    steps: int = 20,
) -> dict:
    """Benchmark model training (forward + loss + backward + optimizer step).

    Args:
        model_factory: Callable that returns an ``nn.Module``.
        input_shape: Shape of the input tensor (including batch dim).
        num_classes: Number of output classes for the loss target.
        device: Device to run on. Uses best available if None.
        steps: Number of training steps to average over.

    Returns:
        Dict with timing results.
    """
    from metalcheck.device import get_device

    if device is None:
        device = get_device()

    model = model_factory().to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    x = torch.randn(*input_shape, device=device)
    targets = torch.randint(0, num_classes, (input_shape[0],), device=device)

    # Warmup
    out = model(x)
    loss = criterion(out, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(steps):
        start = time.perf_counter()
        out = model(x)
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return {
        "device": str(device),
        "input_shape": list(input_shape),
        "num_classes": num_classes,
        "steps": steps,
        "mean_time_s": sum(times) / len(times),
        "min_time_s": min(times),
        "max_time_s": max(times),
    }


def system_info() -> dict:
    """Return system information as a dictionary."""
    return {
        "os": platform.system(),
        "os_version": platform.mac_ver()[0]
        if platform.system() == "Darwin"
        else platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
    }
