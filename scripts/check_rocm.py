#!/usr/bin/env python3
"""Standalone script to verify ROCm (HIP) GPU availability and run a quick benchmark."""

from metalcheck.device import get_device, is_rocm_available, print_device_info
from metalcheck.utils import benchmark_matmul, system_info


def main() -> None:
    print_device_info()

    print("\n📊 System Information:")
    for key, value in system_info().items():
        label = key.replace("_", " ").title()
        print(f"  {label:<20s}: {value}")

    device = get_device()
    print(f"\n🔥 Running matmul benchmark on {device}...")
    results = benchmark_matmul(size=1024, device=device, iterations=20)
    print(f"  Matrix size : {results['matrix_size']}x{results['matrix_size']}")
    print(f"  Iterations  : {results['iterations']}")
    print(f"  Mean time   : {results['mean_time_s']*1000:.2f} ms")
    print(f"  Min time    : {results['min_time_s']*1000:.2f} ms")
    print(f"  Max time    : {results['max_time_s']*1000:.2f} ms")

    # Compare ROCm vs CPU if ROCm is available
    if is_rocm_available():
        import torch

        print("\n📈 ROCm vs CPU Comparison (1024x1024 matmul, 20 iterations):")
        rocm_results = benchmark_matmul(size=1024, device=torch.device("cuda"), iterations=20)
        cpu_results = benchmark_matmul(size=1024, device=torch.device("cpu"), iterations=20)
        speedup = cpu_results["mean_time_s"] / rocm_results["mean_time_s"]
        print(f"  ROCm mean: {rocm_results['mean_time_s']*1000:.2f} ms")
        print(f"  CPU mean : {cpu_results['mean_time_s']*1000:.2f} ms")
        print(f"  Speedup  : {speedup:.1f}x")


if __name__ == "__main__":
    main()
