#!/usr/bin/env python3
"""ROCm Quick Start benchmark — CSV + PNG output version.

Runs the same matmul benchmarks as the companion notebook and writes CSV data
and chart images to the ./output directory.
"""

import csv
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch

from metalcheck import device_info, is_rocm_available
from metalcheck.utils import benchmark_matmul

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = REPO_ROOT / "output"

SIZES = [256, 512, 1024, 2048, 4096]
ITERATIONS = 10

_log_fd: int | None = None


def trace(msg: str) -> None:
    """Write a timestamped line to the log file using os-level write (unbuffered)."""
    global _log_fd
    if _log_fd is None:
        logfile = OUTDIR / "01_mps_quickstart_ryzen_ai_hx370.log"
        _log_fd = os.open(str(logfile), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}\n"
    os.write(_log_fd, line.encode())


def plot_matmul(rows: list[dict]) -> None:
    """Reproduce the notebook's matmul bar chart and save as PNG."""
    trace("generating matmul chart")
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(SIZES))
    width = 0.35

    rocm_times = [float(r["Time (ms)"]) for r in rows if r["Device"] == "ROCm"]
    cpu_times = [float(r["Time (ms)"]) for r in rows if r["Device"] == "CPU"]

    if rocm_times:
        ax.bar([i - width / 2 for i in x], rocm_times, width, label="ROCm (Radeon 890M)", color="#E4002B")
    ax.bar([i + width / 2 for i in x], cpu_times, width, label="CPU", color="#4A90D9")

    ax.set_xlabel("Matrix Size")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Matrix Multiplication: ROCm vs CPU")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{s}x{s}" for s in SIZES])
    ax.legend()
    plt.tight_layout()

    out = OUTDIR / "01_matmul_benchmark.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    trace(f"saved {out}")
    print(f"Chart saved to {out}", file=sys.stderr)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    trace("script started")
    trace(f"pid={os.getpid()}")

    info = device_info()
    trace("device_info() returned")
    for k, v in info.items():
        label = k.replace("_", " ").title()
        print(f"{label:<20s}: {v}", file=sys.stderr)
        trace(f"  {label}: {v}")

    rows: list[dict] = []

    for size in SIZES:
        for dev_name, dev in [("ROCm", torch.device("cuda")), ("CPU", torch.device("cpu"))]:
            if dev_name == "ROCm" and not is_rocm_available():
                continue
            tag = f"matmul {size}x{size} | {dev_name}"
            trace(f"BEGIN {tag}")
            print(f"  {tag} ...", end=" ", flush=True, file=sys.stderr)
            r = benchmark_matmul(size=size, device=dev, iterations=ITERATIONS)
            time_ms = r["mean_time_s"] * 1000
            trace(f"END   {tag} -> {time_ms:.4f} ms")
            rows.append({
                "Size": f"{size}x{size}",
                "Device": dev_name,
                "Time (ms)": f"{time_ms:.4f}",
            })
            print(f"{time_ms:.2f} ms", file=sys.stderr)

    trace("writing CSV")
    csv_out = OUTDIR / "01_matmul_benchmark.csv"
    fieldnames = ["Size", "Device", "Time (ms)"]
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results written to {csv_out}", file=sys.stderr)

    plot_matmul(rows)
    trace("done")


if __name__ == "__main__":
    main()
