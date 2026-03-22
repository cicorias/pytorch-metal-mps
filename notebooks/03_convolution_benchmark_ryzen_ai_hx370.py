#!/usr/bin/env python3
"""Convolution benchmark — CSV + PNG output version.

Runs the same Conv2d benchmarks as the companion notebook and writes CSV data
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
from metalcheck.utils import benchmark_conv2d

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = REPO_ROOT / "output"

SPATIAL_SIZES = [64, 128, 256, 512]
KERNEL_SIZES = [3, 5, 7]
CHANNELS = 64
CHANNEL_COUNTS = [32, 64, 128, 256]
FIXED_SPATIAL = 128
FIXED_KERNEL = 3
ITERATIONS = 10

_log_fd: int | None = None


def trace(msg: str) -> None:
    """Write a timestamped line to the log file using os-level write (unbuffered)."""
    global _log_fd
    if _log_fd is None:
        logfile = OUTDIR / "03_convolution_benchmark_ryzen_ai_hx370.log"
        _log_fd = os.open(str(logfile), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}\n"
    os.write(_log_fd, line.encode())


def run_spatial() -> list[dict]:
    rows: list[dict] = []

    for spatial in SPATIAL_SIZES:
        for ks in KERNEL_SIZES:
            for dev_name, dev in [("ROCm", torch.device("cuda")), ("CPU", torch.device("cpu"))]:
                if dev_name == "ROCm" and not is_rocm_available():
                    continue
                tag = f"conv2d spatial={spatial} kernel={ks} | {dev_name}"
                trace(f"BEGIN {tag}")
                print(f"  spatial={spatial} kernel={ks} | {dev_name} ...", end=" ", flush=True, file=sys.stderr)
                r = benchmark_conv2d(
                    spatial_size=spatial,
                    in_channels=CHANNELS,
                    out_channels=CHANNELS,
                    kernel_size=ks,
                    batch_size=1,
                    device=dev,
                    iterations=ITERATIONS,
                )
                time_ms = r["mean_time_s"] * 1000
                trace(f"END   {tag} -> {time_ms:.4f} ms")
                rows.append({
                    "Spatial": f"{spatial}x{spatial}",
                    "Kernel": ks,
                    "Device": dev_name,
                    "Time (ms)": f"{time_ms:.4f}",
                })
                print(f"{time_ms:.2f} ms", file=sys.stderr)

    trace("writing spatial CSV")
    csv_out = OUTDIR / "03_conv2d_spatial.csv"
    fieldnames = ["Spatial", "Kernel", "Device", "Time (ms)"]
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Spatial results written to {csv_out}", file=sys.stderr)
    return rows


def plot_spatial(rows: list[dict]) -> None:
    trace("generating spatial chart")
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, len(KERNEL_SIZES), figsize=(16, 5), sharey=False)

    for ax, ks in zip(axes, KERNEL_SIZES):
        ks_rows = [r for r in rows if r["Kernel"] == ks]
        spatial_labels = [f"{s}x{s}" for s in SPATIAL_SIZES]
        x = range(len(SPATIAL_SIZES))
        width = 0.35

        rocm_vals = [float(r["Time (ms)"]) for r in ks_rows if r["Device"] == "ROCm"]
        cpu_vals = [float(r["Time (ms)"]) for r in ks_rows if r["Device"] == "CPU"]

        if rocm_vals:
            ax.bar([i - width / 2 for i in x], rocm_vals, width, label="ROCm (Radeon 890M)", color="#E4002B")
        ax.bar([i + width / 2 for i in x], cpu_vals, width, label="CPU", color="#4A90D9")

        ax.set_xlabel("Input Size")
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"Conv2d — kernel {ks}x{ks}")
        ax.set_xticks(list(x))
        ax.set_xticklabels(spatial_labels)
        ax.legend()

    plt.suptitle(f"Conv2d ({CHANNELS} channels): ROCm vs CPU", fontsize=14, y=1.02)
    plt.tight_layout()

    out = OUTDIR / "03_conv2d_spatial.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    trace(f"saved {out}")
    print(f"Chart saved to {out}", file=sys.stderr)


def run_channel_scaling() -> list[dict]:
    rows: list[dict] = []

    for ch in CHANNEL_COUNTS:
        for dev_name, dev in [("ROCm", torch.device("cuda")), ("CPU", torch.device("cpu"))]:
            if dev_name == "ROCm" and not is_rocm_available():
                continue
            tag = f"conv2d channels={ch} | {dev_name}"
            trace(f"BEGIN {tag}")
            print(f"  channels={ch} | {dev_name} ...", end=" ", flush=True, file=sys.stderr)
            r = benchmark_conv2d(
                spatial_size=FIXED_SPATIAL,
                in_channels=ch,
                out_channels=ch,
                kernel_size=FIXED_KERNEL,
                batch_size=1,
                device=dev,
                iterations=ITERATIONS,
            )
            time_ms = r["mean_time_s"] * 1000
            trace(f"END   {tag} -> {time_ms:.4f} ms")
            rows.append({
                "Channels": ch,
                "Device": dev_name,
                "Time (ms)": f"{time_ms:.4f}",
            })
            print(f"{time_ms:.2f} ms", file=sys.stderr)

    trace("writing channel scaling CSV")
    csv_out = OUTDIR / "03_conv2d_channels.csv"
    fieldnames = ["Channels", "Device", "Time (ms)"]
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Channel scaling results written to {csv_out}", file=sys.stderr)
    return rows


def plot_channel_scaling(rows: list[dict]) -> None:
    trace("generating channel scaling chart")
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(CHANNEL_COUNTS))
    width = 0.35

    rocm_vals = [float(r["Time (ms)"]) for r in rows if r["Device"] == "ROCm"]
    cpu_vals = [float(r["Time (ms)"]) for r in rows if r["Device"] == "CPU"]

    if rocm_vals:
        ax.bar([i - width / 2 for i in x], rocm_vals, width, label="ROCm (Radeon 890M)", color="#E4002B")
    ax.bar([i + width / 2 for i in x], cpu_vals, width, label="CPU", color="#4A90D9")

    ax.set_xlabel("Channels")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Conv2d {FIXED_KERNEL}x{FIXED_KERNEL} @ {FIXED_SPATIAL}x{FIXED_SPATIAL}: Channel Scaling")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(c) for c in CHANNEL_COUNTS])
    ax.legend()
    plt.tight_layout()

    out = OUTDIR / "03_conv2d_channels.png"
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

    print("\n--- Conv2d Spatial Benchmark ---", file=sys.stderr)
    trace("=== SPATIAL PHASE ===")
    spatial_rows = run_spatial()
    plot_spatial(spatial_rows)

    print("\n--- Conv2d Channel Scaling Benchmark ---", file=sys.stderr)
    trace("=== CHANNEL SCALING PHASE ===")
    channel_rows = run_channel_scaling()
    plot_channel_scaling(channel_rows)

    trace("done")


if __name__ == "__main__":
    main()
