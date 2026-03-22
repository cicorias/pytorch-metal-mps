#!/usr/bin/env python3
"""Element-wise operations benchmark — CSV + PNG output version.

Runs the same element-wise benchmarks as the companion notebook and writes
CSV data and chart images to the ./output directory.
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
import pandas as pd

import torch

from metalcheck import device_info, is_rocm_available
from metalcheck.utils import benchmark_elementwise

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = REPO_ROOT / "output"

OPS = ["add", "mul", "exp", "sin", "relu"]
SIZES = [1_000_000, 10_000_000, 50_000_000, 100_000_000]
SIZE_LABELS = ["1M", "10M", "50M", "100M"]
ITERATIONS = 10

_log_fd: int | None = None


def trace(msg: str) -> None:
    """Write a timestamped line to the log file using os-level write (unbuffered)."""
    global _log_fd
    if _log_fd is None:
        logfile = OUTDIR / "04_elementwise_benchmark_ryzen_ai_hx370.log"
        _log_fd = os.open(str(logfile), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}\n"
    os.write(_log_fd, line.encode())


def plot_per_op(rows: list[dict]) -> None:
    """Per-operation bar charts (matches notebook section 3)."""
    trace("generating per-op bar charts")
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, len(OPS), figsize=(20, 5), sharey=False)

    for ax, op in zip(axes, OPS):
        op_rows = [r for r in rows if r["Op"] == op]
        x = range(len(SIZES))
        width = 0.35

        rocm_vals = [float(r["Time (ms)"]) for r in op_rows if r["Device"] == "ROCm"]
        cpu_vals = [float(r["Time (ms)"]) for r in op_rows if r["Device"] == "CPU"]

        if rocm_vals:
            ax.bar([i - width / 2 for i in x], rocm_vals, width, label="ROCm", color="#E4002B")
        ax.bar([i + width / 2 for i in x], cpu_vals, width, label="CPU", color="#4A90D9")

        ax.set_xlabel("Elements")
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"torch.{op}")
        ax.set_xticks(list(x))
        ax.set_xticklabels(SIZE_LABELS, fontsize=8)
        ax.legend(fontsize=8)

    plt.suptitle("Element-wise Operations: ROCm vs CPU", fontsize=14, y=1.02)
    plt.tight_layout()

    out = OUTDIR / "04_elementwise_ops.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    trace(f"saved {out}")
    print(f"Chart saved to {out}", file=sys.stderr)


def plot_speedup_heatmap(rows: list[dict]) -> None:
    """Speedup heatmap (matches notebook section 4)."""
    rocm_rows = [r for r in rows if r["Device"] == "ROCm"]
    if not rocm_rows:
        trace("skipping heatmap — no ROCm data")
        return

    trace("generating speedup heatmap")
    sns.set_theme(style="whitegrid")

    speedup_data = []
    for op in OPS:
        row = {}
        for sz_label in SIZE_LABELS:
            rocm_t = next(float(r["Time (ms)"]) for r in rows if r["Op"] == op and r["Elements"] == sz_label and r["Device"] == "ROCm")
            cpu_t = next(float(r["Time (ms)"]) for r in rows if r["Op"] == op and r["Elements"] == sz_label and r["Device"] == "CPU")
            row[sz_label] = cpu_t / rocm_t
        speedup_data.append(row)

    df_speedup = pd.DataFrame(speedup_data, index=OPS)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        df_speedup, annot=True, fmt=".1f", cmap="RdYlGn",
        center=1.0, linewidths=0.5, ax=ax,
        cbar_kws={"label": "Speedup (CPU time / ROCm time)"},
    )
    ax.set_title("ROCm Speedup over CPU")
    ax.set_xlabel("Tensor Size")
    ax.set_ylabel("Operation")
    plt.tight_layout()

    out = OUTDIR / "04_elementwise_speedup.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    trace(f"saved {out}")
    print(f"Heatmap saved to {out}", file=sys.stderr)


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

    for op in OPS:
        for sz, sz_label in zip(SIZES, SIZE_LABELS):
            for dev_name, dev in [("ROCm", torch.device("cuda")), ("CPU", torch.device("cpu"))]:
                if dev_name == "ROCm" and not is_rocm_available():
                    continue
                tag = f"{op} {sz_label} | {dev_name}"
                trace(f"BEGIN {tag}")
                print(f"  {op:>4s} | {sz_label:>4s} | {dev_name} ...", end=" ", flush=True, file=sys.stderr)
                r = benchmark_elementwise(num_elements=sz, op=op, device=dev, iterations=ITERATIONS)
                time_ms = r["mean_time_s"] * 1000
                trace(f"END   {tag} -> {time_ms:.4f} ms")
                rows.append({
                    "Op": op,
                    "Elements": sz_label,
                    "Device": dev_name,
                    "Time (ms)": f"{time_ms:.4f}",
                })
                print(f"{time_ms:.2f} ms", file=sys.stderr)

    trace("writing CSV")
    csv_out = OUTDIR / "04_elementwise.csv"
    fieldnames = ["Op", "Elements", "Device", "Time (ms)"]
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results written to {csv_out}", file=sys.stderr)

    plot_per_op(rows)
    plot_speedup_heatmap(rows)
    trace("done")


if __name__ == "__main__":
    main()
