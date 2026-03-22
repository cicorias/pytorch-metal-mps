#!/usr/bin/env python3
"""Neural Network benchmark — CSV + PNG output version.

Runs the same inference and training benchmarks as the companion notebook and
writes CSV data and chart images to the ./output directory.
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
import torchvision.models as models

from metalcheck import device_info, is_rocm_available
from metalcheck.utils import benchmark_model_inference, benchmark_model_training

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = REPO_ROOT / "output"

MODEL_CONFIGS = {
    "ResNet-18": models.resnet18,
    "MobileNet V2": models.mobilenet_v2,
}
BATCH_SIZES_INFER = [1, 8, 32, 64]
BATCH_SIZES_TRAIN = [1, 8, 32]
TRAIN_STEPS = 20
ITERATIONS = 10

_log_fd: int | None = None


def trace(msg: str) -> None:
    """Write a timestamped line to the log file using os-level write (unbuffered)."""
    global _log_fd
    if _log_fd is None:
        logfile = OUTDIR / "02_neural_network_benchmark_ryzen_ai_hx370.log"
        _log_fd = os.open(str(logfile), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}\n"
    os.write(_log_fd, line.encode())


def run_inference() -> list[dict]:
    rows: list[dict] = []

    for model_name, factory in MODEL_CONFIGS.items():
        for bs in BATCH_SIZES_INFER:
            input_shape = (bs, 3, 224, 224)
            for dev_name, dev in [("ROCm", torch.device("cuda")), ("CPU", torch.device("cpu"))]:
                if dev_name == "ROCm" and not is_rocm_available():
                    continue
                tag = f"infer {model_name} batch={bs} | {dev_name}"
                trace(f"BEGIN {tag}")
                print(f"  {model_name} | batch={bs} | {dev_name} ...", end=" ", flush=True, file=sys.stderr)
                r = benchmark_model_inference(factory, input_shape, device=dev, iterations=ITERATIONS)
                time_ms = r["mean_time_s"] * 1000
                trace(f"END   {tag} -> {time_ms:.4f} ms")
                rows.append({
                    "Model": model_name,
                    "Batch": bs,
                    "Device": dev_name,
                    "Time (ms)": f"{time_ms:.4f}",
                })
                print(f"{time_ms:.2f} ms", file=sys.stderr)

    trace("writing inference CSV")
    csv_out = OUTDIR / "02_nn_inference.csv"
    fieldnames = ["Model", "Batch", "Device", "Time (ms)"]
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Inference results written to {csv_out}", file=sys.stderr)
    return rows


def plot_inference(rows: list[dict]) -> None:
    trace("generating inference chart")
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, len(MODEL_CONFIGS), figsize=(14, 5), sharey=False)

    for ax, model_name in zip(axes, MODEL_CONFIGS):
        model_rows = [r for r in rows if r["Model"] == model_name]
        x = range(len(BATCH_SIZES_INFER))
        width = 0.35

        rocm_vals = [float(r["Time (ms)"]) for r in model_rows if r["Device"] == "ROCm"]
        cpu_vals = [float(r["Time (ms)"]) for r in model_rows if r["Device"] == "CPU"]

        if rocm_vals:
            ax.bar([i - width / 2 for i in x], rocm_vals, width, label="ROCm (Radeon 890M)", color="#E4002B")
        ax.bar([i + width / 2 for i in x], cpu_vals, width, label="CPU", color="#4A90D9")

        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"{model_name} Inference")
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(bs) for bs in BATCH_SIZES_INFER])
        ax.legend()

    plt.tight_layout()
    out = OUTDIR / "02_nn_inference.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    trace(f"saved {out}")
    print(f"Chart saved to {out}", file=sys.stderr)


def run_training() -> list[dict]:
    rows: list[dict] = []

    for bs in BATCH_SIZES_TRAIN:
        input_shape = (bs, 3, 224, 224)
        for dev_name, dev in [("ROCm", torch.device("cuda")), ("CPU", torch.device("cpu"))]:
            if dev_name == "ROCm" and not is_rocm_available():
                continue
            tag = f"train ResNet-18 batch={bs} | {dev_name}"
            trace(f"BEGIN {tag}")
            print(f"  ResNet-18 training | batch={bs} | {dev_name} ...", end=" ", flush=True, file=sys.stderr)
            r = benchmark_model_training(models.resnet18, input_shape, num_classes=1000, device=dev, steps=TRAIN_STEPS)
            time_ms = r["mean_time_s"] * 1000
            trace(f"END   {tag} -> {time_ms:.4f} ms")
            rows.append({
                "Batch": bs,
                "Device": dev_name,
                "Time (ms)": f"{time_ms:.4f}",
            })
            print(f"{time_ms:.2f} ms/step", file=sys.stderr)

    trace("writing training CSV")
    csv_out = OUTDIR / "02_nn_training.csv"
    fieldnames = ["Batch", "Device", "Time (ms)"]
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Training results written to {csv_out}", file=sys.stderr)
    return rows


def plot_training(rows: list[dict]) -> None:
    trace("generating training chart")
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(BATCH_SIZES_TRAIN))
    width = 0.35

    rocm_vals = [float(r["Time (ms)"]) for r in rows if r["Device"] == "ROCm"]
    cpu_vals = [float(r["Time (ms)"]) for r in rows if r["Device"] == "CPU"]

    if rocm_vals:
        ax.bar([i - width / 2 for i in x], rocm_vals, width, label="ROCm (Radeon 890M)", color="#E4002B")
    ax.bar([i + width / 2 for i in x], cpu_vals, width, label="CPU", color="#4A90D9")

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Time per Step (ms)")
    ax.set_title("ResNet-18 Training: ROCm vs CPU")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(bs) for bs in BATCH_SIZES_TRAIN])
    ax.legend()
    plt.tight_layout()

    out = OUTDIR / "02_nn_training.png"
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

    print("\n--- Inference Benchmark ---", file=sys.stderr)
    trace("=== INFERENCE PHASE ===")
    infer_rows = run_inference()
    plot_inference(infer_rows)

    print("\n--- Training Benchmark ---", file=sys.stderr)
    trace("=== TRAINING PHASE ===")
    train_rows = run_training()
    plot_training(train_rows)

    trace("done")


if __name__ == "__main__":
    main()
