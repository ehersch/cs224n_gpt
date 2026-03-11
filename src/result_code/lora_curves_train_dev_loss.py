#!/usr/bin/env python3
"""
Plot training loss (left) and dev loss (right) vs learning rate for batch size 16.
Combined figure with one row of two subplots; legend on the right, not overlaying either plot.
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SWEEP_PATH = REPO_ROOT / "predictions" / "lora" / "sweep_results.json"
RESULTS_DIR = REPO_ROOT / "results"


def main():
    with open(SWEEP_PATH, "r") as f:
        results = json.load(f)

    batch16 = [
        r for r in results
        if r.get("batch_size") == 16 and "train_nll" in r and "dev_nll" in r and "alpha" not in r and 'layer' not in r
    ]

    numeric = [r for r in batch16 if isinstance(r["lora_rank"], (int, float))]
    full_runs = [r for r in batch16 if r.get("lora_rank") == "full"]

    def final_train_nll(r):
        tnll = r["train_nll"]
        return tnll[-1] if isinstance(tnll, list) and tnll else None

    def dev_nll(r):
        return r.get("dev_nll")

    ranks = sorted(set(r["lora_rank"] for r in numeric))
    n_ranks = len(ranks)
    cmap = plt.get_cmap("viridis")
    colors = {rank: cmap(i / max(n_ranks - 1, 1)) for i, rank in enumerate(ranks)}

    fig, (ax_train, ax_dev) = plt.subplots(1, 2, figsize=(10, 4.5))
    legend_handles = []
    legend_labels = []

    for rank in ranks:
        points = [
            (r["lr"], final_train_nll(r))
            for r in numeric
            if r["lora_rank"] == rank and final_train_nll(r) is not None
        ]
        points.sort(key=lambda x: x[0])
        if not points:
            continue
        lrs, losses = zip(*points)
        (h,) = ax_train.plot(lrs, losses, "o-", color=colors[rank], label=f"rank {rank}", markersize=2)
        legend_handles.append(h)
        legend_labels.append(f"rank {rank}")

    if full_runs:
        points = [
            (r["lr"], final_train_nll(r))
            for r in full_runs
            if final_train_nll(r) is not None
        ]
        points.sort(key=lambda x: x[0])
        if points:
            lrs, losses = zip(*points)
            (h,) = ax_train.plot(lrs, losses, "s-", color="red", label="full", markersize=2, linewidth=2)
            legend_handles.append(h)
            legend_labels.append("full")

    ax_train.set_xlabel("Learning rate")
    ax_train.set_ylabel("Train loss (NLL)")
    ax_train.set_title("Train Loss vs Learning Rate")
    ax_train.set_xscale("log")

    for rank in ranks:
        points = [
            (r["lr"], dev_nll(r))
            for r in numeric
            if r["lora_rank"] == rank and dev_nll(r) is not None
        ]
        points.sort(key=lambda x: x[0])
        if not points:
            continue
        lrs, losses = zip(*points)
        ax_dev.plot(lrs, losses, "o-", color=colors[rank], markersize=2)

    if full_runs:
        points = [
            (r["lr"], dev_nll(r))
            for r in full_runs
            if dev_nll(r) is not None
        ]
        points.sort(key=lambda x: x[0])
        if points:
            lrs, losses = zip(*points)
            ax_dev.plot(lrs, losses, "s-", color="red", markersize=2, linewidth=2)

    ax_dev.set_xlabel("Learning rate")
    ax_dev.set_ylabel("Dev loss (NLL)")
    ax_dev.set_title("Dev Loss vs Learning Rate")
    ax_dev.set_xscale("log")

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    fig.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.89, 0.5),
        fontsize="small",
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "lora_curves_train_dev_loss.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
