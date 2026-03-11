#!/usr/bin/env python3
"""
Plot dev chrF vs learning rate for batch size 16 from sweep_results.json.
LoRA ranks use viridis colormap; full fine-tuning is orange.
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

    batch8 = [r for r in results if r.get("batch_size") == 16 and "dev_chrf" in r and "alpha" not in r and 'layer' not in r]
    numeric = [r for r in batch8 if isinstance(r["lora_rank"], (int, float))]
    full_runs = [r for r in batch8 if r.get("lora_rank") == "full"]

    ranks = sorted(set(r["lora_rank"] for r in numeric))
    n_ranks = len(ranks)
    cmap = plt.get_cmap("viridis")
    colors = {rank: cmap(i / max(n_ranks - 1, 1)) for i, rank in enumerate(ranks)}

    print("Best dev chrF (batch size 16):")
    for rank in ranks:
        points = [(r["lr"], r["dev_chrf"]) for r in numeric if r["lora_rank"] == rank]
        if not points:
            continue
        best_lr, best_chrf = max(points, key=lambda x: x[1])
        print(f"  rank {rank}: {best_chrf:.4f} (lr={best_lr})")
    if full_runs:
        points = [(r["lr"], r["dev_chrf"]) for r in full_runs]
        best_lr, best_chrf = max(points, key=lambda x: x[1])
        print(f"  full: {best_chrf:.4f} (lr={best_lr})")

    fig, ax = plt.subplots()

    for rank in ranks:
        points = [(r["lr"], r["dev_chrf"]) for r in numeric if r["lora_rank"] == rank]
        points.sort(key=lambda x: x[0])
        if not points:
            continue
        lrs, chrfs = zip(*points)
        ax.plot(lrs, chrfs, "o-", color=colors[rank], label=f"rank {rank}", markersize=2)

    if full_runs:
        points = [(r["lr"], r["dev_chrf"]) for r in full_runs]
        points.sort(key=lambda x: x[0])
        lrs, chrfs = zip(*points)
        ax.plot(lrs, chrfs, "s-", color="red", label="full", markersize=2, linewidth=2)

    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Dev chrF")
    ax.set_title("Dev chrF vs Learning Rate")
    ax.legend(loc="best", fontsize="small")
    ax.set_xscale("log")
    plt.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "lora_curves_chrf.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
