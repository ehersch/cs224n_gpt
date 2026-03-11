#!/usr/bin/env python3
"""
Plot dev loss vs learning rate for batch size 16 from sweep_results.json.
Uses dev NLL per run. LoRA ranks use viridis colormap; full is red.
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

    batch16 = [r for r in results if r.get("batch_size") == 16 and "dev_nll" in r and "alpha" not in r and 'layer' not in r]

    numeric = [r for r in batch16 if isinstance(r["lora_rank"], (int, float))]
    full_runs = [r for r in batch16 if r.get("lora_rank") == "full"]

    def dev_nll(r):
        return r.get("dev_nll")

    ranks = sorted(set(r["lora_rank"] for r in numeric))
    n_ranks = len(ranks)
    cmap = plt.get_cmap("viridis")
    colors = {rank: cmap(i / max(n_ranks - 1, 1)) for i, rank in enumerate(ranks)}

    fig, ax = plt.subplots()

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
        ax.plot(lrs, losses, "o-", color=colors[rank], label=f"rank {rank}", markersize=6)

    if full_runs:
        points = [
            (r["lr"], dev_nll(r))
            for r in full_runs
            if dev_nll(r) is not None
        ]
        points.sort(key=lambda x: x[0])
        if points:
            lrs, losses = zip(*points)
            ax.plot(lrs, losses, "s-", color="red", label="full", markersize=6, linewidth=2)

    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Dev loss (NLL)")
    ax.set_title("Dev loss vs learning rate (batch size 16)")
    ax.legend(loc="best", fontsize="small")
    ax.set_xscale("log")
    plt.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "lora_curves_dev_loss.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
