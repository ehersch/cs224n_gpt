#!/usr/bin/env python3
"""
Plot training loss vs learning rate for batch size 16 from sweep_results.json.
Uses final-epoch train NLL per run. LoRA ranks use viridis colormap; full is red.
"""

import json
import os
import matplotlib.pyplot as plt

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sweep_path = os.path.join(script_dir, "..", "sweep_results.json")
    with open(sweep_path, "r") as f:
        results = json.load(f)

    batch16 = [r for r in results if r.get("batch_size") == 16 and "train_nll" in r and "alpha" not in r and 'layer' not in r]

    numeric = [r for r in batch16 if isinstance(r["lora_rank"], (int, float))]
    full_runs = [r for r in batch16 if r.get("lora_rank") == "full"]

    def final_train_nll(r):
        tnll = r["train_nll"]
        return tnll[-1] if isinstance(tnll, list) and tnll else None
    ranks = sorted(set(r["lora_rank"] for r in numeric))
    n_ranks = len(ranks)
    cmap = plt.get_cmap("viridis")
    colors = {rank: cmap(i / max(n_ranks - 1, 1)) for i, rank in enumerate(ranks)}

    fig, ax = plt.subplots()

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
        ax.plot(lrs, losses, "o-", color=colors[rank], label=f"rank {rank}", markersize=6)

    if full_runs:
        points = [
            (r["lr"], final_train_nll(r))
            for r in full_runs
            if final_train_nll(r) is not None
        ]
        points.sort(key=lambda x: x[0])
        if points:
            lrs, losses = zip(*points)
            ax.plot(lrs, losses, "s-", color="red", label="full", markersize=6, linewidth=2)

    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Training loss (final epoch NLL)")
    ax.set_title("Training loss vs learning rate (batch size 16)")
    ax.legend(loc="best", fontsize="small")
    ax.set_xscale("log")
    plt.tight_layout()
    out_path = os.path.join(script_dir, "lora_curves_train_loss.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
