#!/usr/bin/env python3
"""
Print dev chrF scores from sweep_results.json for entries that have a 'layers' field.
"""

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SWEEP_PATH = REPO_ROOT / "predictions" / "lora" / "sweep_results.json"


def main():
    with open(SWEEP_PATH, "r") as f:
        results = json.load(f)

    with_layers = [r for r in results if "layers" in r]

    if not with_layers:
        print("No entries with 'layers' in sweep_results.json")
        return

    print("Dev chrF for entries with 'layers':")
    for r in with_layers:
        chrf = r.get("dev_chrf")
        layers = r.get("layers")
        # Print config for identification
        parts = [f"layers={layers}"]
        if "lora_rank" in r:
            parts.append(f"rank={r['lora_rank']}")
        if "lr" in r:
            parts.append(f"lr={r['lr']}")
        if "batch_size" in r:
            parts.append(f"batch_size={r['batch_size']}")
        config = ", ".join(parts)
        if chrf is not None:
            print(f"  {config}: dev_chrf = {chrf:.4f}")
        else:
            print(f"  {config}: dev_chrf = (missing)")


if __name__ == "__main__":
    main()
