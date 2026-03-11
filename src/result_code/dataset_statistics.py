#!/usr/bin/env python3
"""
Dataset statistics and plots for data in the data/ folder.

Covers:
- ids-cfimdb: CSV files (id, sentence, sentiment 0/1)
- ids-sst: CSV files (id, sentence, sentiment 0-4)
- quora: CSV files (id, sentence1, sentence2, is_duplicate)
- sonnets: TXT files (numbered sonnet blocks)
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "results" / "dataset_stats_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def file_label(filename: str) -> str:
    """Map filename to one of 'train', 'dev', 'test' for plot labels."""
    n = filename.lower()
    if "train" in n:
        return "train"
    if "dev" in n:
        return "dev"
    if "test" in n or "held_out" in n:
        return "test"
    return "train"


def aggregate_by_split(by_file: dict[str, int]) -> tuple[list[str], list[int]]:
    """Aggregate by_file counts by train/dev/test labels. Returns (names, counts) in train, dev, test order."""
    agg: dict[str, int] = {"train": 0, "dev": 0, "test": 0}
    for name, count in by_file.items():
        label = file_label(name)
        agg[label] = agg.get(label, 0) + count
    names = [k for k in ("train", "dev", "test") if agg.get(k, 0) > 0]
    counts = [agg[k] for k in names]
    return names, counts

def load_cfimdb_csv(path: Path) -> list[dict]:
    """Load a tab-separated cfimdb CSV (id, sentence, sentiment). Handles index column."""
    df = pd.read_csv(path, sep="\t", index_col=0, on_bad_lines="skip")
    return df.to_dict(orient="records")


def stats_cfimdb() -> dict:
    """Aggregate stats and optional plots for all ids-cfimdb CSVs."""
    pattern = "ids-cfimdb*.csv"
    files = sorted(DATA_DIR.glob(pattern))
    if not files:
        return {"error": f"No files matching {pattern} in {DATA_DIR}"}

    all_rows = []
    by_file = {}
    for f in files:
        rows = load_cfimdb_csv(f)
        by_file[f.name] = rows
        all_rows.extend(rows)

    sentiments = []
    for r in all_rows:
        s = r.get("sentiment", "")
        try:
            sentiments.append(int(float(s)))
        except (ValueError, TypeError):
            pass
    sentence_lens = [len((r.get("sentence") or "").split()) for r in all_rows]

    stats = {
        "dataset": "ids-cfimdb",
        "files": [f.name for f in files],
        "total_examples": len(all_rows),
        "by_file": {f.name: len(by_file[f.name]) for f in files},
        "sentiment_counts": dict(Counter(sentiments)) if sentiments else {},
        "sentence_length": {
            "min": int(min(sentence_lens)) if sentence_lens else 0,
            "max": int(max(sentence_lens)) if sentence_lens else 0,
            "mean": float(np.mean(sentence_lens)) if sentence_lens else 0.0,
            "median": float(np.median(sentence_lens)) if sentence_lens else 0.0,
        },
    }

    df = pd.DataFrame(all_rows)
    if "sentiment" in df.columns:
        df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce").dropna().astype(int)
    df["word_count"] = df["sentence"].fillna("").str.split().str.len()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("CFIMDB Dataset Statistics", fontsize=12)

    ax = axes[0]
    names, counts = aggregate_by_split(stats["by_file"])
    ax.bar(names, counts, color="steelblue", edgecolor="black")
    ax.set_title("Count per Split")
    ax.set_ylabel("Count")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax = axes[1]
    if "sentiment_counts" in stats and stats["sentiment_counts"]:
        s = stats["sentiment_counts"]
        keys = sorted(s.keys())
        ax.bar([str(k) for k in keys], [s[k] for k in keys], color="coral", edgecolor="black")
        ax.set_title("Sentiment Distribution Labels (Train and Dev)")
        ax.set_ylabel("Count")

    plt.tight_layout()
    out = OUTPUT_DIR / "ids_cfimdb_stats.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    stats["plot_path"] = str(out)

    return stats

def load_sst_csv(path: Path) -> list[dict]:
    """Load a tab-separated SST CSV (id, sentence, sentiment)."""
    return load_cfimdb_csv(path)  # same format


def stats_sst() -> dict:
    """Aggregate stats and optional plots for all ids-sst CSVs."""
    pattern = "ids-sst*.csv"
    files = sorted(DATA_DIR.glob(pattern))
    if not files:
        return {"error": f"No files matching {pattern} in {DATA_DIR}"}

    all_rows = []
    by_file = {}
    for f in files:
        rows = load_sst_csv(f)
        by_file[f.name] = rows
        all_rows.extend(rows)

    sentiments = []
    for r in all_rows:
        s = r.get("sentiment", "")
        try:
            sentiments.append(int(float(s)))
        except (ValueError, TypeError):
            pass
    sentence_lens = [len((r.get("sentence") or "").split()) for r in all_rows]

    stats = {
        "dataset": "ids-sst",
        "files": [f.name for f in files],
        "total_examples": len(all_rows),
        "by_file": {f.name: len(by_file[f.name]) for f in files},
        "sentiment_counts": dict(Counter(sentiments)) if sentiments else {},
        "sentence_length": {
            "min": int(min(sentence_lens)) if sentence_lens else 0,
            "max": int(max(sentence_lens)) if sentence_lens else 0,
            "mean": float(np.mean(sentence_lens)) if sentence_lens else 0.0,
            "median": float(np.median(sentence_lens)) if sentence_lens else 0.0,
        },
    }

    df = pd.DataFrame(all_rows)
    if "sentiment" in df.columns:
        df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce").dropna().astype(int)
    df["word_count"] = df["sentence"].fillna("").str.split().str.len()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Stanford Sentiment Treebank Dataset Statistics", fontsize=12)

    ax = axes[0]
    names, counts = aggregate_by_split(stats["by_file"])
    ax.bar(names, counts, color="steelblue", edgecolor="black")
    ax.set_title("Count per Split")
    ax.set_ylabel("Count")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax = axes[1]
    if stats["sentiment_counts"]:
        s = stats["sentiment_counts"]
        ax.bar([str(k) for k in sorted(s.keys())], [s[k] for k in sorted(s.keys())], color="coral", edgecolor="black")
        ax.set_title("Sentiment Distribution Labels (Train and Dev)")
        ax.set_ylabel("Count")

    plt.tight_layout()
    out = OUTPUT_DIR / "ids_sst_stats.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    stats["plot_path"] = str(out)

    return stats

def load_quora_csv(path: Path) -> list[dict]:
    """Load a tab-separated Quora CSV (id, sentence1, sentence2, is_duplicate)."""
    df = pd.read_csv(path, sep="\t", dtype=str, on_bad_lines="skip", low_memory=False)
    return df.to_dict(orient="records")


def stats_quora() -> dict:
    """Aggregate stats and optional plots for all quora CSVs."""
    pattern = "quora*.csv"
    files = sorted(DATA_DIR.glob(pattern))
    if not files:
        return {"error": f"No files matching {pattern} in {DATA_DIR}"}

    all_rows = []
    by_file = {}
    for f in files:
        rows = load_quora_csv(f)
        by_file[f.name] = rows
        all_rows.extend(rows)

    dup = []
    len1 = []
    len2 = []
    for r in all_rows:
        try:
            val = int(float(r.get("is_duplicate", -1)))
            if val in (0, 1):
                dup.append(val)
        except (ValueError, TypeError):
            pass
        s1_raw = r.get("sentence1", "")
        s2_raw = r.get("sentence2", "")
        s1 = (s1_raw if isinstance(s1_raw, str) else "").split()
        s2 = (s2_raw if isinstance(s2_raw, str) else "").split()
        len1.append(len(s1))
        len2.append(len(s2))

    stats = {
        "dataset": "quora",
        "files": [f.name for f in files],
        "total_examples": len(all_rows),
        "by_file": {f.name: len(by_file[f.name]) for f in files},
        "is_duplicate_counts": {k: v for k, v in (Counter(dup).items() if dup else []) if k in (0, 1)},
        "sentence1_length": {
            "min": int(min(len1)) if len1 else 0,
            "max": int(max(len1)) if len1 else 0,
            "mean": float(np.mean(len1)) if len1 else 0.0,
            "median": float(np.median(len1)) if len1 else 0.0,
        },
        "sentence2_length": {
            "min": int(min(len2)) if len2 else 0,
            "max": int(max(len2)) if len2 else 0,
            "mean": float(np.mean(len2)) if len2 else 0.0,
            "median": float(np.median(len2)) if len2 else 0.0,
        },
    }

    df = pd.DataFrame(all_rows)
    df["s1_len"] = df["sentence1"].fillna("").str.split().str.len()
    df["s2_len"] = df["sentence2"].fillna("").str.split().str.len()
    df["is_duplicate"] = pd.to_numeric(df["is_duplicate"], errors="coerce")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Quora Dataset Statistics", fontsize=12)

    ax = axes[0]
    names, counts = aggregate_by_split(stats["by_file"])
    ax.bar(names, counts, color="steelblue", edgecolor="black")
    ax.set_title("Count per Split")
    ax.set_ylabel("Count")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax = axes[1]
    s = {k: stats["is_duplicate_counts"].get(k, 0) for k in (0, 1)}
    if any(s.values()):
        ax.bar([str(k) for k in sorted(s.keys())], [s[k] for k in sorted(s.keys())], color="coral", edgecolor="black")
        ax.set_title("Is Duplicate Labels (Train and Dev)")
        ax.set_ylabel("Count")

    plt.tight_layout()
    out = OUTPUT_DIR / "quora_stats.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    stats["plot_path"] = str(out)

    return stats

def parse_sonnets_txt(path: Path) -> list[list[str]]:
    """
    Parse a sonnets TXT file into a list of sonnets.
    Each sonnet is a list of lines (verse lines only).
    Sonnets are separated by a line that is only digits (sonnet number).
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [line.rstrip() for line in f]

    sonnets = []
    current = []
    number_line = re.compile(r"^(?:\d+|#+\d+#+)$")

    for line in lines:
        if number_line.match(line.strip()):
            if current:
                sonnets.append(current)
            current = []
        else:
            stripped = line.strip()
            if stripped and not stripped.startswith("http") and "Folger" not in stripped and "shakespeare" not in stripped.lower():
                current.append(stripped)
    if current:
        sonnets.append(current)

    return sonnets


def stats_sonnets() -> dict:
    """Aggregate stats and optional plots for all sonnets TXT files."""
    pattern = "*.txt"
    files = sorted(DATA_DIR.glob(pattern))
    sonnet_files = [f for f in files if "sonnet" in f.name.lower() or f.name == "sonnets.txt"]
    if not sonnet_files:
        return {"error": f"No sonnet TXT files found in {DATA_DIR}"}

    all_sonnets = []
    by_file = {}
    for f in sonnet_files:
        sonnets = parse_sonnets_txt(f)
        by_file[f.name] = sonnets
        all_sonnets.extend(sonnets)

    lines_per_sonnet = [len(s) for s in all_sonnets]
    words_per_sonnet = [sum(len(l.split()) for l in s) for s in all_sonnets]
    words_per_line = [len(l.split()) for s in all_sonnets for l in s]

    stats = {
        "dataset": "sonnets",
        "files": [f.name for f in sonnet_files],
        "total_sonnets": len(all_sonnets),
        "by_file": {f.name: len(by_file[f.name]) for f in sonnet_files},
        "lines_per_sonnet": {
            "min": int(min(lines_per_sonnet)) if lines_per_sonnet else 0,
            "max": int(max(lines_per_sonnet)) if lines_per_sonnet else 0,
            "mean": float(np.mean(lines_per_sonnet)) if lines_per_sonnet else 0.0,
            "median": float(np.median(lines_per_sonnet)) if lines_per_sonnet else 0.0,
        },
        "words_per_sonnet": {
            "min": int(min(words_per_sonnet)) if words_per_sonnet else 0,
            "max": int(max(words_per_sonnet)) if words_per_sonnet else 0,
            "mean": float(np.mean(words_per_sonnet)) if words_per_sonnet else 0.0,
            "median": float(np.median(words_per_sonnet)) if words_per_sonnet else 0.0,
        },
        "words_per_line": {
            "min": int(min(words_per_line)) if words_per_line else 0,
            "max": int(max(words_per_line)) if words_per_line else 0,
            "mean": float(np.mean(words_per_line)) if words_per_line else 0.0,
            "median": float(np.median(words_per_line)) if words_per_line else 0.0,
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Sonnet Dataset Statistics", fontsize=12)

    ax = axes[0]
    names, counts = aggregate_by_split(stats["by_file"])
    ax.bar(names, counts, color="steelblue", edgecolor="black")
    ax.set_title("Count per Split")
    ax.set_ylabel("Count")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax = axes[1]
    names_syn, counts_syn = aggregate_by_split(stats["by_file"])
    ax.bar(names_syn, counts_syn, color="steelblue", edgecolor="black")
    synthetic_path = DATA_DIR.parent / "synthetic_data" / "synthetic_sonnets.txt"
    if synthetic_path.exists():
        syn_count = len(parse_sonnets_txt(synthetic_path))
        train_idx = next((i for i, n in enumerate(names_syn) if n == "train"), None)
        if train_idx is not None and syn_count > 0:
            ax.bar([names_syn[train_idx]], [syn_count], bottom=[counts_syn[train_idx]], color="orange", edgecolor="black")
    ax.set_title("Count per Split (with Synthetic Data)")
    ax.set_ylabel("Count")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.legend(
        handles=[
            mpatches.Patch(facecolor="steelblue", edgecolor="black", label="Default Data"),
            mpatches.Patch(facecolor="orange", edgecolor="black", label="Synthetic Data"),
        ]
    )

    plt.tight_layout()
    out = OUTPUT_DIR / "sonnets_stats.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    stats["plot_path"] = str(out)

    return stats

def print_stats(stats: dict) -> None:
    """Pretty-print a stats dict."""
    if "error" in stats:
        print(stats["error"])
        return
    print(f"\n{'='*60}")
    print(f"Dataset: {stats.get('dataset', '?')}")
    print("=" * 60)
    print("Files:", stats.get("files", []))
    print("Total examples/sonnets:", stats.get("total_examples", stats.get("total_sonnets", "?")))
    print("By file:", stats.get("by_file", {}))
    for k in ("sentiment_counts", "is_duplicate_counts"):
        if k in stats and stats[k]:
            print(f"{k}:", stats[k])
    for k in ("sentence_length", "sentence1_length", "sentence2_length", "lines_per_sonnet", "words_per_sonnet", "words_per_line"):
        if k in stats and stats[k]:
            print(f"{k}:", stats[k])
    if "plot_path" in stats:
        print("Plot saved:", stats["plot_path"])


def main() -> None:
    all_stats = []
    for name, fn in [
        ("ids-cfimdb", stats_cfimdb),
        ("ids-sst", stats_sst),
        ("quora", stats_quora),
        ("sonnets", stats_sonnets),
    ]:
        try:
            s = fn()
            all_stats.append((name, s))
            print_stats(s)
        except Exception as e:
            print(f"\nError processing {name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
