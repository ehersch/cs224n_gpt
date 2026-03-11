import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PLOTS_DIR = REPO_ROOT / "results" / "plots"


def _ensure_output_dir() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _apply_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "semibold",
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "font.size": 11,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#d0d0d0",
            "grid.color": "#d9d9d9",
            "grid.alpha": 0.6,
        }
    )


def make_zero_shot_degradation_plot() -> Path:
    labels = [
        "Sonnet Gen\nFull FT",
        "Paraphrase\nZero-Shot",
        "Paraphrase\nFull FT",
        "Sonnet Gen\nQAFT INT8",
        "Paraphrase\nZero-Shot\nfrom QAFT INT8",
    ]
    values = [42.224, 0.6317, 0.884, 41.58872727241384, 0.3685]
    colors = ["#295c77", "#9b2226", "#4d908e", "#577590", "#c1666b"]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    bars = ax.bar(labels, values, color=colors, width=0.68)
    ax.set_title(
        "Zero-Shot Degradation / Task Overfitting After Quantization-Aware Fine-Tuning"
    )
    ax.set_ylabel("Task Score")
    ax.set_ylim(0, max(values) * 1.18)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.018,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#222222",
        )

    ax.text(
        0.02,
        0.96,
        "Sonnet generation uses chrF.\nParaphrase detection uses dev accuracy.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#555555",
        bbox={
            "facecolor": "white",
            "edgecolor": "#dddddd",
            "boxstyle": "round,pad=0.35",
        },
    )
    fig.tight_layout()

    out = PLOTS_DIR / "zero_shot_degradation_qaft_gpt2.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def make_qaft_vs_inference_plot() -> Path:
    inference_only = {
        "FP64": {"size": 953.99, "chrf": 41.818, "tps": np.nan},
        "FP32": {"size": 477.03, "chrf": 42.118, "tps": 103.0},
        "BF16": {"size": 238.56, "chrf": 42.069, "tps": 113.0},
        "FP8": {"size": 232.69, "chrf": 41.73, "tps": 61.0},
        "INT8": {"size": 119.24, "chrf": 41.49, "tps": 100.486},
        "INT4": {"size": 59.62, "chrf": 31.277, "tps": 103.734},
    }
    qaft = {
        "INT8": {"size": 119.24, "chrf": 41.58872727241384, "tps": np.nan},
        "INT4": {"size": 59.62, "chrf": 41.1618, "tps": np.nan},
    }

    order = sorted(
        inference_only.keys(), key=lambda k: inference_only[k]["size"], reverse=True
    )
    xs = [inference_only[k]["size"] for k in order]
    labels = order
    infer_chrf = [inference_only[k]["chrf"] for k in order]
    infer_tps = [inference_only[k]["tps"] for k in order]
    qaft_x = [qaft[k]["size"] for k in ["INT8", "INT4"]]
    qaft_chrf = [qaft[k]["chrf"] for k in ["INT8", "INT4"]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9.5), sharex=True)

    ax1.plot(
        xs,
        infer_chrf,
        marker="o",
        linewidth=2.4,
        markersize=7,
        color="#2a6f97",
        label="Inference-Only Quantization",
    )
    ax1.plot(
        qaft_x,
        qaft_chrf,
        marker="D",
        linewidth=2.4,
        markersize=7,
        linestyle="--",
        color="#c44536",
        label="QAFT",
    )
    for x, y, label in zip(xs, infer_chrf, labels):
        ax1.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
            color="#1f1f1f",
        )
    ax1.set_ylabel("CHRF Score")
    ax1.set_title(
        "Sonnet Generation Performance vs Model Size (QAFT and Inference-Only Quantization)"
    )
    ax1.legend(loc="lower right")

    ax2.plot(
        xs,
        infer_tps,
        marker="o",
        linewidth=2.4,
        markersize=7,
        color="#588157",
        label="Inference-Only Quantization",
    )
    for x, y, label in zip(xs, infer_tps, labels):
        if np.isfinite(y):
            ax2.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
                color="#1f1f1f",
            )
    ax2.text(
        0.99,
        0.05,
        "QAFT throughput points pending measurement",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="#666666",
    )
    ax2.set_xlabel("Model Size (MB)")
    ax2.set_ylabel("Tokens / sec")
    ax2.legend(loc="lower right")
    ax2.set_xlim(max(xs) * 1.03, min(xs) * 0.97)

    fig.tight_layout()
    out = PLOTS_DIR / "qaft_vs_inference_quant_sonnet_2panel.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def make_synthetic_data_sonnet_plot() -> Path:
    baseline_chrf = 42.243
    examples = [100, 500, 1000]
    flash = [41.5758, 43.1197, 46.605]
    flash_lite = [41.89, 41.6656, 42.0915]

    flash_cost_per_1k = 0.11
    flash_lite_cost_per_1k = 0.015
    flash_delta = [score - baseline_chrf for score in flash]
    flash_lite_delta = [score - baseline_chrf for score in flash_lite]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10.8, 9.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.15]},
    )

    ax1.axhline(
        baseline_chrf,
        color="#6c757d",
        linestyle=":",
        linewidth=2.0,
        label=f"Baseline SFT ({baseline_chrf:.3f})",
    )
    ax1.plot(
        examples,
        flash,
        marker="o",
        linewidth=2.6,
        markersize=8,
        color="#bc4749",
        label=f"Gemini 2.5 Flash ($ {flash_cost_per_1k:.3f} / 1k)",
    )
    ax1.plot(
        examples,
        flash_lite,
        marker="s",
        linewidth=2.6,
        markersize=7,
        color="#386641",
        label=f"Gemini 2.5 Flash Lite ($ {flash_lite_cost_per_1k:.3f} / 1k)",
    )
    for x, y in zip(examples, flash):
        ax1.annotate(
            f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=10,
            color="#222222",
        )
    for x, y in zip(examples, flash_lite):
        ax1.annotate(
            f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, -18),
            ha="center",
            fontsize=10,
            color="#222222",
        )
    ax1.set_ylabel("Dev chrF")
    ax1.set_title("Synthetic Sonnet Data Quality vs Scale")
    ax1.legend(loc="upper left")
    ax1.text(
        0.98,
        0.06,
        "Higher-quality synthetic data helps only at larger scale.\n"
        "Flash Lite stays near baseline despite much lower cost.",
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="#555555",
        bbox={
            "facecolor": "white",
            "edgecolor": "#dddddd",
            "boxstyle": "round,pad=0.35",
        },
    )

    width = 72
    x = np.array(examples, dtype=float)
    ax2.bar(
        x - width / 2,
        flash_delta,
        width=width,
        color="#f28482",
        label="Flash improvement over baseline",
    )
    ax2.bar(
        x + width / 2,
        flash_lite_delta,
        width=width,
        color="#84a98c",
        label="Flash Lite improvement over baseline",
    )
    ax2.axhline(0.0, color="#444444", linewidth=1.3)
    for x_pos, delta in zip(x - width / 2, flash_delta):
        ax2.annotate(
            f"{delta:+.2f}",
            (x_pos, delta),
            textcoords="offset points",
            xytext=(0, 6 if delta >= 0 else -16),
            ha="center",
            fontsize=9,
        )
    for x_pos, delta in zip(x + width / 2, flash_lite_delta):
        ax2.annotate(
            f"{delta:+.2f}",
            (x_pos, delta),
            textcoords="offset points",
            xytext=(0, 6 if delta >= 0 else -16),
            ha="center",
            fontsize=9,
        )
    ax2.set_xlabel("Number of Synthetic Sonnets")
    ax2.set_ylabel("Delta vs Baseline")
    ax2.legend(loc="upper left")
    ax2.set_xticks(examples)

    fig.tight_layout()
    out = PLOTS_DIR / "synthetic_data_sonnet_quality_vs_scale.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    _ensure_output_dir()
    _apply_style()
    out1 = make_zero_shot_degradation_plot()
    out2 = make_qaft_vs_inference_plot()
    out3 = make_synthetic_data_sonnet_plot()
    print(out1)
    print(out2)
    print(out3)


if __name__ == "__main__":
    main()
