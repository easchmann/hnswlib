"""Experiment 25 figures: NAF vs EH adapter vs static."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = Path(__file__).resolve().parent

COLORS = {
    "static": "grey",
    "adaptive_eh": "steelblue",
    "adaptive_naf": "darkorange",
    "adaptive_naf_exact": "forestgreen",
}
LABELS = {
    "static": "Static HNSW",
    "adaptive_eh": "Adaptive (EH)",
    "adaptive_naf": "Adaptive (NAF, HNSW pool)",
    "adaptive_naf_exact": "Adaptive (NAF, exact pool)",
}


def _load_config():
    with open(_SCRIPT_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def _vlines(ax, transitions):
    for e in transitions:
        ax.axvline(e, color="black", linestyle="--", linewidth=0.8, alpha=0.45)


def plot_figure1(results, cfg, plots_dir):
    """Recall@10 vs epoch, one subplot per ef value."""
    ef_values = cfg["ef_values"]
    transitions = cfg["drift_transition_epochs"]
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(1, len(ef_values), figsize=(5 * len(ef_values), 5), sharey=True)
    if len(ef_values) == 1:
        axes = [axes]

    for ax, ef in zip(axes, ef_values):
        _vlines(ax, transitions)
        for cond in ("static", "adaptive_eh", "adaptive_naf", "adaptive_naf_exact"):
            rows = sorted(
                [r for r in results if r["condition"] == cond and r["ef"] == ef],
                key=lambda r: r["epoch"],
            )
            if not rows:
                continue
            epochs = [r["epoch"] for r in rows]
            recalls = [r["recall"] for r in rows]
            ax.plot(epochs, recalls, color=COLORS[cond], label=LABELS[cond],
                    linewidth=2.0, marker="o", markersize=3.5)
        ax.set_title(f"ef={ef}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Recall@10", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-0.5, max(r["epoch"] for r in results) + 0.5)
        ax.legend(fontsize=9, loc="lower left", framealpha=0.85)

    fig.suptitle("Recall@10 vs Epoch — NAF vs EH vs Static", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = plots_dir / "recall_vs_epoch.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_figure2(results, cfg, plots_dir):
    """Conjugate edge count vs epoch for both adaptive conditions."""
    transitions = cfg["drift_transition_epochs"]
    ef0 = cfg["ef_values"][0]
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(9, 5))
    _vlines(ax, transitions)

    for cond in ("adaptive_eh", "adaptive_naf", "adaptive_naf_exact"):
        seen = {}
        for r in results:
            if r["condition"] == cond and r["ef"] == ef0:
                seen[r["epoch"]] = r["edge_count"]
        epochs = sorted(seen.keys())
        counts = [seen[e] for e in epochs]
        ax.plot(epochs, counts, color=COLORS[cond], label=LABELS[cond],
                linewidth=2.0, marker="o", markersize=3.5)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Conjugate edge count", fontsize=11)
    ax.set_title("Edge Count vs Epoch", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.85)
    fig.tight_layout()
    out = plots_dir / "edge_count_vs_epoch.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    cfg = _load_config()
    results_path = ROOT / cfg["results_path"]
    with open(results_path) as f:
        results = json.load(f)

    plots_dir = _SCRIPT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_figure1(results, cfg, plots_dir)
    plot_figure2(results, cfg, plots_dir)


if __name__ == "__main__":
    main()
