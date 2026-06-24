"""Experiment 26 figures: layer-0 injection vs EH adapter vs static."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = Path(__file__).resolve().parent

COLORS = {
    "static": "grey",
    "adaptive_eh": "steelblue",
    "adaptive_layer0": "darkorange",
}
LABELS = {
    "static": "Static HNSW",
    "adaptive_eh": "Adaptive (EH, conjugate)",
    "adaptive_layer0": "Adaptive (layer-0 injection)",
}
SCHEDULES = ["gradual", "sudden"]


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _vlines(ax, transitions):
    for e in transitions:
        ax.axvline(e, color="black", linestyle="--", linewidth=0.8, alpha=0.45)


def plot_figure1(df, cfg, plots_dir):
    """Recall@10 vs epoch — 2 rows (schedules) × len(ef_values) cols."""
    ef_values = cfg["ef_values"]
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(2, len(ef_values),
                             figsize=(5 * len(ef_values), 9), sharey=True)

    for row_idx, sched in enumerate(SCHEDULES):
        transitions = cfg[f"drift_transition_epochs_{sched}"]
        sdf = df[df["schedule"] == sched]
        for col_idx, ef in enumerate(ef_values):
            ax = axes[row_idx][col_idx]
            _vlines(ax, transitions)
            for cond in ("static", "adaptive_eh", "adaptive_layer0"):
                sub = sdf[(sdf["condition"] == cond) & (sdf["ef"] == ef)].sort_values("epoch")
                if sub.empty:
                    continue
                ax.plot(sub["epoch"], sub["recall"], color=COLORS[cond],
                        label=LABELS[cond], linewidth=2.0, marker="o", markersize=3.5)
            ax.set_title(f"{sched}, ef={ef}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel("Recall@10", fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8, loc="lower left", framealpha=0.85)

    fig.suptitle("Recall@10 vs Epoch — Layer-0 Injection vs EH vs Static",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = plots_dir / "recall_vs_epoch.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_figure2(df, cfg, plots_dir):
    """Edge count vs epoch — one subplot per schedule."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    ef0 = cfg["ef_values"][0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, sched in zip(axes, SCHEDULES):
        transitions = cfg[f"drift_transition_epochs_{sched}"]
        _vlines(ax, transitions)
        sdf = df[(df["schedule"] == sched) & (df["ef"] == ef0)]
        for cond in ("adaptive_eh", "adaptive_layer0"):
            sub = sdf[sdf["condition"] == cond].sort_values("epoch")
            if sub.empty:
                continue
            ax.plot(sub["epoch"], sub["edge_count"], color=COLORS[cond],
                    label=LABELS[cond], linewidth=2.0, marker="o", markersize=3.5)
        ax.set_title(sched, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Edges injected (cumulative)", fontsize=10)
        ax.legend(fontsize=9, framealpha=0.85)

    fig.suptitle("Edge Count vs Epoch", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = plots_dir / "edge_count_vs_epoch.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else _SCRIPT_DIR / "config.yaml"
    cfg = _load_config(config_path)

    df = pd.read_csv(ROOT / cfg["results_path"])
    plots_dir = _SCRIPT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_figure1(df, cfg, plots_dir)
    plot_figure2(df, cfg, plots_dir)


if __name__ == "__main__":
    main()
