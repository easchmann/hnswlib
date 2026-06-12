"""Ablation comparison figure: recall@10 vs epoch for all 5 conditions (gradual, ef=32)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = _SCRIPT_DIR if (_SCRIPT_DIR / "all.csv").exists() \
    else ROOT / "results" / "03_ablation"
FIGURES_DIR = _SCRIPT_DIR / "figures"
CONFIG_PATH = _SCRIPT_DIR / "config.yaml"

PRIMARY_EF = 32
DPI = 300

CONDITIONS = ["none", "coverage", "diversity", "two_hop", "all"]
LABELS = {
    "none":     "None (naive)",
    "coverage": "+Coverage",
    "diversity": "+Diversity",
    "two_hop":  "+Two-hop",
    "all":      "All improvements",
}


def load_data():
    frames = []
    for name in CONDITIONS:
        path = RESULTS_DIR / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing results file: {path}")
        df = pd.read_csv(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    schedule = config["drift"]["schedule_gradual"] if "drift" in config else None

    data = load_data()
    ef32 = data[data["ef_search"] == PRIMARY_EF]

    palette = sns.color_palette("tab10", len(CONDITIONS))
    colour_map = dict(zip(CONDITIONS, palette))

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(10, 5))

    for name in CONDITIONS:
        subset = ef32[ef32["condition"] == name].sort_values("epoch_idx")
        ax.plot(
            subset["epoch_idx"],
            subset["recall_at_k"],
            color=colour_map[name],
            label=LABELS[name],
            linewidth=2.0,
            marker="o",
            markersize=4,
            zorder=3,
        )

    if schedule is not None:
        # Shade drift phases (gradual)
        _shade_drift(ax, schedule)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Recall@10", fontsize=11)
    ax.set_title(
        f"Ablation: recall@10 vs epoch  (gradual drift, ef={PRIMARY_EF})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlim(-0.5, 24.5)
    all_recall = ef32["recall_at_k"].dropna()
    ymin = max(0.0, all_recall.min() - 0.03)
    ax.set_ylim(ymin, 1.01)
    ax.set_xticks(range(0, 25, 2))
    ax.legend(fontsize=9, loc="lower left", framealpha=0.85)

    fig.tight_layout()
    out = FIGURES_DIR / "ablation_comparison.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _shade_drift(ax, schedule):
    """Lightly shade non-zero drift epochs."""
    in_drift = False
    start = None
    for i, t in enumerate(schedule):
        if t > 0.0 and not in_drift:
            in_drift = True
            start = i
        elif t == 0.0 and in_drift:
            ax.axvspan(start - 0.5, i - 0.5, color="#fdae6b", alpha=0.18, zorder=0, linewidth=0)
            in_drift = False
    if in_drift:
        ax.axvspan(start - 0.5, len(schedule) - 0.5, color="#fdae6b", alpha=0.18, zorder=0, linewidth=0)


if __name__ == "__main__":
    main()
