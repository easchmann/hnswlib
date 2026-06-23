"""Experiment 14 figures: entry point adaptation vs conjugate-only baseline."""

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
CONFIG_PATH = _SCRIPT_DIR / "config.yaml"
DPI = 300
PRIMARY_EF = 32

CONDITION_COLORS = {
    "adaptive_mconj48": "steelblue",
    "adaptive_mconj48_ep": "crimson",
}
CONDITION_LABELS = {
    "adaptive_mconj48": "Adaptive ($M_{\\mathrm{conj}}=48$)",
    "adaptive_mconj48_ep": "Adaptive + Entry Point",
}


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _results_dir(cfg):
    path = ROOT / cfg["output"]["results_dir"]
    if path.exists():
        return path
    return _SCRIPT_DIR


def _load_all(cfg, results_dir):
    data = {}
    for schedule in ("gradual", "sudden"):
        for cond in cfg["conditions"]:
            name = cond["name"]
            p = results_dir / f"{schedule}_{name}.csv"
            if not p.exists():
                print(f"  WARNING: Missing {p}")
                continue
            data[(schedule, name)] = pd.read_csv(p)
    return data


def _shade_drift(ax, schedule):
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


def plot_figure1(data, cfg, figures_dir):
    """1x2: recall@10 vs epoch for both conditions at ef=32."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    schedules = ("gradual", "sudden")
    schedule_titles = ("Gradual drift", "Sudden drift")
    drift_schedules = {
        "gradual": cfg["drift"]["schedule_gradual"],
        "sudden": cfg["drift"]["schedule_sudden"],
    }

    for ax, schedule, title in zip(axes, schedules, schedule_titles):
        _shade_drift(ax, drift_schedules[schedule])

        for cond in cfg["conditions"]:
            name = cond["name"]
            key = (schedule, name)
            if key not in data:
                continue
            df = data[key]
            sub = df[df["ef_search"] == PRIMARY_EF].sort_values("epoch_idx")
            ax.plot(
                sub["epoch_idx"],
                sub["recall_at_k"],
                color=CONDITION_COLORS.get(name, "black"),
                linewidth=2.0,
                marker="o",
                markersize=3.5,
                label=CONDITION_LABELS.get(name, name),
                zorder=3,
            )

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Recall@10", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(-0.5, 24.5)
        ax.set_xticks(range(0, 25, 2))
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, loc="lower left", framealpha=0.85)

    fig.suptitle(
        f"Recall@10 vs epoch — entry point adaptation  (ef={PRIMARY_EF})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    out = figures_dir / "recall_vs_epoch.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_figure2(data, cfg, figures_dir):
    """1x2: n_conjugate_edges vs epoch — should be identical across conditions."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

    schedules = ("gradual", "sudden")
    schedule_titles = ("Gradual drift", "Sudden drift")
    drift_schedules = {
        "gradual": cfg["drift"]["schedule_gradual"],
        "sudden": cfg["drift"]["schedule_sudden"],
    }

    for ax, schedule, title in zip(axes, schedules, schedule_titles):
        _shade_drift(ax, drift_schedules[schedule])

        for cond in cfg["conditions"]:
            name = cond["name"]
            key = (schedule, name)
            if key not in data:
                continue
            df = data[key]
            sub = df[df["ef_search"] == PRIMARY_EF].sort_values("epoch_idx")
            ax.plot(
                sub["epoch_idx"],
                sub["n_conjugate_edges"],
                color=CONDITION_COLORS.get(name, "black"),
                linewidth=2.0,
                marker="o",
                markersize=3.5,
                label=CONDITION_LABELS.get(name, name),
                zorder=3,
            )

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Conjugate edges added", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(-0.5, 24.5)
        ax.set_xticks(range(0, 25, 2))
        ax.legend(fontsize=9, loc="upper left", framealpha=0.85)

    fig.suptitle(
        "Conjugate edge count vs epoch (should be identical across conditions)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    out = figures_dir / "edge_count_vs_epoch.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def print_summary_table(data, cfg):
    schedules = ("gradual", "sudden")
    header = (
        f"{'Schedule':<10}  {'Condition':<25}  {'Post-drift R@10':>15}  "
        f"{'vs baseline (pp)':>16}  {'Edges ep24':>10}"
    )
    print("\n" + header)
    print("-" * len(header))

    for schedule in schedules:
        baseline_post = None
        for cond in cfg["conditions"]:
            name = cond["name"]
            key = (schedule, name)
            if key not in data:
                print(f"  {schedule:<10}  {name:<25}  (no data)")
                continue
            df = data[key]
            ef_df = df[df["ef_search"] == PRIMARY_EF]
            post = ef_df[ef_df["epoch_idx"] >= 20]["recall_at_k"].mean()

            if baseline_post is None:
                baseline_post = post
                vs = 0.0
            else:
                vs = (post - baseline_post) * 100

            ep24_edges = ef_df[ef_df["epoch_idx"] == 24]["n_conjugate_edges"].values
            edges_val = int(ep24_edges[0]) if len(ep24_edges) > 0 else -1

            print(
                f"  {schedule:<10}  {name:<25}  {post:>15.4f}  "
                f"{vs:>+16.2f}  {edges_val:>10,}"
            )
    print()


def main():
    cfg = _load_config()
    results_dir = _results_dir(cfg)
    figures_dir = _SCRIPT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = _load_all(cfg, results_dir)
    if not data:
        print("No results found. Run run.py first.")
        return

    plot_figure1(data, cfg, figures_dir)
    plot_figure2(data, cfg, figures_dir)
    print_summary_table(data, cfg)


if __name__ == "__main__":
    main()
