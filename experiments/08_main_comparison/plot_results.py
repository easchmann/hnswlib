"""Experiment 08 figures: main thesis comparison."""

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
    "static": "grey",
    "periodic_rebuild": "darkorange",
    "adaptive_mconj48": "steelblue",
}
CONDITION_LABELS = {
    "static": "Static HNSW",
    "periodic_rebuild": "Periodic rebuild (every 5 ep.)",
    "adaptive_mconj48": "Adaptive ($M_{\\mathrm{conj}}=48$)",
}


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _results_dir(cfg):
    path = ROOT / cfg["output"]["results_dir"]
    if path.exists():
        return path
    return _SCRIPT_DIR


def _load_schedule_condition(results_dir, schedule, condition_name):
    p = results_dir / f"{schedule}_{condition_name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    return pd.read_csv(p)


def _load_all(cfg, results_dir):
    """Returns dict: {(schedule, condition_name): df}"""
    data = {}
    for schedule in ("gradual", "sudden"):
        for cond in cfg["conditions"]:
            name = cond["name"]
            try:
                df = _load_schedule_condition(results_dir, schedule, name)
                data[(schedule, name)] = df
            except FileNotFoundError as e:
                print(f"  WARNING: {e}")
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
    """1x2: recall@10 vs epoch for all conditions at ef=32."""
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

        # Pre-drift recall annotation on gradual left panel
        if schedule == "gradual":
            key_static = ("gradual", "static")
            if key_static in data:
                df_s = data[key_static]
                pre_r = df_s[(df_s["ef_search"] == PRIMARY_EF) & (df_s["epoch_idx"] <= 4)]["recall_at_k"].mean()
                ax.annotate(
                    f"pre-drift: {pre_r:.3f}",
                    xy=(2, pre_r),
                    xytext=(5, pre_r - 0.07),
                    fontsize=9,
                    color="grey",
                    arrowprops=dict(arrowstyle="->", color="grey", lw=1.0),
                )

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Recall@10", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(-0.5, 24.5)
        ax.set_xticks(range(0, 25, 2))
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, loc="lower left", framealpha=0.85)

    fig.suptitle(
        f"Recall@10 vs epoch  (ef={PRIMARY_EF})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    out = figures_dir / "recall_vs_epoch.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_figure2(data, cfg, figures_dir):
    """1x2: recall@10 vs ef_search grouped bar chart, post-drift mean."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    ef_values = cfg["eval"]["ef_search_values"]
    conditions = cfg["conditions"]
    schedules = ("gradual", "sudden")
    schedule_titles = ("Gradual drift", "Sudden drift")

    n_cond = len(conditions)
    n_ef = len(ef_values)
    bar_width = 0.22
    x = np.arange(n_ef)

    for ax, schedule, title in zip(axes, schedules, schedule_titles):
        for ci, cond in enumerate(conditions):
            name = cond["name"]
            key = (schedule, name)
            recalls = []
            for ef in ef_values:
                if key not in data:
                    recalls.append(0.0)
                    continue
                df = data[key]
                val = df[(df["ef_search"] == ef) & (df["epoch_idx"] >= 20)]["recall_at_k"].mean()
                recalls.append(val)

            offset = (ci - n_cond / 2 + 0.5) * bar_width
            bars = ax.bar(
                x + offset, recalls,
                width=bar_width,
                color=CONDITION_COLORS.get(name, "black"),
                label=CONDITION_LABELS.get(name, name),
                zorder=3,
                alpha=0.85,
            )
            for bar, val in zip(bars, recalls):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7.5,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([f"ef={e}" for e in ef_values], fontsize=10)
        ax.set_xlabel("ef_search", fontsize=11)
        ax.set_ylabel("Post-drift mean Recall@10  (epochs 20–24)", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, loc="lower right", framealpha=0.85)

    fig.suptitle(
        "Post-drift Recall@10 vs ef_search",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    out = figures_dir / "recall_vs_ef.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_figure3(data, cfg, figures_dir):
    """Hero figure: adaptive (gradual) with ±1 std band, overlay static dashed."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(11, 5))

    schedule = "gradual"
    drift_schedule = cfg["drift"]["schedule_gradual"]
    _shade_drift(ax, drift_schedule)

    # Static overlay
    key_static = (schedule, "static")
    if key_static in data:
        df_s = data[key_static]
        sub_s = df_s[df_s["ef_search"] == PRIMARY_EF].sort_values("epoch_idx")
        ax.plot(
            sub_s["epoch_idx"],
            sub_s["recall_at_k"],
            color="grey",
            linewidth=1.8,
            linestyle="--",
            label=CONDITION_LABELS["static"],
            zorder=3,
            alpha=0.8,
        )

    # Adaptive with std band
    key_adapt = (schedule, "adaptive_mconj48")
    if key_adapt in data:
        df_a = data[key_adapt]
        sub_a = df_a[df_a["ef_search"] == PRIMARY_EF].sort_values("epoch_idx")
        epochs = sub_a["epoch_idx"].values
        recalls = sub_a["recall_at_k"].values
        ax.plot(
            epochs, recalls,
            color=CONDITION_COLORS["adaptive_mconj48"],
            linewidth=2.2,
            marker="o",
            markersize=4,
            label=CONDITION_LABELS["adaptive_mconj48"],
            zorder=4,
        )
        if "per_query_recall_std" in sub_a.columns:
            stds = sub_a["per_query_recall_std"].values
            ax.fill_between(
                epochs,
                np.clip(recalls - stds, 0, 1),
                np.clip(recalls + stds, 0, 1),
                color=CONDITION_COLORS["adaptive_mconj48"],
                alpha=0.18,
                zorder=2,
                label="±1 std (per-query)",
            )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Recall@10", fontsize=12)
    ax.set_title(
        f"Adaptive vs Static — gradual drift  (ef={PRIMARY_EF})",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(-0.5, 24.5)
    ax.set_xticks(range(0, 25, 2))
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="lower left", framealpha=0.9)

    fig.tight_layout()
    out = figures_dir / "hero_adaptive_vs_static.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def print_summary_table(data, cfg):
    conditions = cfg["conditions"]
    schedules = ("gradual", "sudden")

    header = (
        f"{'Schedule':<10}  {'Condition':<20}  {'Post-drift R@10':>15}  "
        f"{'Recall drop':>11}  {'vs static (pp)':>14}  {'Edges/Rebuilds ep24':>19}"
    )
    print("\n" + header)
    print("-" * len(header))

    for schedule in schedules:
        static_post = None
        for cond in conditions:
            name = cond["name"]
            key = (schedule, name)
            if key not in data:
                print(f"  {schedule:<10}  {name:<20}  (no data)")
                continue
            df = data[key]
            ef_df = df[df["ef_search"] == PRIMARY_EF]
            pre = ef_df[ef_df["epoch_idx"] <= 4]["recall_at_k"].mean()
            post = ef_df[ef_df["epoch_idx"] >= 20]["recall_at_k"].mean()
            drop = pre - post

            if name == "static":
                static_post = post

            vs_static = (post - static_post) * 100 if static_post is not None and name != "static" else 0.0

            ep24_edges = ef_df[ef_df["epoch_idx"] == 24]["n_conjugate_edges"].values
            ep24_rebuilds = ef_df[ef_df["epoch_idx"] == 24]["n_index_rebuilds"].values
            edges_val = int(ep24_edges[0]) if len(ep24_edges) > 0 else -1
            rebuilds_val = int(ep24_rebuilds[0]) if len(ep24_rebuilds) > 0 else -1

            if name == "static":
                extra = f"{edges_val:>19,}"
            elif name == "periodic_rebuild":
                extra = f"{rebuilds_val:>19d} rebuilds"
            else:
                extra = f"{edges_val:>19,}"

            print(
                f"  {schedule:<10}  {name:<20}  {post:>15.4f}  "
                f"{drop:>11.4f}  {vs_static:>+14.2f}  {extra}"
            )
    print()


def main():
    cfg = _load_config()
    results_dir = _results_dir(cfg)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = _load_all(cfg, results_dir)
    if not data:
        print("No results found. Run run.py first.")
        return

    plot_figure1(data, cfg, figures_dir)
    plot_figure2(data, cfg, figures_dir)
    plot_figure3(data, cfg, figures_dir)
    print_summary_table(data, cfg)


if __name__ == "__main__":
    main()
