"""Experiment 04 tuning figures: sweep comparisons and best-guess summary."""

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

SWEEP_GROUPS = {
    "nodes": {
        "conditions": ["nodes_100", "nodes_250", "nodes_500", "nodes_1000", "nodes_2000", "nodes_5000_twohop"],
        "palette": "Blues",
        "title": "max_repair_nodes sweep",
        "labels": {
            "nodes_100": "100 nodes",
            "nodes_250": "250 nodes",
            "nodes_500": "500 nodes",
            "nodes_1000": "1 000 nodes",
            "nodes_2000": "2 000 nodes",
            "nodes_5000_twohop": "5 000 nodes",
        },
    },
    "thresh": {
        "conditions": ["thresh_85", "thresh_90"],
        "palette": "Oranges",
        "title": "EH threshold percentile sweep",
        "labels": {
            "thresh_85": "85th percentile",
            "thresh_90": "90th percentile",
        },
    },
    "cooldown": {
        "conditions": ["cooldown_2", "cooldown_3"],
        "palette": "Greens",
        "title": "repair_cooldown sweep",
        "labels": {
            "cooldown_2": "cooldown=2",
            "cooldown_3": "cooldown=3",
        },
    },
    "mconj": {
        "conditions": ["mconj_12", "mconj_16"],
        "palette": "Purples",
        "title": "M_conj sweep",
        "labels": {
            "mconj_12": "M_conj=12",
            "mconj_16": "M_conj=16",
        },
    },
}


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _results_dir(cfg):
    path = ROOT / cfg["output"]["results_dir"]
    if path.exists():
        return path
    return _SCRIPT_DIR


def _load_condition(results_dir, name):
    path = results_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    return pd.read_csv(path)


def _load_all(cfg):
    results_dir = _results_dir(cfg)
    frames = []
    for cond in cfg["tuning_conditions"]:
        df = _load_condition(results_dir, cond["name"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _load_ablation_baseline(cfg):
    path = ROOT / cfg["output"]["ablation_two_hop_path"]
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df[df["ef_search"] == PRIMARY_EF].sort_values("epoch_idx")


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


def _mean_post_drift(df_ef32, condition, lo=20, hi=24):
    sub = df_ef32[(df_ef32["condition"] == condition) & (df_ef32["epoch_idx"] >= lo)]
    return sub["recall_at_k"].mean()


def _best_in_group(group_conditions, ef32):
    best, best_recall = None, -1.0
    for c in group_conditions:
        if c not in ef32["condition"].values:
            continue
        r = _mean_post_drift(ef32, c)
        if r > best_recall:
            best_recall = r
            best = c
    return best


def plot_figure1(ef32, ablation_baseline, schedule, figures_dir):
    sns.set_theme(style="whitegrid", font_scale=0.95)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    axes_flat = axes.flatten()
    group_keys = list(SWEEP_GROUPS.keys())

    for ax_idx, group_key in enumerate(group_keys):
        ax = axes_flat[ax_idx]
        group = SWEEP_GROUPS[group_key]
        conditions = group["conditions"]
        labels = group["labels"]
        palette_name = group["palette"]

        n = len(conditions)
        colors = sns.color_palette(palette_name, n + 2)[2:]

        if schedule:
            _shade_drift(ax, schedule)

        if ablation_baseline is not None:
            ax.plot(
                ablation_baseline["epoch_idx"],
                ablation_baseline["recall_at_k"],
                color="grey",
                linestyle="--",
                linewidth=1.5,
                label="two_hop baseline (ablation)",
                zorder=2,
                alpha=0.8,
            )

        for color, cname in zip(colors, conditions):
            sub = ef32[ef32["condition"] == cname].sort_values("epoch_idx")
            if sub.empty:
                continue
            ax.plot(
                sub["epoch_idx"],
                sub["recall_at_k"],
                color=color,
                linewidth=1.8,
                marker="o",
                markersize=3.5,
                label=labels.get(cname, cname),
                zorder=3,
            )

        ax.set_title(group["title"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Recall@10", fontsize=10)
        ax.set_xlim(-0.5, 24.5)
        ax.set_xticks(range(0, 25, 4))
        ax.legend(fontsize=8, loc="lower left", framealpha=0.85)

    fig.suptitle(
        f"Experiment 04 tuning sweeps  (gradual drift, ef={PRIMARY_EF})",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = figures_dir / "tuning_sweeps.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_figure2(ef32, schedule, figures_dir):
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(11, 5))

    best_per_group = {}
    for group_key, group in SWEEP_GROUPS.items():
        best = _best_in_group(group["conditions"], ef32)
        best_per_group[group_key] = best

    featured = ["nodes_500", "best_guess"]
    for group_key, best in best_per_group.items():
        if best and best not in featured:
            featured.append(best)

    # Deduplicate while preserving order
    seen = set()
    featured_deduped = []
    for c in featured:
        if c not in seen:
            seen.add(c)
            featured_deduped.append(c)

    palette = sns.color_palette("tab10", len(featured_deduped))

    if schedule:
        _shade_drift(ax, schedule)

    group_best_label = {v: k for k, v in best_per_group.items() if v}

    for color, cname in zip(palette, featured_deduped):
        sub = ef32[ef32["condition"] == cname].sort_values("epoch_idx")
        if sub.empty:
            continue
        if cname == "nodes_500":
            label = "nodes_500 (two_hop baseline)"
        elif cname == "best_guess":
            label = "best_guess (combined)"
        else:
            group_tag = group_best_label.get(cname, "")
            label = f"{cname}  [best {group_tag}]" if group_tag else cname
        ax.plot(
            sub["epoch_idx"],
            sub["recall_at_k"],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=4,
            label=label,
            zorder=3,
        )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Recall@10", fontsize=11)
    ax.set_title(
        f"Best-of-sweep comparison  (gradual drift, ef={PRIMARY_EF})",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlim(-0.5, 24.5)
    ax.set_xticks(range(0, 25, 2))
    ax.legend(fontsize=9, loc="lower left", framealpha=0.85)

    fig.tight_layout()
    out = figures_dir / "best_of_sweep.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def print_summary_table(ef32, cfg):
    conditions = [c["name"] for c in cfg["tuning_conditions"]]
    header = f"{'Condition':<22}  {'Post-drift R@10':>15}  {'Recall drop':>11}  {'Conj edges ep24':>15}"
    print("\n" + header)
    print("-" * len(header))
    for cname in conditions:
        sub = ef32[ef32["condition"] == cname]
        if sub.empty:
            continue
        pre = sub[sub["epoch_idx"] <= 4]["recall_at_k"].mean()
        post = sub[sub["epoch_idx"] >= 20]["recall_at_k"].mean()
        ep24 = sub[sub["epoch_idx"] == 24]["n_conjugate_edges"].values
        edges_val = int(ep24[0]) if len(ep24) > 0 else -1
        drop = pre - post
        print(f"{cname:<22}  {post:>15.4f}  {drop:>11.4f}  {edges_val:>15d}")
    print()


def main():
    cfg = _load_config()
    results_dir = _results_dir(cfg)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    schedule = cfg.get("drift", {}).get("schedule_gradual")
    all_data = _load_all(cfg)
    ef32 = all_data[all_data["ef_search"] == PRIMARY_EF].copy()
    ablation_baseline = _load_ablation_baseline(cfg)

    plot_figure1(ef32, ablation_baseline, schedule, figures_dir)
    plot_figure2(ef32, schedule, figures_dir)
    print_summary_table(ef32, cfg)


if __name__ == "__main__":
    main()
