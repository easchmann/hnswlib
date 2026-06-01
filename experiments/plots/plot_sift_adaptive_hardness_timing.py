# Generate figures from the CSVs produced by
# experiment_hardness_adaptive_sift.py  OR
# experiment_poolAndRewire_sift_hardness_continuous_small.py
#
# Includes all original plots PLUS a timing / overhead plot.
#
# Usage:
#   python plot_sift_adaptive_hardness_timing.py --results_dir results_hardness_adaptive_sift_with_time
#   python plot_sift_adaptive_hardness_timing.py --results_dir results_poolAndRewire_sift_hardness_continuous_small_with_time
#   python plot_sift_adaptive_hardness_timing.py --results_dir ... --plot_ef 50

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", required=True)
parser.add_argument("--plot_ef", type=int, default=None,
                    help="which ef to use for single-ef plots (default: median)")
parser.add_argument("--out_dir", default=None)
args = parser.parse_args()

out_dir = args.out_dir or args.results_dir
os.makedirs(out_dir, exist_ok=True)


def load(filename):
    path = os.path.join(args.results_dir, filename)
    if not os.path.exists(path):
        sys.exit(f"missing: {path}")
    return pd.read_csv(path)


def load_optional(filename):
    path = os.path.join(args.results_dir, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


summary  = load("summary.csv")
perquery = load("per_query.csv")

# Derive a unified base-knn column that works for both strategies:
# hardness_adaptive stores it as mean_t_global_knn_ms;
# poolAndRewire stores it as mean_t_orig_knn_ms (or not at all in older CSVs).
# Fall back to t_query_ms - t_pool_scan_ms - t_pool_knn_ms when both are absent/NaN.
if "mean_t_global_knn_ms" in summary.columns:
    summary["mean_t_base_knn_ms"] = summary["mean_t_global_knn_ms"]
elif "mean_t_orig_knn_ms" in summary.columns:
    summary["mean_t_base_knn_ms"] = summary["mean_t_orig_knn_ms"]
else:
    summary["mean_t_base_knn_ms"] = float("nan")

if "mean_t_orig_knn_ms" in summary.columns:
    mask = summary["mean_t_base_knn_ms"].isna()
    summary.loc[mask, "mean_t_base_knn_ms"] = summary.loc[mask, "mean_t_orig_knn_ms"]

pool_scan = summary.get("mean_t_pool_scan_ms", 0).fillna(0)
pool_knn  = summary.get("mean_t_pool_knn_ms",  0).fillna(0)
derived   = summary["mean_t_query_ms"] - pool_scan - pool_knn
summary["mean_t_base_knn_ms"] = summary["mean_t_base_knn_ms"].fillna(derived).clip(lower=0)

params = {}
params_path = os.path.join(args.results_dir, "params.csv")
if os.path.exists(params_path):
    for _, row in pd.read_csv(params_path).iterrows():
        params[row["param"]] = row["value"]

k          = int(params.get("k", 10))
strategies = sorted(summary["strategy"].unique())
efs        = sorted(summary["ef_search"].unique())
bins       = sorted(summary["hardness_bin"].unique())

plot_ef = args.plot_ef or efs[len(efs) // 2]
if plot_ef not in efs:
    plot_ef = min(efs, key=lambda e: abs(e - plot_ef))
    print(f"warning: requested ef not found, using {plot_ef}")

print(f"strategies: {strategies}")
print(f"efs:        {efs}")
print(f"bins:       {bins}")
print(f"plot_ef:    {plot_ef}\n")

# build bin labels: "bin N\n[lo, hi)"
def bin_label(b):
    row = summary[summary["hardness_bin"] == b].iloc[0]
    return f"bin {b}\n[{row['hardness_bin_lo']:.2f},{row['hardness_bin_hi']:.2f})"

bin_labels = [bin_label(b) for b in bins]

STRATEGY_COLORS = {
    "no_adaptation":    "#4C72B0",
    "poolAndRewire":    "#DD8452",
    "hardness_adaptive":"#55A868",
}
STRATEGY_LABELS = {
    "no_adaptation":    "no adaptation",
    "poolAndRewire":    "PoolAndRewire",
    "hardness_adaptive":"HardnessAdaptive",
}

def ef_cmap(n):
    return [matplotlib.colormaps["viridis"](i / max(n - 1, 1)) for i in range(n)]

def bin_cmap(n):
    return [matplotlib.colormaps["plasma"](i / max(n - 1, 1)) for i in range(n)]


def get(strategy, ef, b, col):
    row = summary[
        (summary["strategy"] == strategy) &
        (summary["ef_search"] == ef) &
        (summary["hardness_bin"] == b)
    ]
    if row.empty or col not in summary.columns:
        return float("nan")
    return float(row[col].iloc[0])


# --- plot 1: recall vs hardness bin, one line per strategy -------------------
# one panel per ef

def plot_recall_vs_bin():
    n_efs = len(efs)
    fig, axes = plt.subplots(1, n_efs, figsize=(5 * n_efs, 4.5), sharey=True)
    if n_efs == 1:
        axes = [axes]

    for ax, ef in zip(axes, efs):
        for strategy in strategies:
            recalls = [get(strategy, ef, b, "mean_recall") for b in bins]
            color = STRATEGY_COLORS.get(strategy, None)
            label = STRATEGY_LABELS.get(strategy, strategy)
            ax.plot(range(len(bins)), recalls, color=color, marker="o",
                    markersize=5, lw=1.8, label=label)

        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels(bin_labels, fontsize=7, rotation=30, ha="right")
        ax.set_xlabel("hardness bin")
        ax.set_title(f"ef={ef}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel(f"Recall@{k}")
    fig.suptitle(
        f"SIFT hardness drift — Recall@{k} vs hardness bin\n"
        f"easy -> hard query stream, continuous adaptation",
        y=1.02
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "recall_vs_bin.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# --- plot 2: recall delta (hardnessAdaptive - no_adaptation) vs bin -------------

def plot_recall_delta():
    if "no_adaptation" not in strategies or "hardness_adaptive" not in strategies:
        print("skipping recall delta: need both strategies")
        return

    fig, axes = plt.subplots(1, len(efs), figsize=(5 * len(efs), 4), sharey=True)
    if len(efs) == 1:
        axes = [axes]

    for ax, ef in zip(axes, efs):
        deltas = [
            get("hardness_adaptive", ef, b, "mean_recall") -
            get("no_adaptation", ef, b, "mean_recall")
            for b in bins
        ]
        colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]
        ax.bar(range(len(bins)), deltas, color=colors, alpha=0.8)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels(bin_labels, fontsize=7, rotation=30, ha="right")
        ax.set_xlabel("hardness bin")
        ax.set_title(f"ef={ef}")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel(f"ΔRecall@{k}  (hardnessAdaptive - baseline)")
    fig.suptitle(
        f"SIFT hardness drift — recall delta per bin\n"
        f"green = improvement, red = regression",
        y=1.02
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "recall_delta_vs_bin.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# --- plot 3: metrics vs hardness bin at one ef, one panel per metric ---------

def plot_metrics(ef):
    metrics = [
        ("mean_recall",              f"Recall@{k}",              "recall"),
        ("mean_ep_distance",         "entry-point distance",      "L2"),
        ("mean_bl_entry_distance",   "base-layer entry distance", "L2"),
        ("mean_layer1_visits",       "layer-1 visit count",       "# exams"),
        ("mean_base_visited",        "base-layer visited nodes",  "# nodes"),
        ("mean_base_dist_comps",     "base-layer dist comps",     "# comps"),
        ("mean_candidates_remaining","candidates at termination", "# candidates"),
    ]
    available = [m for m in metrics if m[0] in summary.columns]

    fig, axes = plt.subplots(len(available), 1,
                              figsize=(9, 3 * len(available)), sharex=True)
    if len(available) == 1:
        axes = [axes]

    for ax, (col, title, ylabel) in zip(axes, available):
        for strategy in strategies:
            vals = [get(strategy, ef, b, col) for b in bins]
            color = STRATEGY_COLORS.get(strategy, None)
            label = STRATEGY_LABELS.get(strategy, strategy)
            ax.plot(range(len(bins)), vals, color=color, marker="o",
                    markersize=4, lw=1.8, label=label)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    axes[-1].set_xlabel("hardness bin")
    axes[-1].set_xticks(range(len(bins)))
    axes[-1].set_xticklabels(bin_labels, fontsize=7, rotation=30, ha="right")
    fig.suptitle(f"metrics vs hardness bin  [ef={ef}]", y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, f"metrics_vs_bin_ef{ef}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# --- plot 4: per-query recall violin by bin and strategy ---------------------

def plot_violin(ef):
    df = perquery[perquery["ef_search"] == ef]
    if df.empty:
        print(f"no per-query data for ef={ef}, skipping violin")
        return

    bin_colors = bin_cmap(len(bins))
    fig, axes = plt.subplots(1, len(strategies),
                              figsize=(max(6, len(bins) * 0.8) * len(strategies), 5),
                              sharey=True)
    if len(strategies) == 1:
        axes = [axes]

    for ax, strategy in zip(axes, strategies):
        sub = df[df["strategy"] == strategy]
        data = [sub[sub["hardness_bin"] == b]["recall"].tolist() for b in bins]
        data = [d if d else [float("nan")] for d in data]

        parts = ax.violinplot(data, positions=range(len(bins)),
                               showmedians=True, showextrema=True)
        for pc, c in zip(parts["bodies"], bin_colors):
            pc.set_facecolor(c)
            pc.set_alpha(0.7)

        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels(bin_labels, fontsize=7, rotation=30, ha="right")
        ax.set_xlabel("hardness bin")
        ax.set_title(STRATEGY_LABELS.get(strategy, strategy))
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(f"Recall@{k}")
    axes[0].set_ylim(-0.05, 1.05)
    fig.suptitle(f"per-query recall distributions  [ef={ef}]")
    fig.tight_layout()
    path = os.path.join(out_dir, f"recall_violin_ef{ef}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# --- plot 5: timing / overhead -----------------------------------------------
# Three panels per ef value:
#   (a) mean query latency (ms) vs hardness bin, one line per strategy
#   (b) overhead ratio (strategy / no_adaptation) vs hardness bin
#   (c) stacked time breakdown for each adaptation strategy at each bin
#
# Column-name mapping: poolAndRewire scripts emit mean_t_orig_knn_ms;
# hardness_adaptive emits mean_t_global_knn_ms.  Both are the "base search"
# component — normalised to one label here.

def _timing_components(strategy):
    """Return list of (column, label, color) for the stacked breakdown."""
    components = [("mean_t_base_knn_ms", "base knn", "#4C72B0")]
    if "mean_t_pool_scan_ms" in summary.columns:
        components.append(("mean_t_pool_scan_ms", "pool scan", "#55A868"))
    if "mean_t_pool_knn_ms" in summary.columns:
        components.append(("mean_t_pool_knn_ms",  "pool knn",  "#DD8452"))
    if "mean_t_adapt_ms" in summary.columns:
        components.append(("mean_t_adapt_ms", "adapt overhead", "#C44E52"))
    return components


def plot_timing(ef):
    timing_col = "mean_t_query_ms"
    if timing_col not in summary.columns:
        print("skipping timing plot: mean_t_query_ms not in summary")
        return

    adapt_strategies = [s for s in strategies if s != "no_adaptation"]
    n_adapt = len(adapt_strategies)

    # rows: (a) latency, (b) overhead, (c) one stacked bar per adaptation strategy
    n_rows = 2 + n_adapt
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3.5 * n_rows), sharex=True)

    x = np.arange(len(bins))

    # (a) mean query latency per strategy
    ax_lat = axes[0]
    for strategy in strategies:
        vals = [get(strategy, ef, b, timing_col) for b in bins]
        color = STRATEGY_COLORS.get(strategy, None)
        label = STRATEGY_LABELS.get(strategy, strategy)
        ax_lat.plot(x, vals, color=color, marker="o", markersize=5, lw=1.8, label=label)
    ax_lat.set_ylabel("mean query time (ms)")
    ax_lat.set_title(f"query latency vs hardness bin  [ef={ef}]")
    ax_lat.legend(fontsize=8)
    ax_lat.grid(alpha=0.3)

    # (b) overhead ratio vs no_adaptation
    ax_oh = axes[1]
    base_vals = np.array([get("no_adaptation", ef, b, timing_col) for b in bins])
    for strategy in adapt_strategies:
        adapt_vals = np.array([get(strategy, ef, b, timing_col) for b in bins])
        ratio = np.where(base_vals > 0, adapt_vals / base_vals, np.nan)
        color = STRATEGY_COLORS.get(strategy, None)
        label = STRATEGY_LABELS.get(strategy, strategy)
        ax_oh.plot(x, ratio, color=color, marker="s", markersize=5, lw=1.8, label=label)
    ax_oh.axhline(1.0, color="black", lw=0.8, ls="--", label="1× (no overhead)")
    ax_oh.set_ylabel("query time overhead  (×)")
    ax_oh.set_title("overhead ratio vs no_adaptation")
    ax_oh.legend(fontsize=8)
    ax_oh.grid(alpha=0.3)

    # (c) stacked time breakdown per adaptation strategy
    bar_width = 0.6
    for ax_stack, strategy in zip(axes[2:], adapt_strategies):
        components = _timing_components(strategy)
        bottoms = np.zeros(len(bins))
        for col, label, color in components:
            vals = np.array([get(strategy, ef, b, col) for b in bins])
            vals = np.nan_to_num(vals)
            ax_stack.bar(x, vals, bar_width, bottom=bottoms, label=label, color=color, alpha=0.85)
            bottoms += vals
        ax_stack.set_ylabel("time (ms)")
        ax_stack.set_title(f"time breakdown — {STRATEGY_LABELS.get(strategy, strategy)}")
        ax_stack.legend(fontsize=8, loc="upper left")
        ax_stack.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel("hardness bin")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(bin_labels, fontsize=7, rotation=30, ha="right")
    fig.suptitle(f"query timing overhead  [ef={ef}]", y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, f"timing_ef{ef}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def plot_timing_all_efs():
    """Mean query latency across all efs: grouped bars, one group per strategy."""
    timing_col = "mean_t_query_ms"
    if timing_col not in summary.columns:
        return

    fig, axes = plt.subplots(1, len(efs), figsize=(5 * len(efs), 4.5), sharey=False)
    if len(efs) == 1:
        axes = [axes]

    for ax, ef in zip(axes, efs):
        for strategy in strategies:
            vals = [get(strategy, ef, b, timing_col) for b in bins]
            color = STRATEGY_COLORS.get(strategy, None)
            label = STRATEGY_LABELS.get(strategy, strategy)
            ax.plot(range(len(bins)), vals, color=color, marker="o",
                    markersize=4, lw=1.8, label=label)
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels(bin_labels, fontsize=7, rotation=30, ha="right")
        ax.set_title(f"ef={ef}")
        ax.set_xlabel("hardness bin")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("mean query time (ms)")
    fig.suptitle("query latency vs hardness bin — all ef values", y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "timing_all_efs.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# --- run everything ----------------------------------------------------------

plot_recall_vs_bin()
plot_recall_delta()
plot_metrics(plot_ef)
plot_violin(plot_ef)
plot_timing(plot_ef)
plot_timing_all_efs()

print(f"\nall plots written to {os.path.abspath(out_dir)}")
