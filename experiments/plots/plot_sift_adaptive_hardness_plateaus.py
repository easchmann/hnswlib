# Generate figures from the CSVs produced by
# experiment_hardness_adaptive_sift_with_plateaus.py
#
# X-axis is phase_idx (sequential position in the stream), not hardness_bin,
# since the same hardness level appears multiple times (drift + plateau).
# Plateau phases are shaded grey in every plot.
#
# Usage:
#   python plot_sift_adaptive_hardness_plateaus.py --results_dir results_hardness_adaptive_sift_with_plateaus
#   python plot_sift_adaptive_hardness_plateaus.py --results_dir ... --plot_ef 50

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

# unified base-knn column (same logic as timing template)
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

# build phase list from summary (phase_idx, hardness_bin, is_plateau, edges)
_ref = summary[(summary["strategy"] == strategies[0]) &
               (summary["ef_search"] == efs[0])].sort_values("phase_idx")
phase_list = _ref[["phase_idx", "hardness_bin", "is_plateau",
                    "hardness_bin_lo", "hardness_bin_hi"]].drop_duplicates().to_dict("records")
phases = [p["phase_idx"] for p in phase_list]

plot_ef = args.plot_ef or efs[len(efs) // 2]
if plot_ef not in efs:
    plot_ef = min(efs, key=lambda e: abs(e - plot_ef))
    print(f"warning: requested ef not found, using {plot_ef}")

print(f"strategies: {strategies}")
print(f"efs:        {efs}")
print(f"phases:     {len(phases)} ({sum(p['is_plateau'] for p in phase_list)} plateau)")
print(f"plot_ef:    {plot_ef}\n")

STRATEGY_COLORS = {
    "no_adaptation":     "#4C72B0",
    "poolAndRewire":     "#DD8452",
    "hardness_adaptive": "#55A868",
}
STRATEGY_LABELS = {
    "no_adaptation":     "no adaptation",
    "poolAndRewire":     "PoolAndRewire",
    "hardness_adaptive": "HardnessAdaptive",
}

def ef_cmap(n):
    return [matplotlib.colormaps["viridis"](i / max(n - 1, 1)) for i in range(n)]


def phase_label(p):
    marker = "▲" if p["is_plateau"] else ""
    return f"ph{p['phase_idx']}{marker}\nbin{p['hardness_bin']}\n[{p['hardness_bin_lo']:.1f},{p['hardness_bin_hi']:.1f})"

phase_labels = [phase_label(p) for p in phase_list]


def get(strategy, ef, phase_idx, col):
    row = summary[
        (summary["strategy"] == strategy) &
        (summary["ef_search"] == ef) &
        (summary["phase_idx"] == phase_idx)
    ]
    if row.empty or col not in summary.columns:
        return float("nan")
    return float(row[col].iloc[0])


def shade_plateaus(ax):
    """Shade plateau phase positions with a light grey band."""
    for i, p in enumerate(phase_list):
        if p["is_plateau"]:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.12, color="grey", zorder=0)


# --- plot 1: recall vs phase, plateau shaded ---------------------------------

def plot_recall_vs_phase():
    n_efs = len(efs)
    fig, axes = plt.subplots(1, n_efs, figsize=(5 * n_efs, 4.5), sharey=True)
    if n_efs == 1:
        axes = [axes]

    for ax, ef in zip(axes, efs):
        shade_plateaus(ax)
        for strategy in strategies:
            recalls = [get(strategy, ef, ph, "mean_recall") for ph in phases]
            ax.plot(range(len(phases)), recalls,
                    color=STRATEGY_COLORS.get(strategy),
                    marker="o", markersize=5, lw=1.8,
                    label=STRATEGY_LABELS.get(strategy, strategy))
        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels(phase_labels, fontsize=6.5, rotation=35, ha="right")
        ax.set_xlabel("phase (▲ = plateau)")
        ax.set_title(f"ef={ef}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel(f"Recall@{k}")
    fig.suptitle(
        f"SIFT hardness drift with plateaus — Recall@{k} vs phase\n"
        f"grey bands = plateau (stable hardness level)",
        y=1.02
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "recall_vs_phase.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# --- plot 2: recall delta vs phase -------------------------------------------

def plot_recall_delta():
    if "no_adaptation" not in strategies:
        print("skipping recall delta: no_adaptation not present")
        return

    adapt_strategies = [s for s in strategies if s != "no_adaptation"]
    if not adapt_strategies:
        return

    fig, axes = plt.subplots(len(efs), len(adapt_strategies),
                              figsize=(5 * len(adapt_strategies), 3.5 * len(efs)),
                              sharey="row", sharex=True, squeeze=False)

    for row_i, ef in enumerate(efs):
        for col_i, strategy in enumerate(adapt_strategies):
            ax = axes[row_i][col_i]
            shade_plateaus(ax)
            deltas = [
                get(strategy, ef, ph, "mean_recall") -
                get("no_adaptation", ef, ph, "mean_recall")
                for ph in phases
            ]
            colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]
            ax.bar(range(len(phases)), deltas, color=colors, alpha=0.8)
            ax.axhline(0, color="black", lw=0.8, ls="--")
            ax.set_title(f"{STRATEGY_LABELS.get(strategy, strategy)}  ef={ef}", fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            if row_i == len(efs) - 1:
                ax.set_xticks(range(len(phases)))
                ax.set_xticklabels(phase_labels, fontsize=6.5, rotation=35, ha="right")
                ax.set_xlabel("phase (▲ = plateau)")
        axes[row_i][0].set_ylabel(f"ΔRecall@{k}")

    fig.suptitle("recall delta vs baseline — all ef values\ngreen = improvement, red = regression", y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "recall_delta_vs_phase.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# --- plot 3: metrics vs phase at one ef --------------------------------------

def plot_metrics(ef):
    metrics = [
        ("mean_recall",               f"Recall@{k}",              "recall"),
        ("mean_ep_distance",          "entry-point distance",      "L2"),
        ("mean_bl_entry_distance",    "base-layer entry distance", "L2"),
        ("mean_layer1_visits",        "layer-1 visit count",       "# exams"),
        ("mean_base_visited",         "base-layer visited nodes",  "# nodes"),
        ("mean_base_dist_comps",      "base-layer dist comps",     "# comps"),
        ("mean_candidates_remaining", "candidates at termination", "# candidates"),
    ]
    available = [m for m in metrics if m[0] in summary.columns]

    fig, axes = plt.subplots(len(available), 1,
                              figsize=(10, 3 * len(available)), sharex=True)
    if len(available) == 1:
        axes = [axes]

    for ax, (col, title, ylabel) in zip(axes, available):
        shade_plateaus(ax)
        for strategy in strategies:
            vals = [get(strategy, ef, ph, col) for ph in phases]
            ax.plot(range(len(phases)), vals,
                    color=STRATEGY_COLORS.get(strategy),
                    marker="o", markersize=4, lw=1.8,
                    label=STRATEGY_LABELS.get(strategy, strategy))
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    axes[-1].set_xlabel("phase (▲ = plateau)")
    axes[-1].set_xticks(range(len(phases)))
    axes[-1].set_xticklabels(phase_labels, fontsize=6.5, rotation=35, ha="right")
    fig.suptitle(f"metrics vs phase  [ef={ef}]", y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, f"metrics_vs_phase_ef{ef}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# --- plot 4: drift vs plateau recall comparison ------------------------------
# Bar chart: mean recall during drift phases vs plateau phases, per strategy/ef.
# This is the key plot specific to the plateau experiment.

def plot_drift_vs_plateau():
    drift_phases   = [p["phase_idx"] for p in phase_list if not p["is_plateau"]]
    plateau_phases = [p["phase_idx"] for p in phase_list if p["is_plateau"]]
    if not plateau_phases:
        print("skipping drift_vs_plateau: no plateau phases found")
        return

    fig, axes = plt.subplots(1, len(efs), figsize=(5 * len(efs), 4.5), sharey=True)
    if len(efs) == 1:
        axes = [axes]

    x = np.arange(len(strategies))
    width = 0.35

    for ax, ef in zip(axes, efs):
        drift_means   = [np.nanmean([get(s, ef, ph, "mean_recall") for ph in drift_phases])   for s in strategies]
        plateau_means = [np.nanmean([get(s, ef, ph, "mean_recall") for ph in plateau_phases]) for s in strategies]

        bars_d = ax.bar(x - width / 2, drift_means,   width, label="drift phases",   alpha=0.85, color="#4C72B0")
        bars_p = ax.bar(x + width / 2, plateau_means, width, label="plateau phases", alpha=0.85, color="#C44E52")

        ax.bar_label(bars_d, fmt="%.3f", fontsize=7, padding=2)
        ax.bar_label(bars_p, fmt="%.3f", fontsize=7, padding=2)

        ax.set_xticks(x)
        ax.set_xticklabels([STRATEGY_LABELS.get(s, s) for s in strategies], rotation=15, ha="right", fontsize=8)
        ax.set_title(f"ef={ef}")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel(f"mean Recall@{k}")
    fig.suptitle(
        f"Mean recall: drift phases vs plateau phases\n"
        f"plateau = stable hardness region where adaptation should settle",
        y=1.02
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "drift_vs_plateau_recall.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# --- plot 5: timing vs phase -------------------------------------------------

def _timing_components(strategy):
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
    n_rows = 2 + len(adapt_strategies)
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, 3.5 * n_rows), sharex=True)

    x = np.arange(len(phases))

    # (a) mean query latency
    ax_lat = axes[0]
    shade_plateaus(ax_lat)
    for strategy in strategies:
        vals = [get(strategy, ef, ph, timing_col) for ph in phases]
        ax_lat.plot(x, vals, color=STRATEGY_COLORS.get(strategy),
                    marker="o", markersize=5, lw=1.8,
                    label=STRATEGY_LABELS.get(strategy, strategy))
    ax_lat.set_ylabel("mean query time (ms)")
    ax_lat.set_title(f"query latency vs phase  [ef={ef}]")
    ax_lat.legend(fontsize=8)
    ax_lat.grid(alpha=0.3)

    # (b) overhead ratio
    ax_oh = axes[1]
    shade_plateaus(ax_oh)
    base_vals = np.array([get("no_adaptation", ef, ph, timing_col) for ph in phases])
    for strategy in adapt_strategies:
        adapt_vals = np.array([get(strategy, ef, ph, timing_col) for ph in phases])
        ratio = np.where(base_vals > 0, adapt_vals / base_vals, np.nan)
        ax_oh.plot(x, ratio, color=STRATEGY_COLORS.get(strategy),
                   marker="s", markersize=5, lw=1.8,
                   label=STRATEGY_LABELS.get(strategy, strategy))
    ax_oh.axhline(1.0, color="black", lw=0.8, ls="--", label="1× (no overhead)")
    ax_oh.set_ylabel("overhead (×)")
    ax_oh.set_title("overhead ratio vs no_adaptation")
    ax_oh.legend(fontsize=8)
    ax_oh.grid(alpha=0.3)

    # (c) stacked time breakdown per adaptation strategy
    bar_width = 0.6
    for ax_stack, strategy in zip(axes[2:], adapt_strategies):
        shade_plateaus(ax_stack)
        components = _timing_components(strategy)
        bottoms = np.zeros(len(phases))
        for col, label, color in components:
            vals = np.array([get(strategy, ef, ph, col) for ph in phases])
            vals = np.nan_to_num(vals)
            ax_stack.bar(x, vals, bar_width, bottom=bottoms,
                         label=label, color=color, alpha=0.85)
            bottoms += vals
        ax_stack.set_ylabel("time (ms)")
        ax_stack.set_title(f"time breakdown — {STRATEGY_LABELS.get(strategy, strategy)}")
        ax_stack.legend(fontsize=8, loc="upper left")
        ax_stack.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel("phase (▲ = plateau)")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(phase_labels, fontsize=6.5, rotation=35, ha="right")
    fig.suptitle(f"query timing overhead  [ef={ef}]", y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, f"timing_ef{ef}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def plot_timing_all_efs():
    timing_col = "mean_t_query_ms"
    if timing_col not in summary.columns:
        return

    fig, axes = plt.subplots(1, len(efs), figsize=(5 * len(efs), 4.5), sharey=False)
    if len(efs) == 1:
        axes = [axes]

    for ax, ef in zip(axes, efs):
        shade_plateaus(ax)
        for strategy in strategies:
            vals = [get(strategy, ef, ph, timing_col) for ph in phases]
            ax.plot(range(len(phases)), vals,
                    color=STRATEGY_COLORS.get(strategy),
                    marker="o", markersize=4, lw=1.8,
                    label=STRATEGY_LABELS.get(strategy, strategy))
        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels(phase_labels, fontsize=6.5, rotation=35, ha="right")
        ax.set_title(f"ef={ef}")
        ax.set_xlabel("phase (▲ = plateau)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("mean query time (ms)")
    fig.suptitle("query latency vs phase — all ef values", y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "timing_all_efs.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# --- run everything ----------------------------------------------------------

plot_recall_vs_phase()
plot_recall_delta()
plot_metrics(plot_ef)
plot_drift_vs_plateau()
plot_timing(plot_ef)
plot_timing_all_efs()

print(f"\nall plots written to {os.path.abspath(out_dir)}")
