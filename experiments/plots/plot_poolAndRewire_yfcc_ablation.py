# Plots for PoolAndRewire YFCC ablation study.
#
# Reads summary.csv and params.csv from results_poolAndRewire_yfcc_ablation and produces:
#
#   1. recall_vs_sigma.png   — all variants at plot_ef across shift_sigma
#   2. pareto.png            — recall vs latency at high drift, all efs
#   3. mechanism_bars.png    — recall delta + overhead at high drift, ef=plot_ef
#   4. recall_vs_ef.png      — recall at highest sigma across ef sweep
#
# Usage:
#   python plot_poolAndRewire_yfcc_ablation.py --results_dir results_poolAndRewire_yfcc_ablation
#   python plot_poolAndRewire_yfcc_ablation.py --results_dir ... --plot_ef 100

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
parser.add_argument("--plot_ef",     type=int, default=None)
parser.add_argument("--out_dir",     default=None)
args = parser.parse_args()

out_dir = args.out_dir or args.results_dir
os.makedirs(out_dir, exist_ok=True)


def load(filename):
    path = os.path.join(args.results_dir, filename)
    if not os.path.exists(path):
        sys.exit(f"missing: {path}")
    return pd.read_csv(path)


summary = load("summary.csv")
params  = {row["param"]: row["value"] for _, row in load("params.csv").iterrows()}

k = int(params.get("k", 10))

efs      = sorted(summary["ef_search"].unique())
sigmas   = sorted(summary["shift_sigma"].unique())
plot_ef  = args.plot_ef or efs[len(efs) // 2]
if plot_ef not in efs:
    plot_ef = min(efs, key=lambda e: abs(e - plot_ef))

# "high drift" = top 40% of sigmas
high_drift_sigmas = sigmas[len(sigmas) * 6 // 10:]

print(f"efs: {efs}  plot_ef: {plot_ef}")
print(f"sigmas: {sigmas}  high_drift: {high_drift_sigmas}")

DISPLAY_ORDER = [
    "no_adaptation",
    "rewire_only",
    "highway_only",
    "pool_only",
    "pool_rewire",
    "pool_highway",
    "full",
]

COLORS = {
    "no_adaptation": "#4C72B0",
    "rewire_only":   "#C44E52",
    "highway_only":  "#CCB974",
    "pool_only":     "#8172B3",
    "pool_rewire":   "#DD8452",
    "pool_highway":  "#64B5CD",
    "full":          "#55A868",
}

LABELS = {
    "no_adaptation": "no adaptation",
    "rewire_only":   "rewire only",
    "highway_only":  "highway only",
    "pool_only":     "pool only",
    "pool_rewire":   "pool + rewire",
    "pool_highway":  "pool + highway",
    "full":          "full (pool + rewire + highway)",
}

MARKERS = {
    "no_adaptation": "o",
    "rewire_only":   "v",
    "highway_only":  "P",
    "pool_only":     "s",
    "pool_rewire":   "^",
    "pool_highway":  "X",
    "full":          "h",
}

strategies_present = [s for s in DISPLAY_ORDER if s in summary["strategy"].unique()]


def get(strategy, ef, sigma, col):
    row = summary[(summary["strategy"] == strategy) &
                  (summary["ef_search"] == ef) &
                  (summary["shift_sigma"] == sigma)]
    return float(row[col].iloc[0]) if not row.empty else float("nan")


# ── plot 1: recall vs sigma ────────────────────────────────────────────────────

def plot_recall_vs_sigma():
    fig, ax = plt.subplots(figsize=(11, 5))

    for strat in strategies_present:
        recalls = [get(strat, plot_ef, s, "mean_recall") for s in sigmas]
        ls = "--" if strat == "no_adaptation" else "-"
        lw = 2.2 if strat in ("no_adaptation", "full") else 1.4
        ax.plot(range(len(sigmas)), recalls,
                color=COLORS[strat], marker=MARKERS[strat],
                markersize=5, lw=lw, ls=ls, label=LABELS[strat])

    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels([f"σ={s}" for s in sigmas], fontsize=8, rotation=30, ha="right")
    ax.set_xlabel("shift sigma (drift magnitude)")
    ax.set_ylabel(f"Recall@{k}")
    ax.set_title(f"PoolAndRewire YFCC ablation: Recall@{k} vs drift  [ef={plot_ef}]\n"
                 f"dashed = baseline; high drift region = σ≥{high_drift_sigmas[0]}")
    ax.axvspan(sigmas.index(high_drift_sigmas[0]) - 0.5, len(sigmas) - 0.5,
               alpha=0.07, color="orange", zorder=0, label="high drift region")
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "recall_vs_sigma.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ── plot 2: pareto — recall vs latency at high drift ──────────────────────────

def plot_pareto():
    high = summary[summary["shift_sigma"].isin(high_drift_sigmas)]
    pts  = high.groupby(["strategy", "ef_search"])[
        ["mean_recall", "mean_t_query_ms"]].mean().reset_index()

    fig, ax = plt.subplots(figsize=(9, 6))

    no_pts = pts[pts["strategy"] == "no_adaptation"].sort_values("mean_t_query_ms")
    ax.plot(no_pts["mean_t_query_ms"], no_pts["mean_recall"],
            color=COLORS["no_adaptation"], lw=1.5, ls="--", alpha=0.5, zorder=1)

    for strat in strategies_present:
        sub = pts[pts["strategy"] == strat].sort_values("mean_t_query_ms")
        if sub.empty:
            continue
        lw = 1.8 if strat in ("no_adaptation", "full", "pool_only") else 1.0
        ax.scatter(sub["mean_t_query_ms"], sub["mean_recall"],
                   color=COLORS[strat], marker=MARKERS[strat],
                   s=70, zorder=3, label=LABELS[strat])
        ax.plot(sub["mean_t_query_ms"], sub["mean_recall"],
                color=COLORS[strat], lw=lw, alpha=0.6, zorder=2)

    for _, row in no_pts.iterrows():
        ax.annotate(f"ef={int(row['ef_search'])}",
                    (row["mean_t_query_ms"], row["mean_recall"]),
                    fontsize=6.5, color="#4C72B0", alpha=0.7,
                    xytext=(4, -10), textcoords="offset points")

    ax.set_xlabel("mean query latency (ms)")
    ax.set_ylabel(f"mean Recall@{k}")
    ax.set_title(f"PoolAndRewire YFCC ablation: recall vs latency — high drift (σ≥{high_drift_sigmas[0]})\n"
                 f"dashed = no_adaptation baseline")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "pareto.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ── plot 3: mechanism bars ─────────────────────────────────────────────────────

def plot_mechanism_bars():
    high = summary[(summary["ef_search"] == plot_ef) &
                   (summary["shift_sigma"].isin(high_drift_sigmas))]

    no_r = high[high["strategy"] == "no_adaptation"]["mean_recall"].mean()
    no_t = high[high["strategy"] == "no_adaptation"]["mean_t_query_ms"].mean()

    strats = [s for s in strategies_present if s != "no_adaptation"]
    recall_deltas = [high[high["strategy"] == s]["mean_recall"].mean() - no_r for s in strats]
    overheads     = [high[high["strategy"] == s]["mean_t_query_ms"].mean() / no_t for s in strats]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    x = np.arange(len(strats))
    bar_colors = [COLORS[s] for s in strats]

    bars = ax1.bar(x, recall_deltas, color=bar_colors, alpha=0.85)
    ax1.axhline(0, color="black", lw=0.8, ls="--")
    ax1.set_ylabel(f"ΔRecall@{k} vs no adaptation")
    ax1.set_title(f"Mechanism contribution — high drift (σ≥{high_drift_sigmas[0]})  [ef={plot_ef}]")
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, recall_deltas):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.001,
                 f"{val:+.3f}", ha="center", va="bottom", fontsize=8)

    ax2.bar(x, overheads, color=bar_colors, alpha=0.85)
    ax2.axhline(1.0, color="black", lw=0.8, ls="--")
    ax2.set_ylabel("query latency overhead (×)")
    ax2.set_xlabel("strategy")
    ax2.set_xticks(x)
    ax2.set_xticklabels([LABELS[s] for s in strats], rotation=20, ha="right", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    for i, val in enumerate(overheads):
        ax2.text(i, val + 0.02, f"{val:.2f}×", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, "mechanism_bars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ── plot 4: recall vs ef at highest sigma ─────────────────────────────────────

def plot_recall_vs_ef():
    max_sigma = max(sigmas)
    fig, ax = plt.subplots(figsize=(8, 5))

    for strat in strategies_present:
        recalls = [get(strat, ef, max_sigma, "mean_recall") for ef in efs]
        ls = "--" if strat == "no_adaptation" else "-"
        lw = 2.0 if strat in ("no_adaptation", "full", "pool_only") else 1.2
        ax.plot(efs, recalls, color=COLORS[strat], marker=MARKERS[strat],
                markersize=6, lw=lw, ls=ls, label=LABELS[strat])

    ax.set_xscale("log")
    ax.set_xticks(efs)
    ax.set_xticklabels([str(e) for e in efs])
    ax.set_xlabel("ef (log scale)")
    ax.set_ylabel(f"Recall@{k}")
    ax.set_title(f"Recall at maximum drift (σ={max_sigma}) vs ef\n"
                 f"dashed = no_adaptation baseline")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(out_dir, "recall_vs_ef.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ── run all ────────────────────────────────────────────────────────────────────

plot_recall_vs_sigma()
plot_pareto()
plot_mechanism_bars()
plot_recall_vs_ef()

print(f"\nall outputs written to {os.path.abspath(out_dir)}")
