# Plots for the HardnessAdaptive pool-improvement ablation.
#
# Reads summary.csv (and optionally per_query.csv) from results_ha_pool_improvements
# and produces:
#
#   1. recall_vs_phase.png    — all strategies at plot_ef, plateau-shaded
#   2. recall_delta_bars.png  — ΔRecall and latency overhead at hard phases
#   3. recall_vs_ef.png       — recall at hardest phase across ef sweep
#   4. pareto.png             — recall vs mean query latency (hard phases, all efs)
#
# Usage:
#   python plot_ha_pool_improvements.py \
#       --results_dir experiments/results/results_ha_pool_improvements
#   python plot_ha_pool_improvements.py --results_dir ... --plot_ef 50

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
                    help="ef for single-ef plots (default: median ef)")
parser.add_argument("--out_dir", default=None)
args = parser.parse_args()

out_dir = args.out_dir or args.results_dir
os.makedirs(out_dir, exist_ok=True)


def load(filename):
    path = os.path.join(args.results_dir, filename)
    if not os.path.exists(path):
        sys.exit(f"missing: {path}")
    return pd.read_csv(path)


summary = load("summary.csv")

efs = sorted(summary["ef_search"].unique())
plot_ef = args.plot_ef or efs[len(efs) // 2]
if plot_ef not in efs:
    plot_ef = min(efs, key=lambda e: abs(e - plot_ef))

_ref = (summary[(summary["strategy"] == "no_adaptation") &
                (summary["ef_search"] == efs[0])]
        .sort_values("phase_idx"))
phase_list = _ref[["phase_idx", "hardness_bin", "is_plateau",
                    "hardness_bin_lo", "hardness_bin_hi"]].to_dict("records")
phases = [p["phase_idx"] for p in phase_list]

# "hard" = last 4 phases (bins 7-9 including plateaus)
hard_phases = [p["phase_idx"] for p in phase_list if p["hardness_bin"] >= 7]

k = 10

print(f"efs: {efs}   plot_ef: {plot_ef}")
print(f"phases: {len(phases)}   hard_phases: {hard_phases}")

# ── visual identity ────────────────────────────────────────────────────────────

DISPLAY_ORDER = [
    "no_adaptation",
    "ha_original",
    "ha_redundancy",
    "ha_multi_ep",
    "ha_continuous",
    "ha_all",
]

COLORS = {
    "no_adaptation": "#4C72B0",
    "ha_original":   "#C44E52",
    "ha_redundancy": "#8172B3",
    "ha_multi_ep":   "#64B5CD",
    "ha_continuous": "#DD8452",
    "ha_all":        "#55A868",
}

LABELS = {
    "no_adaptation": "no adaptation",
    "ha_original":   "original (dist evict, top-1, binary ef)",
    "ha_redundancy": "+ redundancy eviction",
    "ha_multi_ep":   "+ multi entry-point (top-3)",
    "ha_continuous": "+ continuous ef scaling",
    "ha_all":        "all improvements",
}

MARKERS = {
    "no_adaptation": "o",
    "ha_original":   "v",
    "ha_redundancy": "s",
    "ha_multi_ep":   "D",
    "ha_continuous": "^",
    "ha_all":        "h",
}

present = [s for s in DISPLAY_ORDER if s in summary["strategy"].unique()]


def shade_plateaus(ax):
    for p in phase_list:
        if p["is_plateau"]:
            i = p["phase_idx"]
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.12, color="grey", zorder=0)


def phase_label(p):
    mark = "▲" if p["is_plateau"] else ""
    return f"ph{p['phase_idx']}{mark}\nbin{p['hardness_bin']}"


phase_labels = [phase_label(p) for p in phase_list]


def get(strategy, ef, phase_idx, col):
    row = summary[(summary["strategy"] == strategy) &
                  (summary["ef_search"] == ef) &
                  (summary["phase_idx"] == phase_idx)]
    return float(row[col].iloc[0]) if not row.empty else float("nan")


# ── plot 1: recall vs phase ────────────────────────────────────────────────────

def plot_recall_vs_phase():
    fig, ax = plt.subplots(figsize=(13, 5))
    shade_plateaus(ax)

    for strat in present:
        recalls = [get(strat, plot_ef, ph, "mean_recall") for ph in phases]
        ls = "--" if strat == "no_adaptation" else "-"
        lw = 2.2 if strat in ("no_adaptation", "ha_all") else 1.5
        ax.plot(range(len(phases)), recalls,
                color=COLORS[strat], marker=MARKERS[strat],
                markersize=5, lw=lw, ls=ls, label=LABELS[strat])

    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels(phase_labels, fontsize=7.5, rotation=35, ha="right")
    ax.set_xlabel("phase  (▲ = plateau at same hardness bin)")
    ax.set_ylabel(f"Recall@{k}")
    ax.set_title(f"Pool improvement ablation: Recall@{k} vs phase  [ef={plot_ef}]\n"
                 f"grey = plateau phases; dashed = no-adaptation baseline")
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "recall_vs_phase.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ── plot 2: recall delta + overhead bars at hard phases ───────────────────────

def plot_recall_delta_bars():
    hard = summary[(summary["ef_search"] == plot_ef) &
                   (summary["phase_idx"].isin(hard_phases))]
    no_r = hard[hard["strategy"] == "no_adaptation"]["mean_recall"].mean()
    no_t = hard[hard["strategy"] == "no_adaptation"]["mean_t_query_ms"].mean()

    strats = [s for s in present if s != "no_adaptation"]
    deltas    = []
    overheads = []
    for s in strats:
        sub = hard[hard["strategy"] == s]
        deltas.append(sub["mean_recall"].mean() - no_r)
        overheads.append(sub["mean_t_query_ms"].mean() / no_t)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    x = np.arange(len(strats))
    bar_colors = [COLORS[s] for s in strats]

    bars = ax1.bar(x, deltas, color=bar_colors, alpha=0.85)
    ax1.axhline(0, color="black", lw=0.8, ls="--")
    ax1.set_ylabel(f"ΔRecall@{k} vs no adaptation")
    ax1.set_title(f"Pool improvement contribution — hard phases (bin ≥ 7)  [ef={plot_ef}]")
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, deltas):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.001,
                 f"{val:+.3f}", ha="center", va="bottom", fontsize=9)

    ax2.bar(x, overheads, color=bar_colors, alpha=0.85)
    ax2.axhline(1.0, color="black", lw=0.8, ls="--")
    ax2.set_ylabel("query latency overhead (×baseline)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([LABELS[s] for s in strats], rotation=20, ha="right", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    for i, val in enumerate(overheads):
        ax2.text(i, val + 0.05, f"{val:.2f}×", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    path = os.path.join(out_dir, "recall_delta_bars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ── plot 3: recall vs ef at hardest phase ─────────────────────────────────────

def plot_recall_vs_ef():
    hardest_phase = max(
        p["phase_idx"] for p in phase_list
        if not p["is_plateau"] and p["hardness_bin"] == max(q["hardness_bin"] for q in phase_list)
    )
    fig, ax = plt.subplots(figsize=(8, 5))

    for strat in present:
        recalls = [get(strat, ef, hardest_phase, "mean_recall") for ef in efs]
        ls = "--" if strat == "no_adaptation" else "-"
        lw = 2.2 if strat in ("no_adaptation", "ha_all") else 1.5
        ax.plot(efs, recalls, color=COLORS[strat], marker=MARKERS[strat],
                markersize=6, lw=lw, ls=ls, label=LABELS[strat])

    ax.set_xscale("log")
    ax.set_xticks(efs)
    ax.set_xticklabels([str(e) for e in efs])
    ax.set_xlabel("ef (log scale)")
    ax.set_ylabel(f"Recall@{k}")
    ax.set_title(f"Recall at hardest phase (phase {hardest_phase}) vs ef")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(out_dir, "recall_vs_ef.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ── plot 4: pareto (recall vs latency, hard phases, all efs) ──────────────────

def plot_pareto():
    hard = summary[summary["phase_idx"].isin(hard_phases)]
    pts = (hard.groupby(["strategy", "ef_search"])
           [["mean_recall", "mean_t_query_ms"]].mean().reset_index())

    fig, ax = plt.subplots(figsize=(9, 6))

    no_pts = pts[pts["strategy"] == "no_adaptation"].sort_values("mean_t_query_ms")
    ax.plot(no_pts["mean_t_query_ms"], no_pts["mean_recall"],
            color=COLORS["no_adaptation"], lw=1.5, ls="--", alpha=0.5, zorder=1)
    for _, row in no_pts.iterrows():
        ax.annotate(f"ef={int(row['ef_search'])}",
                    (row["mean_t_query_ms"], row["mean_recall"]),
                    fontsize=6.5, color=COLORS["no_adaptation"], alpha=0.7,
                    xytext=(4, -10), textcoords="offset points")

    for strat in present:
        sub = pts[pts["strategy"] == strat].sort_values("mean_t_query_ms")
        if sub.empty:
            continue
        lw = 2.0 if strat in ("no_adaptation", "ha_all") else 1.0
        ax.scatter(sub["mean_t_query_ms"], sub["mean_recall"],
                   color=COLORS[strat], marker=MARKERS[strat],
                   s=70, zorder=3, label=LABELS[strat])
        ax.plot(sub["mean_t_query_ms"], sub["mean_recall"],
                color=COLORS[strat], lw=lw, alpha=0.6, zorder=2)

    ax.set_xlabel("mean query latency (ms)")
    ax.set_ylabel(f"mean Recall@{k}")
    ax.set_title(f"Recall vs latency — hard phases (bin ≥ 7), all efs\n"
                 f"dashed = no-adaptation reference curve")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "pareto.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ── run all ────────────────────────────────────────────────────────────────────

plot_recall_vs_phase()
plot_recall_delta_bars()
plot_recall_vs_ef()
plot_pareto()

print(f"\nall outputs written to {os.path.abspath(out_dir)}")
