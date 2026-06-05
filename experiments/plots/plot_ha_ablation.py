# Plots for the HardnessAdaptive ablation study.
#
# Reads summary.csv and adapt_log.csv from results_ha_ablation and produces:
#
#   1. recall_vs_phase.png         — all 9 variants at plot_ef, plateau-shaded
#   2. pareto.png                  — recall vs latency at hard phases, all efs
#   3. mechanism_bars.png          — recall delta + overhead at hard phases, ef=plot_ef
#   4. recall_vs_ef.png            — recall at hardest bin across ef sweep
#
# Usage:
#   python plot_ha_ablation.py --results_dir results_ha_ablation
#   python plot_ha_ablation.py --results_dir ... --plot_ef 100

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
parser.add_argument("--plot_ef", type=int, default=None)
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

params = {}
for _, row in load("params.csv").iterrows():
    params[row["param"]] = row["value"]

k = int(params.get("k", 10))
escalation_factor = int(params.get("escalation_factor", 3))

efs = sorted(summary["ef_search"].unique())
plot_ef = args.plot_ef or efs[len(efs) // 2]
if plot_ef not in efs:
    plot_ef = min(efs, key=lambda e: abs(e - plot_ef))

# phase metadata from summary
_ref = summary[(summary["strategy"] == "no_adaptation") &
               (summary["ef_search"] == efs[0])].sort_values("phase_idx")
phase_list = _ref[["phase_idx", "hardness_bin", "is_plateau",
                    "hardness_bin_lo", "hardness_bin_hi"]].to_dict("records")
phases = [p["phase_idx"] for p in phase_list]
hard_phases = [p["phase_idx"] for p in phase_list if p["phase_idx"] >= 8]

print(f"efs: {efs}  plot_ef: {plot_ef}")
print(f"phases: {len(phases)}  hard phases: {hard_phases}")

# ── display order and visual identity ──────────────────────────────────────────

DISPLAY_ORDER = [
    "no_adaptation",
    "static_ef_escalated",
    "rewire_only",
    "pool_only",
    "pool_rewire",
    "escalation_only",
    "rewire_escalation",
    "pool_escalation",
    "full",
]

COLORS = {
    "no_adaptation":      "#4C72B0",
    "static_ef_escalated":"#000000",
    "rewire_only":        "#C44E52",
    "pool_only":          "#8172B3",
    "pool_rewire":        "#937860",
    "escalation_only":    "#DD8452",
    "rewire_escalation":  "#CCB974",
    "pool_escalation":    "#64B5CD",
    "full":               "#55A868",
}

LABELS = {
    "no_adaptation":      "no adaptation",
    "static_ef_escalated":f"static ef×{escalation_factor}",
    "rewire_only":        "rewire",
    "pool_only":          "pool",
    "pool_rewire":        "pool + rewire",
    "escalation_only":    "escalation",
    "rewire_escalation":  "escalation + rewire",
    "pool_escalation":    "pool + escalation",
    "full":               "full (all three)",
}

MARKERS = {
    "no_adaptation":      "o",
    "static_ef_escalated":"*",
    "rewire_only":        "v",
    "pool_only":          "s",
    "pool_rewire":        "D",
    "escalation_only":    "^",
    "rewire_escalation":  "P",
    "pool_escalation":    "X",
    "full":               "h",
}

strategies_present = [s for s in DISPLAY_ORDER if s in summary["strategy"].unique()]


def shade_plateaus(ax):
    for i, p in enumerate(phase_list):
        if p["is_plateau"]:
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


# ── plot 1: recall vs phase (all strategies, plot_ef) ─────────────────────────

def plot_recall_vs_phase():
    fig, ax = plt.subplots(figsize=(13, 5))
    shade_plateaus(ax)

    for strat in strategies_present:
        recalls = [get(strat, plot_ef, ph, "mean_recall") for ph in phases]
        ls = "--" if strat in ("no_adaptation", "static_ef_escalated") else "-"
        lw = 2.2 if strat in ("no_adaptation", "static_ef_escalated", "escalation_only", "full") else 1.4
        ax.plot(range(len(phases)), recalls,
                color=COLORS[strat], marker=MARKERS[strat],
                markersize=5, lw=lw, ls=ls,
                label=LABELS[strat])

    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels(phase_labels, fontsize=7.5, rotation=35, ha="right")
    ax.set_xlabel("phase  (▲ = plateau)")
    ax.set_ylabel(f"Recall@{k}")
    ax.set_title(f"Ablation: Recall@{k} vs phase  [ef={plot_ef}]\n"
                 f"grey = plateau; dashed = baselines")
    ax.legend(fontsize=8, ncol=3, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "recall_vs_phase.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ── plot 2: pareto — recall vs latency at hard phases ─────────────────────────

def plot_pareto():
    hard = summary[summary["phase_idx"].isin(hard_phases)]
    pts = hard.groupby(["strategy", "ef_search"])[
        ["mean_recall", "mean_t_query_ms"]].mean().reset_index()

    fig, ax = plt.subplots(figsize=(9, 6))

    # draw the no_adaptation curve as a reference baseline
    no_pts = pts[pts["strategy"] == "no_adaptation"].sort_values("mean_t_query_ms")
    ax.plot(no_pts["mean_t_query_ms"], no_pts["mean_recall"],
            color=COLORS["no_adaptation"], lw=1.5, ls="--", alpha=0.5, zorder=1)

    # draw the static_ef_escalated curve to show the "free" upper bound
    se_pts = pts[pts["strategy"] == "static_ef_escalated"].sort_values("mean_t_query_ms")
    ax.plot(se_pts["mean_t_query_ms"], se_pts["mean_recall"],
            color=COLORS["static_ef_escalated"], lw=1.5, ls="--", alpha=0.4, zorder=1)

    ef_labels = {ef: str(ef) for ef in efs}

    for strat in strategies_present:
        sub = pts[pts["strategy"] == strat].sort_values("mean_t_query_ms")
        if sub.empty:
            continue
        lw = 1.8 if strat in ("escalation_only", "pool_escalation", "full",
                               "no_adaptation", "static_ef_escalated") else 1.0
        ax.scatter(sub["mean_t_query_ms"], sub["mean_recall"],
                   color=COLORS[strat], marker=MARKERS[strat],
                   s=70, zorder=3, label=LABELS[strat])
        ax.plot(sub["mean_t_query_ms"], sub["mean_recall"],
                color=COLORS[strat], lw=lw, alpha=0.6, zorder=2)

    # annotate ef values on no_adaptation curve
    for _, row in no_pts.iterrows():
        ax.annotate(f"ef={int(row['ef_search'])}",
                    (row["mean_t_query_ms"], row["mean_recall"]),
                    fontsize=6.5, color="#4C72B0", alpha=0.7,
                    xytext=(4, -10), textcoords="offset points")

    ax.set_xlabel("mean query latency (ms)")
    ax.set_ylabel(f"mean Recall@{k}")
    ax.set_title(f"Ablation: recall vs latency — hard phases (bins 7–9)\n"
                 f"dashed = no-adaptation baseline curve; "
                 f"dashed black = always ef×{escalation_factor}")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "pareto.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ── plot 3: mechanism decomposition bars (recall delta + overhead) ─────────────

def plot_mechanism_bars():
    hard = summary[(summary["ef_search"] == plot_ef) &
                   (summary["phase_idx"].isin(hard_phases))]

    no_r = hard[hard["strategy"] == "no_adaptation"]["mean_recall"].mean()
    no_t = hard[hard["strategy"] == "no_adaptation"]["mean_t_query_ms"].mean()

    strats = [s for s in strategies_present if s != "no_adaptation"]
    recall_deltas = []
    overheads = []
    for s in strats:
        sub = hard[hard["strategy"] == s]
        recall_deltas.append(sub["mean_recall"].mean() - no_r)
        overheads.append(sub["mean_t_query_ms"].mean() / no_t)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    x = np.arange(len(strats))
    bar_colors = [COLORS[s] for s in strats]
    bars = ax1.bar(x, recall_deltas, color=bar_colors, alpha=0.85)
    ax1.axhline(0, color="black", lw=0.8, ls="--")
    ax1.set_ylabel(f"ΔRecall@{k} vs no adaptation")
    ax1.set_title(f"Mechanism contribution — hard phases (bins 7–9)  [ef={plot_ef}]")
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, recall_deltas):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.001,
                 f"{val:+.3f}", ha="center", va="bottom", fontsize=8)

    ax2.bar(x, overheads, color=bar_colors, alpha=0.85)
    ax2.axhline(1.0, color="black", lw=0.8, ls="--")
    ax2.set_ylabel("query latency overhead (×)")
    ax2.set_xlabel("strategy")
    ax2.set_xticks(x)
    ax2.set_xticklabels([LABELS[s] for s in strats], rotation=25, ha="right", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    for i, val in enumerate(overheads):
        ax2.text(i, val + 0.05, f"{val:.2f}×", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, "mechanism_bars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ── plot 4: recall at hardest bin vs ef sweep ──────────────────────────────────

def plot_recall_vs_ef():
    # use hardest non-plateau phase (phase 10 = bin 9)
    hardest_phase = max(p["phase_idx"] for p in phase_list
                        if not p["is_plateau"] and p["hardness_bin"] == max(
                            p2["hardness_bin"] for p2 in phase_list))

    fig, ax = plt.subplots(figsize=(8, 5))

    for strat in strategies_present:
        recalls = [get(strat, ef, hardest_phase, "mean_recall") for ef in efs]
        ls = "--" if strat in ("no_adaptation", "static_ef_escalated") else "-"
        lw = 2.0 if strat in ("no_adaptation", "static_ef_escalated",
                               "escalation_only", "pool_escalation", "full") else 1.2
        ax.plot(efs, recalls, color=COLORS[strat], marker=MARKERS[strat],
                markersize=6, lw=lw, ls=ls, label=LABELS[strat])

    ax.set_xscale("log")
    ax.set_xticks(efs)
    ax.set_xticklabels([str(e) for e in efs])
    ax.set_xlabel("ef (log scale)")
    ax.set_ylabel(f"Recall@{k}")
    ax.set_title(f"Recall at hardest bin (phase {hardest_phase}) vs ef\n"
                 f"dashed = baselines; static ef×{escalation_factor} shown at base ef for comparison")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(out_dir, "recall_vs_ef.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ── run all ────────────────────────────────────────────────────────────────────

plot_recall_vs_phase()
plot_pareto()
plot_mechanism_bars()
plot_recall_vs_ef()

print(f"\nall outputs written to {os.path.abspath(out_dir)}")
