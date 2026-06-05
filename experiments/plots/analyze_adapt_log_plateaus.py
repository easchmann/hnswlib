# Analyse the adapt_log.csv produced by
# experiment_hardness_adaptive_sift_with_plateaus.py
#
# Assigns each adaptation event to a phase using the drift schedule from
# params.csv, then produces per-phase statistics and four plots:
#
#   1. edges_per_event vs phase - declining adaptation intensity over stream
#   2. events_per_1000_queries vs phase - normalised firing rate (flat = blind)
#   3. HA dist_comps vs threshold at events - did rewiring actually help?
#   4. first vs second occurrence recall/dist_comps comparison (plateau effect)
#
# Usage:
#   python analyze_adapt_log_plateaus.py --results_dir results_hardness_adaptive_sift_with_plateaus
#   python analyze_adapt_log_plateaus.py --results_dir ... --plot_ef 50

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


adapt_log = load_optional("adapt_log.csv")
if adapt_log is None:
    sys.exit("no adapt_log.csv found — nothing to analyse")

summary = load("summary.csv")

params = {}
params_path = os.path.join(args.results_dir, "params.csv")
if os.path.exists(params_path):
    for _, row in pd.read_csv(params_path).iterrows():
        params[row["param"]] = row["value"]

k            = int(params.get("k", 10))
schedule_str = params.get("drift_schedule", "")
if not schedule_str:
    sys.exit("drift_schedule not found in params.csv")

# parse schedule → eval phases (drop warmup = first entry)
raw_phases = []
for entry in schedule_str.split(","):
    b, n = entry.strip().split(":")
    raw_phases.append({"bin": int(b), "n_queries": int(n)})

# mark plateaus: second+ occurrence of the same bin
first_seen = set()
for p in raw_phases:
    p["is_plateau"] = p["bin"] in first_seen
    first_seen.add(p["bin"])

eval_phases = raw_phases[1:]   # drop warmup
n_eval_phases = len(eval_phases)
phase_sizes   = [p["n_queries"] for p in eval_phases]
total_eval_q  = sum(phase_sizes)
cum_phase_q   = np.cumsum([0] + phase_sizes)   # boundaries in query space

strategies = sorted(adapt_log["strategy"].unique())
efs        = sorted(adapt_log["ef"].unique())

plot_ef = args.plot_ef or efs[len(efs) // 2]
if plot_ef not in efs:
    plot_ef = min(efs, key=lambda e: abs(e - plot_ef))
    print(f"warning: requested ef not found, using {plot_ef}")

print(f"strategies: {strategies}")
print(f"efs:        {efs}")
print(f"plot_ef:    {plot_ef}")
print(f"eval phases: {n_eval_phases}  "
      f"({sum(p['is_plateau'] for p in eval_phases)} plateau)\n")

STRATEGY_COLORS = {
    "poolAndRewire":     "#DD8452",
    "hardness_adaptive": "#55A868",
}
STRATEGY_LABELS = {
    "poolAndRewire":     "PoolAndRewire",
    "hardness_adaptive": "HardnessAdaptive",
}


# ---------------------------------------------------------------------------
# Phase assignment
# ---------------------------------------------------------------------------
# For each (strategy, ef) group events are numbered 1..N via update_n.
# The event rate is proportional to query count for both strategies
# (PAR: 1 per cooldown queries; HA: 1 per hard_rewire_cooldown hard queries,
# and nearly all eval queries are hard).  We assign events to phases by
# mapping cumulative event index to cumulative query count.

def assign_phases(df):
    """Add phase_idx column to a single (strategy, ef) group."""
    df = df.sort_values("update_n").copy()
    n  = len(df)
    boundaries = np.round(
        cum_phase_q[1:] / total_eval_q * n
    ).astype(int)
    boundaries = np.clip(boundaries, 0, n)

    phase_col = np.empty(n, dtype=int)
    prev = 0
    for phase_idx, boundary in enumerate(boundaries):
        phase_col[prev:boundary] = phase_idx
        prev = boundary
    df["phase_idx"] = phase_col
    return df


groups = []
for (strategy, ef), grp in adapt_log.groupby(["strategy", "ef"]):
    groups.append(assign_phases(grp))
adapt_log = pd.concat(groups, ignore_index=True)

# attach phase metadata
phase_df = pd.DataFrame([
    {"phase_idx": i, "bin": p["bin"],
     "is_plateau": p["is_plateau"], "n_queries": p["n_queries"]}
    for i, p in enumerate(eval_phases)
])
adapt_log = adapt_log.merge(phase_df, on="phase_idx", how="left")


# ---------------------------------------------------------------------------
# Per-phase statistics
# ---------------------------------------------------------------------------

def phase_stats(df):
    rows = []
    for (strategy, ef, phase_idx), grp in df.groupby(
            ["strategy", "ef", "phase_idx"]):
        meta = eval_phases[phase_idx]
        rows.append({
            "strategy":              strategy,
            "ef":                    ef,
            "phase_idx":             phase_idx,
            "bin":                   meta["bin"],
            "is_plateau":            meta["is_plateau"],
            "n_queries":             meta["n_queries"],
            "n_events":              len(grp),
            "total_edges":           grp["edges_added"].sum(),
            "edges_per_event":       grp["edges_added"].mean(),
            "events_per_1000q":      len(grp) / meta["n_queries"] * 1000,
            "mean_dist_comps":       grp["dist_comps"].mean(),
            "mean_threshold":        grp["threshold"].mean(),
            "mean_t_rewire_ms":      grp["t_rewire_ms"].mean(),
        })
    return pd.DataFrame(rows)

stats = phase_stats(adapt_log)

# save
path = os.path.join(out_dir, "adapt_log_phase_stats.csv")
stats.round(4).to_csv(path, index=False)
print(f"saved {path}")


# ---------------------------------------------------------------------------
# Table 1 — side-by-side per-phase summary at plot_ef
# ---------------------------------------------------------------------------

def print_phase_table(ef):
    sub = stats[stats["ef"] == ef]
    ha  = sub[sub["strategy"] == "hardness_adaptive"].set_index("phase_idx")
    par = sub[sub["strategy"] == "poolAndRewire"].set_index("phase_idx")

    header = (f"{'ph':>3}  {'bin':>3}  {'plat':>5}  {'n_q':>6}  "
              f"{'HA_ev':>6}  {'HA_e/ev':>8}  {'HA_dcomp':>9}  "
              f"{'PAR_ev':>7}  {'PAR_e/ev':>9}")
    print(f"\n--- adaptation events per phase  [ef={ef}] ---")
    print(header)
    for i, p in enumerate(eval_phases):
        ha_row  = ha.loc[i]  if i in ha.index  else None
        par_row = par.loc[i] if i in par.index else None
        flag = "▲" if p["is_plateau"] else " "
        ha_ev   = f"{ha_row['n_events']:6.0f}"    if ha_row  is not None else f"{'—':>6}"
        ha_eev  = f"{ha_row['edges_per_event']:8.2f}" if ha_row is not None else f"{'—':>8}"
        ha_dc   = f"{ha_row['mean_dist_comps']:9.0f}" if ha_row is not None and not np.isnan(ha_row['mean_dist_comps']) else f"{'—':>9}"
        par_ev  = f"{par_row['n_events']:7.0f}"   if par_row is not None else f"{'—':>7}"
        par_eev = f"{par_row['edges_per_event']:9.2f}" if par_row is not None else f"{'—':>9}"
        print(f"{i:>3}{flag} {p['bin']:>3}  {str(p['is_plateau']):>5}  {p['n_queries']:>6}  "
              f"{ha_ev}  {ha_eev}  {ha_dc}  {par_ev}  {par_eev}")


print_phase_table(plot_ef)


# ---------------------------------------------------------------------------
# Table 2 — first vs second occurrence of same bin
# ---------------------------------------------------------------------------

def print_occurrence_table(ef):
    sub       = stats[stats["ef"] == ef]
    warmup_ef = float(params.get("warmup_ef", 50))
    thr_scale = ef / warmup_ef
    repeated_bins = sorted(set(
        p["bin"] for p in eval_phases if p["is_plateau"]
    ))

    print(f"\n--- first vs second occurrence of repeated bins  [ef={ef}, thresh×{thr_scale:.1f}] ---")
    print(f"{'bin':>4}  {'occ':>5}  {'ph':>3}  "
          f"{'HA_ev':>6}  {'HA_e/ev':>8}  {'HA_dcomp':>9}  {'HA_thresh_eff':>14}  "
          f"{'PAR_ev':>7}  {'PAR_e/ev':>9}")

    for b in repeated_bins:
        occurrences = [i for i, p in enumerate(eval_phases) if p["bin"] == b]
        for occ_n, phase_idx in enumerate(occurrences):
            ha_row  = sub[(sub["strategy"] == "hardness_adaptive") &
                          (sub["phase_idx"] == phase_idx)]
            par_row = sub[(sub["strategy"] == "poolAndRewire") &
                          (sub["phase_idx"] == phase_idx)]
            label = "drift" if occ_n == 0 else "plateau"

            ha_ev   = f"{ha_row['n_events'].values[0]:6.0f}"        if len(ha_row)  else f"{'—':>6}"
            ha_eev  = f"{ha_row['edges_per_event'].values[0]:8.2f}" if len(ha_row)  else f"{'—':>8}"
            ha_dc   = f"{ha_row['mean_dist_comps'].values[0]:9.0f}" if len(ha_row) and not np.isnan(ha_row['mean_dist_comps'].values[0]) else f"{'—':>9}"
            ha_thr  = f"{ha_row['mean_threshold'].values[0] * thr_scale:14.0f}" if len(ha_row) and not np.isnan(ha_row['mean_threshold'].values[0]) else f"{'—':>14}"
            par_ev  = f"{par_row['n_events'].values[0]:7.0f}"       if len(par_row) else f"{'—':>7}"
            par_eev = f"{par_row['edges_per_event'].values[0]:9.2f}"if len(par_row) else f"{'—':>9}"
            print(f"{b:>4}  {label:>5}  {phase_idx:>3}  "
                  f"{ha_ev}  {ha_eev}  {ha_dc}  {ha_thr}  {par_ev}  {par_eev}")


print_occurrence_table(plot_ef)


# ---------------------------------------------------------------------------
# Helpers shared by plots
# ---------------------------------------------------------------------------

def shade_plateaus(ax):
    for i, p in enumerate(eval_phases):
        if p["is_plateau"]:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.12, color="grey", zorder=0)


phase_labels = [
    f"ph{i}{'▲' if p['is_plateau'] else ''}\nbin{p['bin']}"
    for i, p in enumerate(eval_phases)
]
x = np.arange(n_eval_phases)


# ---------------------------------------------------------------------------
# Plot 1 — edges_per_event vs phase (all ef values)
# ---------------------------------------------------------------------------

def plot_edges_per_event():
    fig, axes = plt.subplots(1, len(efs), figsize=(5 * len(efs), 4.5), sharey=False)
    if len(efs) == 1:
        axes = [axes]

    for ax, ef in zip(axes, efs):
        shade_plateaus(ax)
        for strategy in strategies:
            sub = stats[(stats["strategy"] == strategy) & (stats["ef"] == ef)]
            sub = sub.sort_values("phase_idx")
            vals = sub.set_index("phase_idx").reindex(range(n_eval_phases))["edges_per_event"]
            ax.plot(x, vals, color=STRATEGY_COLORS.get(strategy),
                    marker="o", markersize=5, lw=1.8,
                    label=STRATEGY_LABELS.get(strategy, strategy))
        ax.set_xticks(x)
        ax.set_xticklabels(phase_labels, fontsize=7, rotation=35, ha="right")
        ax.set_title(f"ef={ef}")
        ax.set_xlabel("phase (▲ = plateau)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("mean edges added per event")
    fig.suptitle(
        "rewiring intensity — edges per event vs phase\n"
        "declining = graph saturating; plateau dip = less useful edges found",
        y=1.02
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "adapt_edges_per_event.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ---------------------------------------------------------------------------
# Plot 2 — events_per_1000_queries vs phase (normalised firing rate)
# ---------------------------------------------------------------------------

def plot_firing_rate():
    fig, axes = plt.subplots(1, len(efs), figsize=(5 * len(efs), 4.5), sharey=False)
    if len(efs) == 1:
        axes = [axes]

    for ax, ef in zip(axes, efs):
        shade_plateaus(ax)
        for strategy in strategies:
            sub = stats[(stats["strategy"] == strategy) & (stats["ef"] == ef)]
            sub = sub.sort_values("phase_idx")
            vals = sub.set_index("phase_idx").reindex(range(n_eval_phases))["events_per_1000q"]
            ax.plot(x, vals, color=STRATEGY_COLORS.get(strategy),
                    marker="o", markersize=5, lw=1.8,
                    label=STRATEGY_LABELS.get(strategy, strategy))
        ax.set_xticks(x)
        ax.set_xticklabels(phase_labels, fontsize=7, rotation=35, ha="right")
        ax.set_title(f"ef={ef}")
        ax.set_xlabel("phase (▲ = plateau)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("adaptation events per 1000 queries")
    fig.suptitle(
        "normalised adaptation firing rate vs phase\n"
        "flat line = controller fires uniformly regardless of drift/plateau",
        y=1.02
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "adapt_firing_rate.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ---------------------------------------------------------------------------
# Plot 3 — HA: dist_comps at adaptation events vs threshold (one ef panel)
# ---------------------------------------------------------------------------

def plot_ha_dist_comps(ef):
    ha_stats = stats[(stats["strategy"] == "hardness_adaptive") &
                     (stats["ef"] == ef)].sort_values("phase_idx")
    if ha_stats["mean_dist_comps"].isna().all():
        print("skipping dist_comps plot: no dist_comps in HA adapt_log")
        return

    # mean dist_comps per phase for each strategy (all queries, from summary.csv)
    def summary_dc(strategy):
        sub = summary[(summary["strategy"] == strategy) &
                      (summary["ef_search"] == ef)].sort_values("phase_idx")
        return sub.set_index("phase_idx").reindex(range(n_eval_phases))["mean_base_dist_comps"]

    fig, ax = plt.subplots(figsize=(10, 4))
    shade_plateaus(ax)

    dc_events = ha_stats.set_index("phase_idx").reindex(range(n_eval_phases))["mean_dist_comps"]
    dc_ha     = summary_dc("hardness_adaptive")
    dc_base   = summary_dc("no_adaptation")

    ax.plot(x, dc_events, color="#55A868", marker="o", markersize=5, lw=1.8,
            label="HA: mean dist_comps at rewire events (hard queries only)")
    ax.plot(x, dc_ha,     color="#55A868", marker="^", markersize=5, lw=1.4,
            ls=":", label="HA: mean dist_comps all queries")
    ax.plot(x, dc_base,   color="#777777", marker="s", markersize=4, lw=1.4,
            ls="--", label="no_adaptation: mean dist_comps all queries")

    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels, fontsize=7, rotation=35, ha="right")
    ax.set_xlabel("phase (▲ = plateau)")
    ax.set_ylabel("distance computations")
    ax.set_title(
        f"HardnessAdaptive — dist_comps vs phase  [ef={ef}]\n"
        f"rewire events (solid) vs all-query mean (dotted) vs no-adaptation baseline (dashed)"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"adapt_ha_dist_comps_ef{ef}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ---------------------------------------------------------------------------
# Plot 4 — first vs second occurrence: recall and dist_comps side-by-side
# ---------------------------------------------------------------------------

def plot_first_vs_second(ef):
    repeated_bins = sorted(set(
        p["bin"] for p in eval_phases if p["is_plateau"]
    ))
    if not repeated_bins:
        print("skipping first_vs_second: no repeated bins")
        return

    # gather recall from summary.csv
    sum_ef = summary[summary["ef_search"] == ef]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    width = 0.25
    n_bins = len(repeated_bins)
    x_b = np.arange(n_bins)

    for col_i, (metric, ylabel, src) in enumerate([
        ("mean_recall", f"Recall@{k}", "summary"),
        ("mean_dist_comps",  "dist comps at events",  "stats"),
    ]):
        ax = axes[col_i]
        for strat_i, strategy in enumerate(strategies):
            first_vals = []
            second_vals = []
            for b in repeated_bins:
                occurrences = [i for i, p in enumerate(eval_phases) if p["bin"] == b]
                first_ph, second_ph = occurrences[0], occurrences[1]
                if src == "summary":
                    sub = sum_ef[sum_ef["strategy"] == strategy]
                    def _get_recall(ph):
                        row = sub[sub["phase_idx"] == ph]
                        return float(row[metric].values[0]) if len(row) else float("nan")
                    first_vals.append(_get_recall(first_ph))
                    second_vals.append(_get_recall(second_ph))
                else:
                    sub = stats[(stats["strategy"] == strategy) & (stats["ef"] == ef)]
                    def _get_stat(ph):
                        row = sub[sub["phase_idx"] == ph]
                        return float(row[metric].values[0]) if len(row) else float("nan")
                    first_vals.append(_get_stat(first_ph))
                    second_vals.append(_get_stat(second_ph))

            offset = (strat_i - (len(strategies) - 1) / 2) * width
            color = STRATEGY_COLORS.get(strategy, "grey")
            label = STRATEGY_LABELS.get(strategy, strategy)
            ax.bar(x_b + offset - width / 2, first_vals,  width, color=color, alpha=0.5, label=f"{label} drift",  hatch="")
            ax.bar(x_b + offset + width / 2, second_vals, width, color=color, alpha=0.9, label=f"{label} plateau", hatch="//")

        ax.set_xticks(x_b)
        ax.set_xticklabels([f"bin {b}" for b in repeated_bins])
        ax.set_ylabel(ylabel)
        ax.set_title(f"{'recall' if col_i == 0 else 'dist_comps at events'}  [ef={ef}]")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "first arrival (drift) vs repeated visit (plateau) at same hardness bin\n"
        "no change = controller does not learn from prior exposure",
        y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, f"adapt_first_vs_second_ef{ef}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")



plot_edges_per_event()
plot_firing_rate()
plot_ha_dist_comps(plot_ef)
plot_first_vs_second(plot_ef)

print(f"\nall outputs written to {os.path.abspath(out_dir)}")
