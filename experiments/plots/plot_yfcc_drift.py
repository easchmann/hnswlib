# Generate figures from the CSVs produced by experiment_yfcc_drift
#
# Usage:
#   python plot_yfcc_drift.py --results_dir DIR --plot_ef 200

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


summary  = load("summary.csv")
perquery = load("per_query.csv")
traces   = load("lb_traces.csv")

# load params for labels
params = {}
params_path = os.path.join(args.results_dir, "params.csv")
if os.path.exists(params_path):
    for _, row in pd.read_csv(params_path).iterrows():
        params[row["param"]] = row["value"]

cluster_std = float(params.get("cluster_std", 1.0))
k = int(params.get("k_neighbours", 10))
mode = params.get("mode", "")

query_years = sorted(summary["query_year"].unique())
efs    = sorted(summary["ef_search"].unique())

plot_ef = args.plot_ef or efs[len(efs) // 2]
if plot_ef not in efs:
    plot_ef = min(efs, key=lambda e: abs(e - plot_ef))
    print(f"warning: requested ef not in sweep, using {plot_ef}")

print(f"query years: {query_years}")
print(f"efs:    {efs}")
print(f"plot_ef: {plot_ef}\n")


def get(query_year, ef, col):
    row = summary[(summary["query_year"] == query_year) & (summary["ef_search"] == ef)]
    if row.empty:
        return float("nan")
    return float(row[col].iloc[0])


# colours: blue -> red for increasing shift, viridis for increasing ef
def shift_cmap(n):
    return [matplotlib.colormaps["coolwarm"](i / max(n - 1, 1)) for i in range(n)]

def ef_cmap(n):
    return [matplotlib.colormaps["viridis"](i / max(n - 1, 1)) for i in range(n)]


# --- plot 1: recall surface --------------------------------------------------
# two panels: recall vs shift (one line per ef) and recall vs ef (one per sigma)
# this is the primary result figure

def plot_recall_surface():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ef, color in zip(efs, ef_cmap(len(efs))):
        recalls = [get(s, ef, "mean_recall") for s in query_years]
        ax1.plot(query_years, recalls, color=color, marker="o", markersize=5,
                 lw=1.8, label=f"ef={ef}")

    ax1.set_xlabel("query year")
    ax1.set_ylabel(f"Recall@{k}")
    ax1.set_title("recall vs query year")
    ax1.legend(fontsize=8, title="ef_search")
    ax1.set_xticks(query_years)
    ax1.grid(alpha=0.3)

    for query_year, color in zip(query_years, shift_cmap(len(query_years))):
        recalls = [get(query_year, ef, "mean_recall") for ef in efs]
        ax2.plot(efs, recalls, color=color, marker="o", markersize=5,
                 lw=1.8, label=f"query year={query_year}")

    ax2.set_xlabel("ef_search")
    ax2.set_ylabel(f"Recall@{k}")
    ax2.set_title("recall vs ef_search per query year")
    ax2.set_xscale("log")
    ax2.set_xticks(efs)
    ax2.set_xticklabels([str(e) for e in efs], fontsize=9)
    ax2.legend(fontsize=8, title="query year", loc="upper left")
    ax2.grid(alpha=0.3)

    fig.suptitle(
        f"continuous distribution shift — Recall@{k}  [{mode} mode]\n",
        y=1.01
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "recall_vs_shift.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# --- plot 2: all metrics vs shift at one ef ----------------------------------

def plot_metrics(ef):
    metrics = [
        ("mean_recall",             f"Recall@{k}",              "recall"),
        ("mean_ep_distance",        "entry-point distance",      "L2"),
        ("mean_bl_entry_distance",  "base-layer entry distance", "L2"),
        ("mean_layer1_visits",      "layer-1 visit count",       "# exams"),
        ("mean_base_visited",       "base-layer visited nodes",  "# nodes"),
        ("mean_candidates_remaining","candidates at termination","# candidates"),
    ]

    colors = shift_cmap(len(query_years))
    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 18), sharex=True)

    for ax, (col, title, ylabel) in zip(axes, metrics):
        if col not in summary.columns:
            ax.set_title(f"{title} (not available)")
            continue
        vals = [get(s, ef, col) for s in query_years]
        ax.plot(query_years, vals, color="#4C72B0", marker="o", lw=2)
        # recolour each point by query year so the gradient is visible
        for s, v, c in zip(query_years, vals, colors):
            ax.scatter([s], [v], color=c, s=60, zorder=5)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("shift magnitude")
    axes[-1].set_xticks(query_years)
    fig.suptitle(f"metrics vs query year  [ef={ef}, {mode}]", y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, f"metrics_vs_query_year_ef{ef}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# --- plot 3: per-query recall violin at one ef -------------------------------

def plot_violin(ef):
    df = perquery[perquery["ef_search"] == ef]
    if df.empty:
        print(f"no per-query data for ef={ef}, skipping violin")
        return

    colors = shift_cmap(len(query_years))
    fig, ax = plt.subplots(figsize=(max(8, len(query_years) * 0.9), 5))

    data   = [df[df["query_year"] == y]["recall"].tolist() for y in query_years]
    labels = [f"query year={y}" for y in query_years]

    parts = ax.violinplot(data, positions=range(len(query_years)),
                          showmedians=True, showextrema=True)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(query_years)))
    ax.set_xticklabels(labels, fontsize=9)
    for tick, c in zip(ax.get_xticklabels(), colors):
        tick.set_color(c)

    ax.set_ylabel(f"Recall@{k}")
    ax.set_xlabel("query year")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.set_title(f"per-query recall distributions  [ef={ef}, {mode}]")
    fig.tight_layout()
    path = os.path.join(out_dir, f"recall_violin_ef{ef}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


# --- plot 4: lowerbound traces at one ef -------------------------------------
def plot_lb_traces(ef):
    df = traces[traces["ef_search"] == ef]
    if df.empty:
        return

    all_query_years = sorted(df["query_year"].unique())
    colors = shift_cmap(len(all_query_years))

    fig, axes = plt.subplots(1, len(all_query_years),
                             figsize=(3.5 * len(all_query_years), 4))
    if len(all_query_years) == 1:
        axes = [axes]

    # pre-compute all subsets to derive shared axis limits first
    all_subs = [df[df["query_year"] == s].sort_values("iteration") for s in all_query_years]
    x_min = min(sub["iteration"].min() for sub in all_subs)
    x_max = max(sub["iteration"].max() for sub in all_subs)
    y_min = min(sub["mean_lowerbound"].min() for sub in all_subs)
    y_max = max(sub["mean_lowerbound"].max() for sub in all_subs)
    y_pad = (y_max - y_min) * 0.05 or 0.1  # 5 % padding; fallback if flat

    for ax, query_year, color, sub in zip(axes, all_query_years, colors, all_subs):
        ax.plot(sub["iteration"], sub["mean_lowerbound"], color=color, lw=2)
        ax.set_title(f"query year={query_year}", color=color)
        ax.set_xlabel("iteration", fontsize=9)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        if ax is axes[0]:
            ax.set_ylabel("mean lowerBound (L2)")
        ax.grid(alpha=0.3)

    fig.suptitle(f"lowerBound convergence  [ef={ef}, {mode}]")
    fig.tight_layout()
    path = os.path.join(out_dir, f"lb_traces_ef{ef}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")

# --- run everything ----------------------------------------------------------

plot_recall_surface()
plot_metrics(plot_ef)
plot_violin(plot_ef)
plot_lb_traces(plot_ef)

print(f"\nall plots written to {os.path.abspath(out_dir)}")