# Compare per-layer visit counts across experiments and shift levels.
#
# Shows how much navigational work happens at each HNSW layer, and how that changes under distribution shift. 
# Trying to analyse whether layer 1 is a suitable target for graph adaptation
#
# python plots/plot_layer_visits.py --experiments synthetic:results/results_continuous_drift_large sift:results/results_sift_drift_pc1split_10M yfcc:results/results_yfcc_directional_drift_5M --ef 200 --out_dir results/comparison_layer_visits

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--experiments", nargs="+", required=True,
                    help="label:path pairs, e.g. synthetic:results_continuous_drift_large")
parser.add_argument("--ef", type=int, default=200,
                    help="which ef_search value to analyse (default: 200)")
parser.add_argument("--out_dir", default="figures_layer_visits")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# layer columns present in per_query.csv
LAYER_COLS = [f"layer{i}_visits" for i in range(7)]

# shift column name differs between experiments
SHIFT_COL = "shift_sigma"


# --- loading -----------------------------------------------------------------

def load_experiment(label, path):
    csv_path = os.path.join(path, "per_query.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"missing {csv_path}")
    df = pd.read_csv(csv_path)
    df["experiment"] = label
    print(f"  {label}: {len(df):,} rows, "
          f"shifts={sorted(df[SHIFT_COL].unique())}, "
          f"efs={sorted(df['ef_search'].unique())}")
    return df


print("loading experiments...")
experiments = {}
for spec in args.experiments:
    label, path = spec.split(":", 1)
    experiments[label] = load_experiment(label, path)


# --- helpers -----------------------------------------------------------------

def get_layer_means(df, ef):
    # For a single experiment dataframe, return a table ofcmean visits per layer per shift level at the given ef.
    # Rows = shift levels, columns = layer0..layerN.
    sub = df[df["ef_search"] == ef].copy()
    if sub.empty:
        print(f"  warning: no data for ef={ef}")
        return None

    # only keep layer columns that have any nonzero values
    present = [c for c in LAYER_COLS if c in sub.columns and sub[c].sum() > 0]
    result  = sub.groupby(SHIFT_COL)[present].mean()
    return result


def shift_cmap(n):
    return [matplotlib.colormaps["coolwarm"](i / max(n - 1, 1)) for i in range(n)]


# --- plot 1: stacked bar, mean visits per layer at sigma=0 ------------------
# shows the absolute distribution of work across layers in-distribution

def plot_layer_breakdown():
    fig, axes = plt.subplots(1, len(experiments),
                             figsize=(5 * len(experiments), 5),
                             sharey=False)
    if len(experiments) == 1:
        axes = [axes]

    for ax, (label, df) in zip(axes, experiments.items()):
        means = get_layer_means(df, args.ef)
        if means is None:
            continue

        # use sigma=0 (or the smallest available shift) as the baseline
        baseline_sigma = means.index.min()
        row    = means.loc[baseline_sigma]
        layers = [c.replace("_visits", "") for c in row.index]
        values = row.values

        colors = matplotlib.colormaps["Blues"](
            np.linspace(0.3, 0.9, len(layers))
        )
        bars = ax.bar(layers, values, color=colors, edgecolor="white", linewidth=0.5)

        # annotate each bar with its percentage of total visits
        total = values.sum()
        for bar, v in zip(bars, values):
            pct = 100 * v / total if total > 0 else 0
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + total * 0.005,
                    f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)

        ax.set_title(label)
        ax.set_xlabel("layer")
        ax.set_ylabel("mean visits per query")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"mean visits per HNSW layer at σ=0  [ef={args.ef}]\n",
        y=1.02
    )
    fig.tight_layout()
    path = os.path.join(args.out_dir, "layer_breakdown_baseline.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# --- plot 2: layer visits vs shift, one line per layer -----------------------
# shows which layers change under shift and which stay flat

def plot_layer_visits_vs_shift():
    n = len(experiments)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (label, df) in zip(axes, experiments.items()):
        means = get_layer_means(df, args.ef)
        if means is None:
            continue

        present_layers = means.columns.tolist()
        layer_colors = matplotlib.colormaps["tab10"](
            np.linspace(0, 0.6, len(present_layers))
        )

        for col, color in zip(present_layers, layer_colors):
            layer_name = col.replace("_visits", "")
            ax.plot(means.index, means[col],
                    marker="o", markersize=4, lw=1.8,
                    color=color, label=layer_name)

        ax.set_title(label)
        ax.set_xlabel("shift magnitude (σ)")
        ax.set_ylabel("mean visits per query")
        ax.legend(fontsize=8, title="layer")
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"mean layer visits vs shift  [ef={args.ef}]\n",
        y=1.02
    )
    fig.tight_layout()
    path = os.path.join(args.out_dir, "layer_visits_vs_shift.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")




# --- plot 3: fraction of total upper-layer work done at each layer ----------
# normalised version, shows relative importance of each layer regardless of ef

def plot_layer_fractions():
    fig, axes = plt.subplots(1, len(experiments),
                             figsize=(5 * len(experiments), 4))
    if len(experiments) == 1:
        axes = [axes]

    for ax, (label, df) in zip(axes, experiments.items()):
        means = get_layer_means(df, args.ef)
        if means is None:
            continue

        # only upper layers (exclude layer 0 which is the base layer)
        upper_cols = [c for c in means.columns if c != "layer0_visits"]
        if not upper_cols:
            continue

        upper = means[upper_cols]
        fractions = upper.div(upper.sum(axis=1), axis=0)

        sigmas = fractions.index.tolist()
        colors = matplotlib.colormaps["tab10"](
            np.linspace(0, 0.6, len(upper_cols))
        )
        bottom = np.zeros(len(sigmas))

        for col, color in zip(upper_cols, colors):
            vals = fractions[col].values
            ax.bar(range(len(sigmas)), vals, bottom=bottom,
                   color=color, label=col.replace("_visits", ""),
                   edgecolor="white", linewidth=0.3)
            bottom += vals

        ax.set_xticks(range(len(sigmas)))
        ax.set_xticklabels([str(s) for s in sigmas], fontsize=7, rotation=45)
        ax.set_xlabel("shift magnitude (σ)")
        ax.set_ylabel("fraction of upper-layer visits")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, title="layer", loc="upper right")
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"fraction of upper-layer work per layer vs shift  [ef={args.ef}]\n",
        y=1.02
    )
    fig.tight_layout()
    path = os.path.join(args.out_dir, "layer_fractions_vs_shift.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")




# --- run everything ----------------------------------------------------------

plot_layer_breakdown()
plot_layer_visits_vs_shift()
plot_layer_fractions()

print(f"\nall plots written to {os.path.abspath(args.out_dir)}")