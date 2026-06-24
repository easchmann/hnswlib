"""Exp 29 plots: recall, edge count, and layer-1-native acceptance rate."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

COLORS = {
    "static": "grey",
    "adaptive_eh": "steelblue",
    "adaptive_layer1_native": "mediumpurple",
    "hybrid_conj_l1n": "forestgreen",
}
LABELS = {
    "static": "Static",
    "adaptive_eh": "Adaptive EH (conjugate)",
    "adaptive_layer1_native": "Adaptive L1-native",
    "hybrid_conj_l1n": "Hybrid conj + L1-native",
}
LINESTYLES = {
    "static": "-",
    "adaptive_eh": "-",
    "adaptive_layer1_native": "-",
    "hybrid_conj_l1n": "--",
}
COND_ORDER = ["static", "adaptive_eh", "adaptive_layer1_native", "hybrid_conj_l1n"]


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def plot_schedule(df, schedule, cfg, plots_dir):
    ef_values = sorted(df["ef"].unique())
    transitions = cfg.get(f"drift_transition_epochs_{schedule}", [])
    present = set(df["condition"].unique())

    # Figure 1: recall@10 vs epoch, one subplot per ef
    fig, axes = plt.subplots(1, len(ef_values), figsize=(5 * len(ef_values), 4), sharey=True)
    if len(ef_values) == 1:
        axes = [axes]

    for ax, ef in zip(axes, ef_values):
        sub = df[df["ef"] == ef]
        for cond in COND_ORDER:
            if cond not in present:
                continue
            d = sub[sub["condition"] == cond].sort_values("epoch")
            ax.plot(d["epoch"], d["recall"], color=COLORS[cond], label=LABELS[cond],
                    linewidth=2.0 if cond == "hybrid_conj_l1n" else 1.5,
                    linestyle=LINESTYLES[cond])
        for t in transitions:
            ax.axvline(t, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.set_title(f"ef={ef}")
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Recall@10" if ef == ef_values[0] else "")
        ax.legend(fontsize=7)

    fig.suptitle(f"Recall@10 — {schedule} drift", fontsize=11)
    fig.tight_layout()
    out = plots_dir / f"{schedule}_recall.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")

    # Figure 2: cumulative edge count vs epoch (skip static)
    fig, ax = plt.subplots(figsize=(6, 4))
    edge_cols = {
        "adaptive_eh": "edge_count",
        "adaptive_layer1_native": "edge_count",
        "hybrid_conj_l1n": "l1n_edge_count",
    }
    for cond in ["adaptive_eh", "adaptive_layer1_native", "hybrid_conj_l1n"]:
        if cond not in present:
            continue
        col = edge_cols[cond]
        if col not in df.columns:
            continue
        sub = df[(df["condition"] == cond) & (df["ef"] == ef_values[0])].sort_values("epoch")
        if sub[col].isna().all():
            continue
        ax.plot(sub["epoch"], sub[col], color=COLORS[cond], label=LABELS[cond],
                linestyle=LINESTYLES[cond])
    for t in transitions:
        ax.axvline(t, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cumulative edges added")
    ax.set_title(f"Edge count — {schedule} drift")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = plots_dir / f"{schedule}_edges.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")

    # Figure 3: layer-1-native acceptance rate (edges_added / attempts per epoch)
    fig, ax = plt.subplots(figsize=(6, 3))
    plotted = False
    for cond, edge_col, attempt_col in [
        ("adaptive_layer1_native", "edge_count", "lN_attempts"),
        ("hybrid_conj_l1n", "l1n_edge_count", "lN_attempts"),
    ]:
        if cond not in present:
            continue
        if edge_col not in df.columns or attempt_col not in df.columns:
            continue
        sub = df[(df["condition"] == cond) & (df["ef"] == ef_values[0])].sort_values("epoch").reset_index(drop=True)
        if sub[edge_col].isna().all() or sub[attempt_col].isna().all():
            continue
        per_epoch_edges = sub[edge_col].diff().fillna(sub[edge_col].iloc[0])
        per_epoch_attempts = sub[attempt_col].diff().fillna(sub[attempt_col].iloc[0])
        rate = per_epoch_edges / per_epoch_attempts.replace(0, float("nan"))
        ax.plot(sub["epoch"], rate, color=COLORS[cond], label=LABELS[cond],
                linestyle=LINESTYLES[cond])
        plotted = True
    if plotted:
        for t in transitions:
            ax.axvline(t, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Acceptance rate (edges / attempts)")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Layer-1-native injection acceptance rate — {schedule} drift")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = plots_dir / f"{schedule}_l1n_acceptance.pdf"
        fig.savefig(out)
        print(f"  saved {out}")
    plt.close(fig)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "config.yaml"
    cfg = _load_config(config_path)
    plots_dir = ROOT / cfg["plots_dir"]
    plots_dir.mkdir(parents=True, exist_ok=True)

    for schedule, results_key in [("gradual", "results_gradual"), ("sudden", "results_sudden")]:
        path = ROOT / cfg[results_key]
        if not path.exists():
            print(f"Missing {path}, skipping {schedule}")
            continue
        df = pd.read_csv(path)
        plot_schedule(df, schedule, cfg, plots_dir)


if __name__ == "__main__":
    main()
