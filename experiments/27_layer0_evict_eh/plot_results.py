"""Exp 27 plots: recall and edge-count curves for all conditions."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

COLORS = {
    "static": "grey",
    "adaptive_eh": "steelblue",
    "adaptive_layer0_eh": "darkorange",
    "hybrid_eh": "forestgreen",
}
LABELS = {
    "static": "Static",
    "adaptive_eh": "Adaptive EH (conjugate)",
    "adaptive_layer0_eh": "Adaptive Layer0 EH (evict)",
    "hybrid_eh": "Hybrid EH (conjugate + layer-0)",
}
# Draw order: hybrid on top so it's visible over conjugate
COND_ORDER = ["static", "adaptive_layer0_eh", "adaptive_eh", "hybrid_eh"]


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def plot_schedule(df, schedule, cfg, plots_dir):
    ef_values = sorted(df["ef"].unique())
    transitions = cfg.get(f"drift_transition_epochs_{schedule}", [])
    present = set(df["condition"].unique())

    # Figure 1: recall per ef
    fig, axes = plt.subplots(1, len(ef_values), figsize=(5 * len(ef_values), 4), sharey=True)
    if len(ef_values) == 1:
        axes = [axes]

    for ax, ef in zip(axes, ef_values):
        sub = df[df["ef"] == ef]
        for cond in COND_ORDER:
            if cond not in present:
                continue
            d = sub[sub["condition"] == cond].sort_values("epoch")
            lw = 2.0 if cond == "hybrid_eh" else 1.5
            ls = "--" if cond == "hybrid_eh" else "-"
            ax.plot(d["epoch"], d["recall"], color=COLORS[cond], label=LABELS[cond],
                    linewidth=lw, linestyle=ls)
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

    # Figure 2: conjugate edge count (comparable across adaptive_eh and hybrid_eh)
    fig, ax = plt.subplots(figsize=(6, 4))
    for cond in ["adaptive_eh", "adaptive_layer0_eh", "hybrid_eh"]:
        if cond not in present:
            continue
        sub = df[(df["condition"] == cond) & (df["ef"] == ef_values[0])].sort_values("epoch")
        ax.plot(sub["epoch"], sub["edge_count"], color=COLORS[cond], label=LABELS[cond])
    for t in transitions:
        ax.axvline(t, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cumulative edges added (conjugate)")
    ax.set_title(f"Edge count — {schedule} drift")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = plots_dir / f"{schedule}_edges.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")

    # Figure 3: layer-0 injection count for hybrid (if column present)
    if "l0_edge_count" in df.columns and "hybrid_eh" in present:
        fig, ax = plt.subplots(figsize=(6, 3))
        sub = df[(df["condition"] == "hybrid_eh") & (df["ef"] == ef_values[0])].sort_values("epoch")
        ax.plot(sub["epoch"], sub["l0_edge_count"], color=COLORS["hybrid_eh"],
                label="Layer-0 injections (hybrid)")
        # Compare to layer0-only
        sub2 = df[(df["condition"] == "adaptive_layer0_eh") & (df["ef"] == ef_values[0])].sort_values("epoch")
        if not sub2.empty:
            ax.plot(sub2["epoch"], sub2["edge_count"], color=COLORS["adaptive_layer0_eh"],
                    linestyle="--", label="Layer-0 injections (standalone)")
        for t in transitions:
            ax.axvline(t, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cumulative layer-0 injections")
        ax.set_title(f"Layer-0 injection count — {schedule} drift")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = plots_dir / f"{schedule}_l0_count.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"  saved {out}")


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
