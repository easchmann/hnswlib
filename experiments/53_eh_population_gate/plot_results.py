"""Exp 53 plots: recall and edge-count curves for static vs improved (ungated) vs
improved (EH population gate) on the clustered structural hardness drift dataset."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

COLORS = {
    "static":                  "grey",
    "adaptive_improved":       "steelblue",
    "adaptive_improved_gated": "darkorange",
}
LABELS = {
    "static":                  "Static",
    "adaptive_improved":       "Improved (ungated)",
    "adaptive_improved_gated": "Improved (EH gate)",
}
COND_ORDER = ["static", "adaptive_improved", "adaptive_improved_gated"]


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _plateau_summary(df, cfg, schedule):
    primary_ef = cfg.get("primary_ef", 32)
    plateau = df[(df["epoch"] >= 20) & (df["epoch"] <= 24) & (df["ef"] == primary_ef)]
    if plateau.empty:
        return

    static_sub = plateau[plateau["condition"] == "static"]
    if static_sub.empty:
        return
    static_rec = static_sub["recall"].mean()

    print(f"\nPlateau summary — {schedule} (epochs 20–24, ef={primary_ef}):")
    print(f"  {'condition':>26}  {'mean_recall':>12}  {'mean_edges':>12}  {'recovery':>10}")
    for cond in COND_ORDER:
        sub = plateau[plateau["condition"] == cond]
        if sub.empty:
            continue
        mean_rec = sub["recall"].mean()
        mean_edges = sub["edge_count"].mean()
        denom = 1.0 - static_rec
        recovery = (mean_rec - static_rec) / denom if denom > 0 else float("nan")
        print(f"  {cond:>26}  {mean_rec:>12.4f}  {mean_edges:>12.0f}  {recovery:>10.3f}")


def plot_schedule(df, schedule, cfg, plots_dir):
    ef_values = sorted(df["ef"].unique())
    transitions = cfg.get(f"drift_transition_epochs_{schedule}", [])
    present = set(df["condition"].unique())

    fig, axes = plt.subplots(1, len(ef_values), figsize=(5 * len(ef_values), 4), sharey=True)
    if len(ef_values) == 1:
        axes = [axes]

    for ax, ef in zip(axes, ef_values):
        sub = df[df["ef"] == ef]
        for cond in COND_ORDER:
            if cond not in present:
                continue
            d = sub[sub["condition"] == cond].sort_values("epoch")
            ax.plot(d["epoch"], d["recall"], color=COLORS[cond], label=LABELS[cond], linewidth=1.5)
        for t in transitions:
            ax.axvline(t, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.set_title(f"ef={ef}")
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Recall@10" if ef == ef_values[0] else "")
        ax.legend(fontsize=7)

    fig.suptitle(f"Clustered hardness drift — population EH gate ({schedule})", fontsize=11)
    fig.tight_layout()
    out = plots_dir / f"{schedule}_recall.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")

    fig, ax = plt.subplots(figsize=(6, 4))
    for cond in ["adaptive_improved", "adaptive_improved_gated"]:
        if cond not in present:
            continue
        sub = df[(df["condition"] == cond) & (df["ef"] == ef_values[0])].sort_values("epoch")
        ax.plot(sub["epoch"], sub["edge_count"], color=COLORS[cond], label=LABELS[cond])
    for t in transitions:
        ax.axvline(t, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cumulative edges added")
    ax.set_title(f"Edge count — {schedule} drift (population EH gate)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = plots_dir / f"{schedule}_edges.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")

    _plateau_summary(df, cfg, schedule)


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
