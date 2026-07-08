"""Exp 52 / 50ep plots: recall and edge-count curves for 50-epoch convergence run.

Plateau summary uses epochs 40-49 (the converged plateau window for 50 epochs).
Also prints the ep20-24 window for direct comparison with the 25-epoch parent run.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]

COLORS = {
    "static":            "grey",
    "adaptive_current":  "steelblue",
    "adaptive_improved": "darkorange",
}
LABELS = {
    "static":            "Static",
    "adaptive_current":  "Adaptive EH (current)",
    "adaptive_improved": "Adaptive EH (improved)",
}
COND_ORDER = ["static", "adaptive_current", "adaptive_improved"]


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _plateau_summary(df, cfg, schedule, ep_start, ep_end, label):
    primary_ef = cfg.get("primary_ef", 32)
    plateau = df[(df["epoch"] >= ep_start) & (df["epoch"] <= ep_end) & (df["ef"] == primary_ef)]
    if plateau.empty:
        return

    static_sub = plateau[plateau["condition"] == "static"]
    if static_sub.empty:
        return
    static_rec = static_sub["recall"].mean()

    print(f"\nPlateau summary — {schedule} ({label}, ef={primary_ef}):")
    print(f"  {'condition':>22}  {'mean_recall':>12}  {'mean_edges':>12}  {'recovery':>10}")
    for cond in COND_ORDER:
        sub = plateau[plateau["condition"] == cond]
        if sub.empty:
            continue
        mean_rec = sub["recall"].mean()
        mean_edges = sub["edge_count"].mean()
        denom = 1.0 - static_rec
        recovery = (mean_rec - static_rec) / denom if denom > 0 else float("nan")
        print(f"  {cond:>22}  {mean_rec:>12.4f}  {mean_edges:>12.0f}  {recovery:>10.3f}")


def plot_schedule(df, schedule, cfg, plots_dir):
    ef_values = sorted(df["ef"].unique())
    transitions = cfg.get(f"drift_transition_epochs_{schedule}", [])
    plateau_start = cfg.get("plateau_start", 40)
    plateau_end = cfg.get("plateau_end", 49)
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
        # shade the 50-epoch plateau window
        ax.axvspan(plateau_start, plateau_end, alpha=0.06, color="orange", label="_plateau")
        ax.set_title(f"ef={ef}")
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Recall@10" if ef == ef_values[0] else "")
        ax.legend(fontsize=7)

    fig.suptitle(
        f"Clustered structural hardness drift — improved adapter ({schedule}, 50 ep)",
        fontsize=11,
    )
    fig.tight_layout()
    out = plots_dir / f"{schedule}_recall.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")

    fig, ax = plt.subplots(figsize=(6, 4))
    for cond in ["adaptive_current", "adaptive_improved"]:
        if cond not in present:
            continue
        sub = df[(df["condition"] == cond) & (df["ef"] == ef_values[0])].sort_values("epoch")
        ax.plot(sub["epoch"], sub["edge_count"], color=COLORS[cond], label=LABELS[cond])
    for t in transitions:
        ax.axvline(t, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cumulative edges added")
    ax.set_title(
        f"Edge count — {schedule} drift "
        f"(clustered structural hardness drift — 50 ep)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = plots_dir / f"{schedule}_edges.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")

    # Print both plateau windows for comparison with the 25-epoch parent run
    _plateau_summary(df, cfg, schedule, 20, 24, "epochs 20–24 (compare to 25ep run)")
    _plateau_summary(df, cfg, schedule, plateau_start, plateau_end,
                     f"epochs {plateau_start}–{plateau_end} (converged plateau)")


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
