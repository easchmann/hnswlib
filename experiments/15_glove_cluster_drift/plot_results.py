"""Experiment 15 figures: GloVe-100 cluster-reweighted drift."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = _SCRIPT_DIR / "config.yaml"
DPI = 300
PRIMARY_EF = 32

CONDITION_COLORS = {
    "static": "grey",
    "periodic_rebuild": "darkorange",
    "adaptive_mconj48": "steelblue",
}
CONDITION_LABELS = {
    "static": "Static HNSW",
    "periodic_rebuild": "Periodic rebuild (every 5 ep.)",
    "adaptive_mconj48": "Adaptive ($M_{\\mathrm{conj}}=48$)",
}


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _results_dir(cfg):
    path = ROOT / cfg["output"]["results_dir"]
    if path.exists():
        return path
    return _SCRIPT_DIR


def _load_all(cfg, results_dir):
    data = {}
    for schedule in ("gradual", "sudden"):
        for cond in cfg["conditions"]:
            name = cond["name"]
            p = results_dir / f"{schedule}_{name}.csv"
            if not p.exists():
                print(f"  WARNING: Missing {p}")
                continue
            data[(schedule, name)] = pd.read_csv(p)
    return data


def _shade_drift(ax, schedule):
    in_drift = False
    start = None
    for i, t in enumerate(schedule):
        if t > 0.0 and not in_drift:
            in_drift = True
            start = i
        elif t == 0.0 and in_drift:
            ax.axvspan(start - 0.5, i - 0.5, color="#fdae6b", alpha=0.18, zorder=0, linewidth=0)
            in_drift = False
    if in_drift:
        ax.axvspan(start - 0.5, len(schedule) - 0.5, color="#fdae6b", alpha=0.18, zorder=0, linewidth=0)


def plot_figure1(data, cfg, plots_dir):
    """recall@10 vs epoch for gradual schedule, ef=32, all conditions."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(10, 5))

    schedule = cfg["drift"]["schedule_gradual"]
    _shade_drift(ax, schedule)

    for cond in cfg["conditions"]:
        name = cond["name"]
        key = ("gradual", name)
        if key not in data:
            continue
        df = data[key]
        sub = df[df["ef_search"] == PRIMARY_EF].sort_values("epoch_idx")
        ax.plot(
            sub["epoch_idx"], sub["recall_at_k"],
            color=CONDITION_COLORS.get(name, "black"),
            linewidth=2.0, marker="o", markersize=3.5,
            label=CONDITION_LABELS.get(name, name), zorder=3,
        )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Recall@10", fontsize=11)
    ax.set_title("GloVe-100 cluster-reweighted drift (gradual) — recall@10 vs epoch",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(-0.5, 24.5)
    ax.set_xticks(range(0, 25, 2))
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.85)

    fig.tight_layout()
    out = plots_dir / "recall_gradual.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_figure2(data, cfg, plots_dir):
    """recall@10 vs epoch for sudden schedule, ef=32, all conditions."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(10, 5))

    schedule = cfg["drift"]["schedule_sudden"]
    _shade_drift(ax, schedule)

    for cond in cfg["conditions"]:
        name = cond["name"]
        key = ("sudden", name)
        if key not in data:
            continue
        df = data[key]
        sub = df[df["ef_search"] == PRIMARY_EF].sort_values("epoch_idx")
        ax.plot(
            sub["epoch_idx"], sub["recall_at_k"],
            color=CONDITION_COLORS.get(name, "black"),
            linewidth=2.0, marker="o", markersize=3.5,
            label=CONDITION_LABELS.get(name, name), zorder=3,
        )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Recall@10", fontsize=11)
    ax.set_title("GloVe-100 cluster-reweighted drift (sudden) — recall@10 vs epoch",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(-0.5, 24.5)
    ax.set_xticks(range(0, 25, 2))
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.85)

    fig.tight_layout()
    out = plots_dir / "recall_sudden.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_figure3(cfg, plots_dir):
    """Drift diagnostics (OOD, MMD², mean NN) vs epoch, gradual schedule."""
    dataset_path = ROOT / cfg["output"]["dataset_gradual_path"]
    diag_path = dataset_path / "diagnostics.json"
    if not diag_path.exists():
        print(f"  WARNING: diagnostics not found at {diag_path}, skipping Figure 3")
        return

    with open(diag_path) as f:
        diagnostics = json.load(f)

    epochs = [d["epoch_idx"] for d in diagnostics]
    ood = [d["ood_distance"] for d in diagnostics]
    mmd2 = [d["mmd_squared"] for d in diagnostics]
    mean_nn = [d["mean_nn_distance"] for d in diagnostics]

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    schedule = cfg["drift"]["schedule_gradual"]

    for ax, vals, title in zip(
        axes,
        [ood, mmd2, mean_nn],
        ["OOD distance", "MMD²", "Mean NN distance"],
    ):
        _shade_drift(ax, schedule)
        ax.plot(epochs, vals, color="steelblue", linewidth=2.0, marker="o", markersize=3.5)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlim(-0.5, 24.5)
        ax.set_xticks(range(0, 25, 4))

    fig.suptitle("GloVe-100 cluster drift diagnostics (gradual)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = plots_dir / "drift_diagnostics.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    cfg = _load_config()
    results_dir = _results_dir(cfg)
    plots_dir = _SCRIPT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    data = _load_all(cfg, results_dir)
    if not data:
        print("No results found. Run run_eval.py first.")
        return

    plot_figure1(data, cfg, plots_dir)
    plot_figure2(data, cfg, plots_dir)
    plot_figure3(cfg, plots_dir)


if __name__ == "__main__":
    main()
