"""Experiment 20: plot recall and edge-accumulation results."""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

sns.set_theme(style="whitegrid")

RESULTS_DIR = ROOT / "experiments" / "20_msmarco_focused_ood"
PLOTS_DIR = Path(__file__).parent / "plots"
DATASET_GRADUAL = ROOT / "experiments" / "20_msmarco_focused_ood" / "dataset_gradual"
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _load_csv(schedule, condition):
    path = RESULTS_DIR / f"{schedule}_{condition}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Results not found: {path}")
    return pd.read_csv(path)


def _load_cluster_diag():
    path = DATASET_GRADUAL / "cluster_diagnostics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _get_schedule_t(cfg, schedule_tag):
    key = f"schedule_{schedule_tag}"
    return cfg["drift"][key]


def _shade_drift_epochs(ax, t_values, alpha=0.08):
    for i, t in enumerate(t_values):
        if t > 0:
            ax.axvspan(i - 0.5, i + 0.5, color="orange", alpha=alpha, linewidth=0)


def figure1_gradual(cfg):
    """Recall@10 vs epoch, gradual, ef=32, static vs adaptive."""
    t_vals = _get_schedule_t(cfg, "gradual")
    primary_ef = cfg["eval"]["primary_ef"]

    df_static = _load_csv("gradual", "static")
    df_adapt = _load_csv("gradual", "adaptive_mconj48")

    df_static = df_static[df_static["ef_search"] == primary_ef].copy()
    df_adapt = df_adapt[df_adapt["ef_search"] == primary_ef].copy()

    cluster_diag = _load_cluster_diag()

    fig, ax = plt.subplots(figsize=(10, 5))
    _shade_drift_epochs(ax, t_vals)

    ax.plot(df_static["epoch_idx"], df_static["recall_at_k"], label="static", color="steelblue")
    ax.plot(
        df_adapt["epoch_idx"],
        df_adapt["recall_at_k"],
        label="adaptive (M_conj=48)",
        color="darkorange",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recall@10")
    ax.set_title("MS MARCO focused OOD drift (gradual) — recall@10 vs epoch")
    ax.legend()

    if cluster_diag is not None:
        cd = cluster_diag["cluster_diagnostics"]
        ann = (
            f"cluster size={cd['chosen_cluster_size']}, "
            f"mean dist={cd['chosen_cluster_mean_dist_to_centroid']:.4f}"
        )
        ax.text(
            0.99, 0.02, ann,
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=8, color="gray",
        )

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = PLOTS_DIR / "figure1_gradual_recall.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def figure2_sudden(cfg):
    """Recall@10 vs epoch, sudden, ef=32, static vs adaptive."""
    t_vals = _get_schedule_t(cfg, "sudden")
    primary_ef = cfg["eval"]["primary_ef"]

    df_static = _load_csv("sudden", "static")
    df_adapt = _load_csv("sudden", "adaptive_mconj48")

    df_static = df_static[df_static["ef_search"] == primary_ef].copy()
    df_adapt = df_adapt[df_adapt["ef_search"] == primary_ef].copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    _shade_drift_epochs(ax, t_vals)

    ax.plot(df_static["epoch_idx"], df_static["recall_at_k"], label="static", color="steelblue")
    ax.plot(
        df_adapt["epoch_idx"],
        df_adapt["recall_at_k"],
        label="adaptive (M_conj=48)",
        color="darkorange",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recall@10")
    ax.set_title("MS MARCO focused OOD drift (sudden) — recall@10 vs epoch")
    ax.legend()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = PLOTS_DIR / "figure2_sudden_recall.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def figure3_edge_accumulation(cfg):
    """n_conjugate_edges (left) and mean_eh (right) vs epoch, gradual, adaptive only."""
    primary_ef = cfg["eval"]["primary_ef"]
    df = _load_csv("gradual", "adaptive_mconj48")
    df = df[df["ef_search"] == primary_ef].copy()

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_edges = "darkorange"
    color_eh = "steelblue"

    ax1.plot(df["epoch_idx"], df["n_conjugate_edges"], color=color_edges, label="n_conjugate_edges")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("n_conjugate_edges", color=color_edges)
    ax1.tick_params(axis="y", labelcolor=color_edges)

    ax2 = ax1.twinx()
    ax2.plot(df["epoch_idx"], df["mean_eh"], color=color_eh, linestyle="--", label="mean_eh")
    ax2.set_ylabel("mean_eh", color=color_eh)
    ax2.tick_params(axis="y", labelcolor=color_eh)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.suptitle("Edge accumulation and EH under focused OOD drift (gradual)")
    fig.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = PLOTS_DIR / "figure3_edge_accumulation.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    cfg = _load_config()
    figure1_gradual(cfg)
    figure2_sudden(cfg)
    figure3_edge_accumulation(cfg)


if __name__ == "__main__":
    main()
