"""Exp 49 plots: recall, edges, ef-scaling, and ratio-gap diagnostic for gateway_ratio_gate."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

COLORS = {
    "static": "grey",
    "adaptive_eh": "steelblue",
    "gateway_ratio_gate": "darkorange",
}
LABELS = {
    "static": "Static",
    "adaptive_eh": "Adaptive EH (conjugate)",
    "gateway_ratio_gate": "Gateway ratio gate",
}
COND_ORDER = ["static", "adaptive_eh", "gateway_ratio_gate"]

PLATEAU_EPOCHS = list(range(20, 25))


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _consecutive_bands(epoch_set):
    if not epoch_set:
        return []
    epochs_sorted = sorted(epoch_set)
    bands = []
    start = epochs_sorted[0]
    prev = epochs_sorted[0]
    for ep in epochs_sorted[1:]:
        if ep == prev + 1:
            prev = ep
        else:
            bands.append((start, prev))
            start = ep
            prev = ep
    bands.append((start, prev))
    return bands


def _gate_open_epochs(df):
    """Return set of epochs where gate_open=True for gateway_ratio_gate."""
    sub = df[(df["condition"] == "gateway_ratio_gate") & df["gate_open"]]
    return set(sub["epoch"].unique())


def plot_schedule(df, schedule, cfg, plots_dir):
    ef_values = sorted(df["ef"].unique())
    transitions = cfg.get(f"drift_transition_epochs_{schedule}", [])
    present = set(df["condition"].unique())
    gate_epochs = _gate_open_epochs(df)
    bands = _consecutive_bands(gate_epochs)

    # Figure 1: recall@10 vs epoch, one subplot per ef
    fig, axes = plt.subplots(1, len(ef_values), figsize=(5 * len(ef_values), 4), sharey=True)
    if len(ef_values) == 1:
        axes = [axes]

    for ax, ef in zip(axes, ef_values):
        for band_start, band_end in bands:
            ax.axvspan(band_start - 0.5, band_end + 0.5,
                       color="darkorange", alpha=0.10, zorder=0)
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

    fig.suptitle(f"Recall@10 — {schedule} drift (DEEP-96 cluster, exp49)", fontsize=11)
    fig.tight_layout()
    out = plots_dir / f"{schedule}_recall.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")

    # Figure 2: cumulative edge count vs epoch
    fig, ax = plt.subplots(figsize=(6, 4))
    for band_start, band_end in bands:
        ax.axvspan(band_start - 0.5, band_end + 0.5,
                   color="darkorange", alpha=0.10, zorder=0)
    for cond in ["adaptive_eh", "gateway_ratio_gate"]:
        if cond not in present:
            continue
        sub = df[(df["condition"] == cond) & (df["ef"] == ef_values[0])].sort_values("epoch")
        ax.plot(sub["epoch"], sub["edge_count"], color=COLORS[cond], label=LABELS[cond])
    for t in transitions:
        ax.axvline(t, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cumulative edges added")
    ax.set_title(f"Edge count — {schedule} drift (DEEP-96 cluster)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = plots_dir / f"{schedule}_edges.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")

    # Figure 3: ef-scaling — mean plateau recall vs ef (epochs 20–24)
    fig, ax = plt.subplots(figsize=(5, 4))
    for cond in COND_ORDER:
        if cond not in present:
            continue
        plateau = df[(df["condition"] == cond) & (df["epoch"].isin(PLATEAU_EPOCHS))]
        vals = [plateau[plateau["ef"] == ef]["recall"].mean() for ef in ef_values]
        ax.plot(ef_values, vals, color=COLORS[cond], label=LABELS[cond],
                marker="o", linewidth=1.5)
    ax.set_xlabel("ef_search")
    ax.set_ylabel("Mean recall@10 (epochs 20–24)")
    ax.set_title(f"Ef-scaling at full drift — {schedule}")
    ax.set_xticks(ef_values)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = plots_dir / f"{schedule}_efscaling.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")

    # Figure 4: ratio-signal diagnostic
    # Primary y-axis (left): gap_ratio vs epoch (darkorange solid)
    # Horizontal dashed line at ratio_threshold
    # Secondary y-axis (right): n_efhigh_queries per epoch (darkorange dotted)
    fig, ax1 = plt.subplots(figsize=(7, 4))

    for band_start, band_end in bands:
        ax1.axvspan(band_start - 0.5, band_end + 0.5,
                    color="darkorange", alpha=0.10, zorder=0)

    ratio_sub = df[(df["condition"] == "gateway_ratio_gate") &
                   (df["ef"] == ef_values[0])].sort_values("epoch")

    ax1.plot(ratio_sub["epoch"], ratio_sub["gap_ratio"],
             color="darkorange", linewidth=1.8, label="Rolling gap ratio (mean/calib)")
    ax1.axhline(cfg["gap_ratio_threshold"], color="darkorange", linestyle="--", alpha=0.5,
                label=f"Threshold ({cfg['gap_ratio_threshold']:.1f}×)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Rolling gap ratio (mean/calib)", color="darkorange")
    ax1.tick_params(axis="y", labelcolor="darkorange")

    for t in transitions:
        ax1.axvline(t, color="black", linestyle=":", linewidth=0.8, alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(ratio_sub["epoch"], ratio_sub["n_efhigh_queries"],
             color="darkorange", linewidth=0.8, linestyle=":", alpha=0.7,
             label="ef_high queries / epoch")
    ax2.set_ylabel("ef_high queries / epoch", color="darkorange", alpha=0.7)
    ax2.tick_params(axis="y", labelcolor="darkorange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    ax1.set_title(f"Gap ratio signal and ef_high trigger rate — {schedule} drift")
    fig.tight_layout()
    out = plots_dir / f"{schedule}_ratio_signal.pdf"
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
