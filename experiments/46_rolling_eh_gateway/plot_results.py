"""Exp 46 plots: recall and edge-count curves for static vs conjugate EH vs gateway rolling EH."""

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
    "gateway_rolling_eh": "darkorange",
}
LABELS = {
    "static": "Static",
    "adaptive_eh": "Adaptive EH (conjugate)",
    "gateway_rolling_eh": "Gateway rolling EH",
}
COND_ORDER = ["static", "adaptive_eh", "gateway_rolling_eh"]

PLATEAU_EPOCHS = list(range(20, 25))


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _gate_open_epochs(df):
    """Return set of epochs where gate_open=True for gateway_rolling_eh."""
    sub = df[(df["condition"] == "gateway_rolling_eh") & df["gate_open"]]
    return set(sub["epoch"].unique())


def plot_schedule(df, schedule, cfg, plots_dir):
    ef_values = sorted(df["ef"].unique())
    transitions = cfg.get(f"drift_transition_epochs_{schedule}", [])
    present = set(df["condition"].unique())
    gate_epochs = _gate_open_epochs(df)

    # Figure 1: recall@10 vs epoch, one subplot per ef
    # Shade epochs where gate_open=True with a light orange background band.
    fig, axes = plt.subplots(1, len(ef_values), figsize=(5 * len(ef_values), 4), sharey=True)
    if len(ef_values) == 1:
        axes = [axes]

    for ax, ef in zip(axes, ef_values):
        # Shade gate-open epochs
        if gate_epochs:
            epochs_sorted = sorted(gate_epochs)
            # Group consecutive epochs into bands
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

    fig.suptitle(f"Recall@10 — {schedule} drift (DEEP-96 cluster, exp46)", fontsize=11)
    fig.tight_layout()
    out = plots_dir / f"{schedule}_recall.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")

    # Figure 2: cumulative edge count vs epoch
    # Primary axis: edge counts for adaptive_eh and gateway_rolling_eh.
    # Gate-open epochs shaded in light orange.
    fig, ax = plt.subplots(figsize=(6, 4))

    for ep_start, ep_end in _consecutive_bands(gate_epochs):
        ax.axvspan(ep_start - 0.5, ep_end + 0.5, color="darkorange", alpha=0.10, zorder=0)

    for cond in ["adaptive_eh", "gateway_rolling_eh"]:
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
