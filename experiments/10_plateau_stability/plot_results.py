"""Experiment 10 figures: plateau stability analysis."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = _SCRIPT_DIR / "config.yaml"
DPI = 300
PRIMARY_EF = 32

SCHEDULES = ["gradual_plateau", "sudden_plateau"]
SCHEDULE_TITLES = {"gradual_plateau": "Gradual ramp → plateau", "sudden_plateau": "Sudden jump → plateau"}

# Phase colour palette
PHASE_COLORS = {"pre_drift": "#aaaaaa", "ramp": "#fd8d3c", "plateau": "#4292c6"}


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _out_dir(cfg):
    p = ROOT / cfg["output"]["results_dir"]
    if p.exists():
        return p
    return _SCRIPT_DIR


def _load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing results: {path}")
    return pd.read_csv(path)


def _load_all(cfg):
    out_dir = _out_dir(cfg)
    data = {}
    for sname in SCHEDULES:
        p = out_dir / f"{sname}.csv"
        try:
            data[sname] = _load_csv(p)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
    return data


def _get_schedule_values(cfg, schedule_name):
    key = "schedule_sudden_plateau" if "sudden" in schedule_name else "schedule_gradual_plateau"
    return cfg["drift"][key]


def _phase_of_epoch(t):
    if t == 0.0:
        return "pre_drift"
    elif t < 1.0:
        return "ramp"
    else:
        return "plateau"


def _shade_drift(ax, schedule):
    """Shade epochs where t > 0."""
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


def plot_figure1(data, cfg, figures_dir):
    """Recall@10 vs epoch with repair tick marks."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    for ax, sname in zip(axes, SCHEDULES):
        if sname not in data:
            ax.set_title(f"{SCHEDULE_TITLES[sname]}  (no data)")
            continue

        df = data[sname]
        schedule = _get_schedule_values(cfg, sname)
        sub = df[df["ef_search"] == PRIMARY_EF].sort_values("epoch_idx")

        _shade_drift(ax, schedule)

        ax.plot(
            sub["epoch_idx"], sub["recall_at_k"],
            color="steelblue", linewidth=2.0, marker="o", markersize=3.5, zorder=3,
        )

        # Repair tick marks on x-axis
        repair_epochs = sub[sub["repair_happened"] == True]["epoch_idx"].values
        for ep in repair_epochs:
            ax.axvline(ep, color="tomato", alpha=0.25, linewidth=1.0, zorder=2)

        # Annotate a proxy for the legend
        ax.axvline(-99, color="tomato", alpha=0.5, linewidth=1.5, label="repair event")

        n_epochs = len(schedule)
        ax.set_xlim(-0.5, n_epochs - 0.5)
        ax.set_xticks(range(0, n_epochs, 5))
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Recall@10", fontsize=11)
        ax.set_title(SCHEDULE_TITLES[sname], fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, loc="lower right", framealpha=0.85)

    fig.suptitle(
        f"Recall@10 vs epoch  (ef={PRIMARY_EF}, M_conj=48)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    out = figures_dir / "recall_vs_epoch.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_figure2(data, cfg, figures_dir):
    """mean_eh and mmd_squared vs epoch, dual y-axes."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    calib_epochs = cfg["adaptation"]["n_calibration_epochs"]

    for ax, sname in zip(axes, SCHEDULES):
        if sname not in data:
            ax.set_title(f"{SCHEDULE_TITLES[sname]}  (no data)")
            continue

        df = data[sname]
        schedule = _get_schedule_values(cfg, sname)
        sub = df[df["ef_search"] == PRIMARY_EF].sort_values("epoch_idx")

        _shade_drift(ax, schedule)

        # Calibration-phase EH baseline
        calib_eh = sub[sub["epoch_idx"] < calib_epochs]["mean_eh"].mean()

        color_eh = "steelblue"
        color_mmd = "darkorange"

        ax.plot(sub["epoch_idx"], sub["mean_eh"], color=color_eh,
                linewidth=2.0, label="mean EH", zorder=3)
        ax.axhline(calib_eh, color=color_eh, linewidth=1.2, linestyle="--",
                   alpha=0.7, label=f"calib. EH baseline ({calib_eh:.4f})", zorder=4)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Mean EH", fontsize=11, color=color_eh)
        ax.tick_params(axis="y", labelcolor=color_eh)

        ax2 = ax.twinx()
        ax2.plot(sub["epoch_idx"], sub["mmd_squared"], color=color_mmd,
                 linewidth=1.8, linestyle="--", alpha=0.85, label="MMD²", zorder=3)
        ax2.set_ylabel("MMD²", fontsize=11, color=color_mmd)
        ax2.tick_params(axis="y", labelcolor=color_mmd)

        n_epochs = len(schedule)
        ax.set_xlim(-0.5, n_epochs - 0.5)
        ax.set_xticks(range(0, n_epochs, 5))
        ax.set_title(SCHEDULE_TITLES[sname], fontsize=12, fontweight="bold")

        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9,
                  loc="upper right", framealpha=0.85)

    fig.suptitle(
        "EH signal and MMD² vs epoch — does the detector settle?",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    out = figures_dir / "eh_mmd_vs_epoch.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_figure3(data, cfg, figures_dir):
    """edges_added_this_epoch bar chart, coloured by phase."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, sname in zip(axes, SCHEDULES):
        if sname not in data:
            ax.set_title(f"{SCHEDULE_TITLES[sname]}  (no data)")
            continue

        df = data[sname]
        schedule = _get_schedule_values(cfg, sname)
        # edges_added_this_epoch is only stored for primary_ef rows to avoid duplication
        sub = df[df["ef_search"] == PRIMARY_EF].sort_values("epoch_idx")

        epochs = sub["epoch_idx"].values
        edges_added = sub["edges_added_this_epoch"].values
        colors = [PHASE_COLORS[_phase_of_epoch(schedule[e])] for e in epochs]

        ax.bar(epochs, edges_added, color=colors, width=0.8, zorder=3)

        # Phase legend proxies
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=PHASE_COLORS["pre_drift"], label="Pre-drift (t=0)"),
            Patch(facecolor=PHASE_COLORS["ramp"], label="Ramp (0<t<1)"),
            Patch(facecolor=PHASE_COLORS["plateau"], label="Plateau (t=1)"),
        ]
        ax.legend(handles=legend_elements, fontsize=9, loc="upper right", framealpha=0.85)

        n_epochs = len(schedule)
        ax.set_xlim(-0.5, n_epochs - 0.5)
        ax.set_xticks(range(0, n_epochs, 5))
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Conjugate edges added", fontsize=11)
        ax.set_title(SCHEDULE_TITLES[sname], fontsize=12, fontweight="bold")

    fig.suptitle(
        "Edges added per epoch — does repair activity decay on the plateau?",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    out = figures_dir / "edges_added_per_epoch.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_figure4(data, cfg, figures_dir):
    """Rolling repair event rate for both schedules on one axis."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(12, 5))

    window = 5
    colors_sched = {"gradual_plateau": "steelblue", "sudden_plateau": "tomato"}

    for sname in SCHEDULES:
        if sname not in data:
            continue
        df = data[sname]
        sub = df[df["ef_search"] == PRIMARY_EF].sort_values("epoch_idx")
        repair_series = sub["repair_happened"].astype(float)
        rolling = repair_series.rolling(window=window, center=True, min_periods=1).mean()
        ax.plot(
            sub["epoch_idx"], rolling,
            color=colors_sched[sname],
            linewidth=2.2,
            label=SCHEDULE_TITLES[sname],
            zorder=3,
        )
        # Scatter individual repair events
        repair_epochs = sub[sub["repair_happened"] == True]["epoch_idx"].values
        ax.scatter(
            repair_epochs,
            [1.02] * len(repair_epochs),
            marker="|", s=60,
            color=colors_sched[sname],
            zorder=4, alpha=0.6,
        )

    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--", zorder=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(f"Repair rate  ({window}-epoch rolling mean)", fontsize=11)
    ax.set_title(
        "Does the repair mechanism quiesce on a stable drift plateau?",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=10, framealpha=0.9)

    fig.tight_layout()
    out = figures_dir / "repair_rate.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def print_summary(data, cfg):
    calib_n = cfg["adaptation"]["n_calibration_epochs"]
    header = (
        f"{'Schedule':<22}  {'Plateau start':>13}  {'Last repair ep':>14}  "
        f"{'Total edges':>11}  {'R@10 ep10':>9}  {'R@10 final':>10}"
    )
    print("\n" + header)
    print("-" * len(header))

    for sname in SCHEDULES:
        if sname not in data:
            print(f"  {sname:<22}  (no data)")
            continue
        df = data[sname]
        schedule = _get_schedule_values(cfg, sname)
        sub = df[df["ef_search"] == PRIMARY_EF].sort_values("epoch_idx")

        # First plateau epoch
        plateau_start = next((i for i, t in enumerate(schedule) if t == 1.0), None)

        # Last epoch where repair fired
        repair_epochs = sub[sub["repair_happened"] == True]["epoch_idx"].values
        last_repair = int(repair_epochs[-1]) if len(repair_epochs) > 0 else -1

        # Total edges at end
        total_edges = int(sub["n_conjugate_edges"].iloc[-1])

        # Recall at epoch 10 and final
        r10 = sub[sub["epoch_idx"] == 10]["recall_at_k"]
        r10_val = float(r10.iloc[0]) if not r10.empty else float("nan")
        r_final = float(sub["recall_at_k"].iloc[-1])

        print(
            f"  {sname:<22}  {plateau_start:>13}  {last_repair:>14}  "
            f"{total_edges:>11,}  {r10_val:>9.4f}  {r_final:>10.4f}"
        )
    print()


def main():
    cfg = _load_config()
    out_dir = _out_dir(cfg)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = _load_all(cfg)
    if not data:
        print("No results found. Run run.py first.")
        return

    plot_figure1(data, cfg, figures_dir)
    plot_figure2(data, cfg, figures_dir)
    plot_figure3(data, cfg, figures_dir)
    plot_figure4(data, cfg, figures_dir)
    print_summary(data, cfg)


if __name__ == "__main__":
    main()
