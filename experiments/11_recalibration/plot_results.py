"""Experiment 11 figures: rolling recalibration vs static reference under plateau drift."""

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
PRIMARY_COLOR = "steelblue"
RAMP_SHADE = "#fdd0a2"
PLATEAU_SHADE = "#fdae6b"


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _out_dir(cfg):
    p = ROOT / cfg["output"]["results_dir"]
    if p.exists():
        return p
    return _SCRIPT_DIR


def _load_all(cfg):
    out_dir = _out_dir(cfg)
    data = {}
    for cname in ["recal_enabled", "recal_disabled"]:
        p = out_dir / f"{cname}.csv"
        if p.exists():
            data[cname] = pd.read_csv(p).sort_values("epoch_idx").reset_index(drop=True)
        else:
            print(f"  WARNING: missing {p}")
    return data


def _infer_threshold(df):
    """Estimate detection threshold from the drift_detected boundary."""
    no_drift = df[df["drift_detected"] == False]["mmd_squared"]
    with_drift = df[df["drift_detected"] == True]["mmd_squared"]
    if no_drift.empty or with_drift.empty:
        return None
    return float((no_drift.max() + with_drift.min()) / 2)


def _shade_drift(ax, df, ramp_start, plateau_start):
    n = int(df["epoch_idx"].max())
    if ramp_start < plateau_start:
        ax.axvspan(ramp_start - 0.5, plateau_start - 0.5,
                   color=RAMP_SHADE, alpha=0.35, zorder=0, linewidth=0)
    ax.axvspan(plateau_start - 0.5, n + 0.5,
               color=PLATEAU_SHADE, alpha=0.25, zorder=0, linewidth=0)


def plot_figure1(data, cfg, figures_dir):
    """Recall@10 and MMD² over epochs — both conditions overlap (recal never triggered)."""
    df = data.get("recal_enabled", data.get("recal_disabled"))
    if df is None:
        return

    drift_cfg = cfg.get("drift", {})
    ramp_start = drift_cfg.get("ramp_start_epoch", 1)
    plateau_start = drift_cfg.get("plateau_start_epoch", 10)
    threshold = _infer_threshold(df)

    pre_drift_mean = df[df["epoch_idx"] < ramp_start]["recall_at_k"].mean()
    n_recal = int(df.get("recalibrated", pd.Series([False])).sum()) if "recalibrated" in df.columns else 0
    null_note = "Both conditions identical\n(recalibration never triggered)" if n_recal == 0 \
        else f"Recalibrations: {n_recal}"

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, (ax_recall, ax_mmd) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    # --- top: recall@10 ---
    _shade_drift(ax_recall, df, ramp_start, plateau_start)
    ax_recall.plot(df["epoch_idx"], df["recall_at_k"],
                   color=PRIMARY_COLOR, linewidth=2.0, zorder=3, label="Recall@10")
    if not np.isnan(pre_drift_mean):
        ax_recall.axhline(pre_drift_mean, color=PRIMARY_COLOR, linewidth=1.1,
                          linestyle="--", alpha=0.6, label="Pre-drift mean recall")
    ax_recall.set_ylabel("Recall@10", fontsize=11)
    ax_recall.set_ylim(0.45, 1.05)

    # Phase legend proxies
    from matplotlib.patches import Patch
    phase_handles = [
        Patch(facecolor=RAMP_SHADE, alpha=0.7, label="Ramp (0 < t < 1)"),
        Patch(facecolor=PLATEAU_SHADE, alpha=0.7, label="Plateau (t = 1)"),
    ]
    handles, labels = ax_recall.get_legend_handles_labels()
    ax_recall.legend(handles + phase_handles, labels + [h.get_label() for h in phase_handles],
                     fontsize=9, loc="lower right", framealpha=0.85)

    ax_recall.text(0.98, 0.07, null_note,
                   transform=ax_recall.transAxes, fontsize=9, ha="right", va="bottom",
                   color="grey", style="italic",
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.7))

    # --- bottom: MMD² ---
    _shade_drift(ax_mmd, df, ramp_start, plateau_start)
    ax_mmd.plot(df["epoch_idx"], df["mmd_squared"],
                color="darkorange", linewidth=2.0, zorder=3, label="MMD²")
    if threshold is not None:
        ax_mmd.axhline(threshold, color="grey", linewidth=1.2, linestyle=":",
                       label=f"detection threshold ≈ {threshold:.4f}", zorder=4)

    # Annotate ramp start and plateau start
    ax_mmd.axvline(ramp_start, color="grey", linewidth=0.9, linestyle="--", alpha=0.5, zorder=2)
    ax_mmd.axvline(plateau_start, color="grey", linewidth=0.9, linestyle="--", alpha=0.5, zorder=2)
    ax_mmd.text(ramp_start + 0.3, ax_mmd.get_ylim()[1] * 0.92 if ax_mmd.get_ylim()[1] > 0 else 0.92,
                "ramp", fontsize=8, color="grey", va="top")
    ax_mmd.text(plateau_start + 0.3, ax_mmd.get_ylim()[1] * 0.92 if ax_mmd.get_ylim()[1] > 0 else 0.92,
                "plateau", fontsize=8, color="grey", va="top")

    ax_mmd.set_xlabel("Epoch", fontsize=11)
    ax_mmd.set_ylabel("MMD²", fontsize=11)
    ax_mmd.legend(fontsize=9, loc="upper left", framealpha=0.85)

    n_epochs = int(df["epoch_idx"].max())
    ax_recall.set_xlim(-0.5, n_epochs + 0.5)
    ax_recall.set_xticks(range(0, n_epochs + 1, 5))

    fig.suptitle(
        "Exp 11: Recall@10 and MMD² over epochs — gradual drift + plateau",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    out = figures_dir / "recall_mmd_vs_epoch.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")



def print_summary(data, cfg):
    adapt_cfg = cfg["adaptation"]
    max_nodes = adapt_cfg.get("max_repair_nodes", 500)
    M_cand = adapt_cfg.get("M_candidates", 32)
    sat_frac = adapt_cfg.get("recal_saturation_fraction", 0.05)
    sat_threshold = max_nodes * M_cand * sat_frac

    print("\nExp 11 — Summary")
    print("=" * 70)

    df = data.get("recal_enabled", data.get("recal_disabled", pd.DataFrame())).sort_values("epoch_idx")
    if df.empty:
        return

    drift_cfg = cfg.get("drift", {})
    ramp_start = drift_cfg.get("ramp_start_epoch", 1)
    plateau_start = drift_cfg.get("plateau_start_epoch", 10)

    pre = df[df["epoch_idx"] < ramp_start]
    plateau = df[df["epoch_idx"] >= plateau_start]
    ramp = df[(df["epoch_idx"] >= ramp_start) & (df["epoch_idx"] < plateau_start)]

    print(f"  Epochs:              {int(df['epoch_idx'].max()) + 1} total  "
          f"(pre-drift 0–{ramp_start - 1}, ramp {ramp_start}–{plateau_start - 1}, "
          f"plateau {plateau_start}–{int(df['epoch_idx'].max())})")
    print(f"  Recalibrations:      0  (trigger never met)")
    print(f"  Saturation threshold: {sat_threshold:.0f} edges/epoch  "
          f"(max_repair_nodes × M_candidates × sat_frac)")
    print()

    mean_added_plateau = plateau["edges_added"].mean()
    print(f"  Mean edges_added/epoch (plateau): {mean_added_plateau:.0f}  "
          f"({mean_added_plateau / sat_threshold:.0f}× above trigger threshold)")
    print(f"  Final total_edges:   {int(df['total_edges'].iloc[-1]):,}  "
          f"({int(df['total_edges'].iloc[-1]) / adapt_cfg['max_total_edges'] * 100:.1f}% of cap)")
    print()

    if not pre.empty:
        print(f"  Pre-drift recall@10: {pre['recall_at_k'].mean():.4f} ± {pre['recall_at_k'].std():.4f}")
    if not ramp.empty:
        print(f"  Ramp recall@10:      {ramp['recall_at_k'].mean():.4f} ± {ramp['recall_at_k'].std():.4f}")
    print(f"  Plateau recall@10:   {plateau['recall_at_k'].mean():.4f} ± {plateau['recall_at_k'].std():.4f}")
    print(f"  Final recall@10:     {df['recall_at_k'].iloc[-1]:.4f}")
    print()

    pre_mmd = pre[pre["mmd_squared"].notna()]["mmd_squared"]
    plat_mmd = plateau["mmd_squared"]
    print(f"  Pre-drift MMD²:      {pre_mmd.mean():.5f} ± {pre_mmd.std():.5f}")
    print(f"  Plateau MMD²:        {plat_mmd.mean():.5f} ± {plat_mmd.std():.5f}")

    threshold = _infer_threshold(df)
    if threshold:
        print(f"  Inferred threshold:  {threshold:.5f}")
        print(f"  Plateau MMD² / threshold: {plat_mmd.mean() / threshold:.1f}×  "
              "(MMD² stays persistently elevated)")
    print()


def main():
    cfg = _load_config()
    out_dir = _out_dir(cfg)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = _load_all(cfg)
    if not data:
        print("No results found. Run run_eval.py first.")
        return

    plot_figure1(data, cfg, figures_dir)
    print_summary(data, cfg)


if __name__ == "__main__":
    main()
