"""Experiment 13 figures: bridge repair vs baseline under gradual drift + plateau."""

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
RAMP_SHADE = "#fdd0a2"
PLATEAU_SHADE = "#fdae6b"
COLORS = {
    "baseline": "steelblue",
    "bridge_only": "crimson",
}
LABELS = {
    "baseline": "Baseline (hot-cell repair only)",
    "bridge_only": "Bridge repair (hot-cell + bridge)",
}


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
    for cname in ["baseline", "bridge_only"]:
        p = out_dir / f"{cname}.csv"
        if p.exists():
            data[cname] = pd.read_csv(p).sort_values("epoch_idx").reset_index(drop=True)
        else:
            print(f"  WARNING: missing {p}")
    return data


def _shade_drift(ax, df, ramp_start, plateau_start):
    n = int(df["epoch_idx"].max())
    if ramp_start < plateau_start:
        ax.axvspan(ramp_start - 0.5, plateau_start - 0.5,
                   color=RAMP_SHADE, alpha=0.35, zorder=0, linewidth=0)
    ax.axvspan(plateau_start - 0.5, n + 0.5,
               color=PLATEAU_SHADE, alpha=0.25, zorder=0, linewidth=0)


def plot_recall(data, cfg, figures_dir):
    """Recall@10 vs epoch for all conditions with bridge edge counts."""
    if not data:
        print("No data to plot.")
        return

    drift_cfg = cfg.get("drift", {})
    ramp_start = drift_cfg.get("ramp_start_epoch", 1)
    plateau_start = drift_cfg.get("plateau_start_epoch", 10)

    # Infer n_epochs from any available df
    any_df = next(iter(data.values()))
    n_epochs = int(any_df["epoch_idx"].max())

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    ax_recall, ax_edges, ax_bridge = axes

    from matplotlib.patches import Patch

    # --- top: recall@10 ---
    _shade_drift(ax_recall, any_df, ramp_start, plateau_start)
    for cname, df in data.items():
        ax_recall.plot(
            df["epoch_idx"], df["recall_at_k"],
            color=COLORS.get(cname, "grey"), linewidth=2.0, zorder=3,
            label=LABELS.get(cname, cname),
        )
    ax_recall.set_ylabel("Recall@10", fontsize=11)
    ax_recall.set_ylim(0.45, 1.05)

    phase_handles = [
        Patch(facecolor=RAMP_SHADE, alpha=0.7, label="Ramp (0 < t < 1)"),
        Patch(facecolor=PLATEAU_SHADE, alpha=0.7, label="Plateau (t = 1)"),
    ]
    handles, labels = ax_recall.get_legend_handles_labels()
    ax_recall.legend(
        handles + phase_handles,
        labels + [h.get_label() for h in phase_handles],
        fontsize=9, loc="lower right", framealpha=0.85,
    )

    # Annotate plateau recall delta if both conditions present
    if "baseline" in data and "bridge_only" in data:
        df_b = data["baseline"]
        df_br = data["bridge_only"]
        plateau_b = df_b[df_b["epoch_idx"] >= plateau_start]["recall_at_k"].mean()
        plateau_br = df_br[df_br["epoch_idx"] >= plateau_start]["recall_at_k"].mean()
        delta = plateau_br - plateau_b
        note = f"Plateau mean: baseline={plateau_b:.3f}, bridge={plateau_br:.3f}  (Δ={delta:+.3f})"
        ax_recall.text(
            0.01, 0.04, note, transform=ax_recall.transAxes,
            fontsize=8, ha="left", va="bottom", color="grey", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.7),
        )

    # --- middle: total conjugate edges ---
    _shade_drift(ax_edges, any_df, ramp_start, plateau_start)
    for cname, df in data.items():
        ax_edges.plot(
            df["epoch_idx"], df["total_edges"],
            color=COLORS.get(cname, "grey"), linewidth=1.8, zorder=3,
            label=LABELS.get(cname, cname),
        )
    ax_edges.set_ylabel("Total conjugate edges", fontsize=11)
    ax_edges.legend(fontsize=9, loc="upper left", framealpha=0.85)

    # --- bottom: bridge edges added per epoch ---
    _shade_drift(ax_bridge, any_df, ramp_start, plateau_start)
    if "bridge_only" in data:
        df_br = data["bridge_only"]
        ax_bridge.bar(
            df_br["epoch_idx"], df_br["bridge_edges_added"],
            color=COLORS["bridge_only"], alpha=0.6, width=0.6, zorder=3,
            label="Bridge edges added",
        )
    ax_bridge.set_ylabel("Bridge edges / epoch", fontsize=11)
    ax_bridge.set_xlabel("Epoch", fontsize=11)
    ax_bridge.legend(fontsize=9, loc="upper right", framealpha=0.85)

    ax_recall.set_xlim(-0.5, n_epochs + 0.5)
    ax_recall.set_xticks(range(0, n_epochs + 1, 5))

    fig.suptitle(
        "Exp 13: Bridge repair — stuck node bridging vs baseline",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    out = figures_dir / "recall_bridge_vs_epoch.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def print_summary(data, cfg):
    drift_cfg = cfg.get("drift", {})
    ramp_start = drift_cfg.get("ramp_start_epoch", 1)
    plateau_start = drift_cfg.get("plateau_start_epoch", 10)

    print("\nExp 13 — Summary")
    print("=" * 70)

    for cname in ["baseline", "bridge_only"]:
        df = data.get(cname)
        if df is None:
            continue
        pre = df[df["epoch_idx"] < ramp_start]
        ramp = df[(df["epoch_idx"] >= ramp_start) & (df["epoch_idx"] < plateau_start)]
        plateau = df[df["epoch_idx"] >= plateau_start]

        print(f"\nCondition: {cname}")
        if not pre.empty:
            print(f"  Pre-drift recall@10:  {pre['recall_at_k'].mean():.4f}")
        if not ramp.empty:
            print(f"  Ramp recall@10:       {ramp['recall_at_k'].mean():.4f}")
        if not plateau.empty:
            print(f"  Plateau recall@10:    {plateau['recall_at_k'].mean():.4f}")
        print(f"  Final total_edges:    {int(df['total_edges'].iloc[-1]):,}")
        if "bridge_edges_added" in df.columns:
            total_bridge = int(df["bridge_edges_added"].sum())
            mean_bridge = df[df["bridge_edges_added"] > 0]["bridge_edges_added"].mean()
            print(f"  Total bridge edges:   {total_bridge:,}")
            if not np.isnan(mean_bridge):
                print(f"  Mean bridge/repair epoch: {mean_bridge:.0f}")

    if "baseline" in data and "bridge_only" in data:
        df_b = data["baseline"]
        df_br = data["bridge_only"]
        plateau_b = df_b[df_b["epoch_idx"] >= plateau_start]["recall_at_k"].mean()
        plateau_br = df_br[df_br["epoch_idx"] >= plateau_start]["recall_at_k"].mean()
        print(f"\nPlateau recall delta (bridge - baseline): {plateau_br - plateau_b:+.4f}")


def main():
    cfg = _load_config()
    out_dir = _out_dir(cfg)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = _load_all(cfg)
    if not data:
        print("No results found. Run run_eval.py first.")
        return

    plot_recall(data, cfg, figures_dir)
    print_summary(data, cfg)


if __name__ == "__main__":
    main()
