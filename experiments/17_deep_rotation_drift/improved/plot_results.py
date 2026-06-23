"""Plots for exp17-improved: static vs adaptive_current vs adaptive_improved
on DEEP-96 rotation drift.

Figures produced:
  recall_vs_epoch_gradual.pdf   — recall@10 vs epoch, 3 subplots (ef=32/64/128)
  recall_vs_epoch_sudden.pdf    — same for sudden schedule
  edges_vs_epoch.pdf            — cumulative edge count, gradual + sudden side by side
  ef_scaling_plateau.pdf        — key diagnostic: recall and gain vs ef at full-drift plateau
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]

COLORS = {
    "static":            "grey",
    "adaptive_current":  "steelblue",
    "adaptive_improved": "darkorange",
}
LABELS = {
    "static":            "Static HNSW",
    "adaptive_current":  r"Adaptive EH (current, $B$=1M)",
    "adaptive_improved": r"Adaptive EH (improved, $B$=5M)",
}
COND_ORDER = ["static", "adaptive_current", "adaptive_improved"]
DPI = 300


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _shade_transitions(ax, transitions, n_epochs=25):
    """Draw light vertical lines at drift-parameter transition epochs."""
    for t in transitions:
        ax.axvline(t, color="black", linestyle=":", linewidth=0.8, alpha=0.35)


def _shade_plateau(ax, start, end):
    """Shade the full-drift plateau region."""
    ax.axvspan(start - 0.5, end + 0.5, color="#fdae6b", alpha=0.12, zorder=0, linewidth=0)


# ---------------------------------------------------------------------------
# Figure 1+2: recall@10 vs epoch for each schedule
# ---------------------------------------------------------------------------

def plot_recall_vs_epoch(df, schedule, cfg, plots_dir):
    ef_values = sorted(df["ef"].unique())
    transitions = cfg.get(f"drift_transition_epochs_{schedule}", [])
    plateau_start = cfg.get("plateau_start", 18)
    plateau_end = cfg.get("plateau_end", 24)
    present = set(df["condition"].unique())

    fig, axes = plt.subplots(1, len(ef_values), figsize=(5.0 * len(ef_values), 4.2), sharey=True)
    if len(ef_values) == 1:
        axes = [axes]

    for ax, ef in zip(axes, ef_values):
        _shade_plateau(ax, plateau_start, plateau_end)
        _shade_transitions(ax, transitions)

        for cond in COND_ORDER:
            if cond not in present:
                continue
            sub = df[(df["condition"] == cond) & (df["ef"] == ef)].sort_values("epoch")
            ax.plot(
                sub["epoch"], sub["recall"],
                color=COLORS[cond], label=LABELS[cond],
                linewidth=1.8, marker="o", markersize=3, zorder=3,
            )

        max_epoch = df["epoch"].max()
        tick_step = 5 if max_epoch >= 30 else 4
        ax.set_title(f"ef={ef}", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_xlim(-0.5, max_epoch + 0.5)
        ax.set_xticks(range(0, max_epoch + 1, tick_step))
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylabel("Recall@10" if ef == ef_values[0] else "")
        ax.legend(fontsize=7, loc="lower left", framealpha=0.85)
        ax.grid(axis="y", linewidth=0.5, alpha=0.5)

    schedule_label = schedule.capitalize()
    fig.suptitle(
        f"DEEP-96 rotation drift ({schedule_label}) — Recall@10 vs Epoch",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    out = plots_dir / f"recall_vs_epoch_{schedule}.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 3: cumulative edge count vs epoch
# ---------------------------------------------------------------------------

def plot_edges_vs_epoch(df_grad, df_sudd, cfg, plots_dir):
    primary_ef = cfg["primary_ef"]
    transitions_g = cfg.get("drift_transition_epochs_gradual", [])
    transitions_s = cfg.get("drift_transition_epochs_sudden", [])
    plateau_start = cfg.get("plateau_start", 18)
    plateau_end = cfg.get("plateau_end", 24)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, df, transitions, title in [
        (axes[0], df_grad, transitions_g, "Gradual"),
        (axes[1], df_sudd, transitions_s, "Sudden"),
    ]:
        _shade_plateau(ax, plateau_start, plateau_end)
        _shade_transitions(ax, transitions)
        present = set(df["condition"].unique())

        for cond in ["adaptive_current", "adaptive_improved"]:
            if cond not in present:
                continue
            sub = df[(df["condition"] == cond) & (df["ef"] == primary_ef)].sort_values("epoch")
            ax.plot(
                sub["epoch"], sub["edge_count"] / 1e6,
                color=COLORS[cond], label=LABELS[cond],
                linewidth=1.8, marker="o", markersize=3,
            )

        max_epoch = df["epoch"].max()
        tick_step = 5 if max_epoch >= 30 else 4
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cumulative edges (×10⁶)")
        ax.set_title(f"{title} schedule", fontsize=10)
        ax.set_xlim(-0.5, max_epoch + 0.5)
        ax.set_xticks(range(0, max_epoch + 1, tick_step))
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, framealpha=0.85)
        ax.grid(axis="y", linewidth=0.5, alpha=0.5)

    fig.suptitle("DEEP-96 rotation drift — Cumulative conjugate edges", fontsize=11)
    fig.tight_layout()
    out = plots_dir / "edges_vs_epoch.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 4: ef-scaling diagnostic at full-drift plateau
# ---------------------------------------------------------------------------

def plot_ef_scaling(df_grad, df_sudd, cfg, plots_dir):
    """Two panels: absolute recall at plateau (top), gain over static (bottom).

    The slope of gain vs ef distinguishes structural failure (gain large,
    decreases moderately) from beam-width failure (gain near zero, collapses
    with ef).  Key thesis diagnostic for whether DEEP-96 rotation drift is
    fixable by conjugate repair.
    """
    plateau_start = cfg.get("plateau_start", 18)
    plateau_end = cfg.get("plateau_end", 24)
    ef_values = sorted(df_grad["ef"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for col, (df, schedule_label) in enumerate([(df_grad, "Gradual"), (df_sudd, "Sudden")]):
        plateau = df[(df["epoch"] >= plateau_start) & (df["epoch"] <= plateau_end)]
        present = set(df["condition"].unique())

        # Panel top: absolute plateau recall per (condition, ef)
        ax_top = axes[0, col]
        for cond in COND_ORDER:
            if cond not in present:
                continue
            means = [
                plateau[(plateau["condition"] == cond) & (plateau["ef"] == ef)]["recall"].mean()
                for ef in ef_values
            ]
            ax_top.plot(
                ef_values, means,
                color=COLORS[cond], label=LABELS[cond],
                linewidth=2, marker="o", markersize=5,
            )
        ax_top.set_title(f"{schedule_label} — Recall@10 at plateau (ep{plateau_start}–{plateau_end})")
        ax_top.set_xlabel("ef")
        ax_top.set_ylabel("Mean recall@10")
        ax_top.set_xticks(ef_values)
        ax_top.set_ylim(0, 1.05)
        ax_top.legend(fontsize=8, framealpha=0.85)
        ax_top.grid(linewidth=0.5, alpha=0.5)

        # Panel bottom: gain over static (adaptive - static) vs ef
        ax_bot = axes[1, col]
        static_means = {
            ef: plateau[(plateau["condition"] == "static") & (plateau["ef"] == ef)]["recall"].mean()
            for ef in ef_values
        }
        for cond in ["adaptive_current", "adaptive_improved"]:
            if cond not in present:
                continue
            gains = [
                plateau[(plateau["condition"] == cond) & (plateau["ef"] == ef)]["recall"].mean()
                - static_means[ef]
                for ef in ef_values
            ]
            ax_bot.plot(
                ef_values, [g * 100 for g in gains],
                color=COLORS[cond], label=LABELS[cond],
                linewidth=2, marker="o", markersize=5,
            )
            # Annotate each point with the gain value
            for ef, g in zip(ef_values, gains):
                ax_bot.annotate(
                    f"+{g*100:.1f}pp",
                    xy=(ef, g * 100),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=7, color=COLORS[cond],
                )

        ax_bot.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        ax_bot.set_title(f"{schedule_label} — Adaptive gain over static (pp)")
        ax_bot.set_xlabel("ef")
        ax_bot.set_ylabel("Gain over static (pp)")
        ax_bot.set_xticks(ef_values)
        ax_bot.legend(fontsize=8, framealpha=0.85)
        ax_bot.grid(linewidth=0.5, alpha=0.5)

    fig.suptitle(
        "DEEP-96 rotation drift — ef-scaling at full-drift plateau\n"
        "(decreasing gain → beam-width problem; sustained gain → structural failure)",
        fontsize=11,
    )
    fig.tight_layout()
    out = plots_dir / "ef_scaling_plateau.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Figure 5: detector signal vs epoch (mean_eh + drift_detected)
# ---------------------------------------------------------------------------

def plot_detector_signal(df_grad, df_sudd, cfg, plots_dir):
    """Show mean_eh and edge count on the same axes, one panel per schedule.

    Useful for diagnosing the sudden-schedule MMD² dropout that plagued exp17:
    if mean_eh stays elevated while edges plateau or stall, the improved adapter
    continued repairing via per-query EH even when epoch-level MMD² collapsed.
    """
    primary_ef = cfg["primary_ef"]
    plateau_start = cfg.get("plateau_start", 18)
    plateau_end = cfg.get("plateau_end", 24)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, df, schedule_label in [
        (axes[0], df_grad, "Gradual"),
        (axes[1], df_sudd, "Sudden"),
    ]:
        present = set(df["condition"].unique())
        _shade_plateau(ax, plateau_start, plateau_end)

        ax2 = ax.twinx()

        for cond in ["adaptive_current", "adaptive_improved"]:
            if cond not in present:
                continue
            sub = df[(df["condition"] == cond) & (df["ef"] == primary_ef)].sort_values("epoch")

            ax.plot(
                sub["epoch"], sub["mean_eh"],
                color=COLORS[cond], linewidth=1.5, linestyle="--", alpha=0.7,
                label=f"EH ({LABELS[cond]})",
            )
            ax2.plot(
                sub["epoch"], sub["edge_count"] / 1e6,
                color=COLORS[cond], linewidth=1.5,
                label=f"Edges ({LABELS[cond]})",
            )

        max_epoch = df["epoch"].max()
        tick_step = 5 if max_epoch >= 30 else 4
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean EH (dashed)", color="black")
        ax2.set_ylabel("Cumulative edges ×10⁶ (solid)")
        ax.set_title(f"{schedule_label} — EH signal & edge accumulation")
        ax.set_xlim(-0.5, max_epoch + 0.5)
        ax.set_xticks(range(0, max_epoch + 1, tick_step))
        ax.set_ylim(0, 1.05)
        ax2.set_ylim(bottom=0)

        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc="upper left", framealpha=0.85)
        ax.grid(axis="y", linewidth=0.5, alpha=0.4)

    fig.suptitle(
        "DEEP-96 rotation drift — Drift signal and edge accumulation\n"
        "(exp17 sudden-schedule: MMD² collapsed at ep19 but EH stayed elevated → "
        "improved adapter continues repair via per-query EH)",
        fontsize=9,
    )
    fig.tight_layout()
    out = plots_dir / "detector_signal.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "config.yaml"
    cfg = _load_config(config_path)
    plots_dir = ROOT / cfg["plots_dir"]
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for schedule in ("gradual", "sudden"):
        key = f"results_{schedule}"
        path = ROOT / cfg[key]
        if not path.exists():
            print(f"Missing {path} — skipping {schedule} schedule")
            continue
        results[schedule] = pd.read_csv(path)
        print(f"Loaded {path}: {len(results[schedule])} rows, "
              f"conditions: {sorted(results[schedule]['condition'].unique())}")

    if not results:
        print("No results found. Run run_experiment.py first.")
        return

    df_grad = results.get("gradual")
    df_sudd = results.get("sudden")

    if df_grad is not None:
        print("\nPlotting recall vs epoch (gradual)...")
        plot_recall_vs_epoch(df_grad, "gradual", cfg, plots_dir)

    if df_sudd is not None:
        print("Plotting recall vs epoch (sudden)...")
        plot_recall_vs_epoch(df_sudd, "sudden", cfg, plots_dir)

    if df_grad is not None and df_sudd is not None:
        print("Plotting edge count vs epoch...")
        plot_edges_vs_epoch(df_grad, df_sudd, cfg, plots_dir)

        print("Plotting ef-scaling diagnostic...")
        plot_ef_scaling(df_grad, df_sudd, cfg, plots_dir)

        print("Plotting detector signal...")
        plot_detector_signal(df_grad, df_sudd, cfg, plots_dir)

    print(f"\nAll plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
