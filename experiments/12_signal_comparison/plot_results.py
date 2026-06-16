"""Experiment 12 figures: signal comparison (EH vs QRD vs NND vs CSR)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, auc
import yaml

ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = _SCRIPT_DIR / "config.yaml"
DPI = 300

SIGNAL_COLORS = {
    "eh": "#555555",
    "qrd": "#2171b5",
    "nnd": "#cb181d",
    "csr": "#238b45",
    "joint": "#7b2d8b",
}
SIGNAL_LABELS = {
    "eh": "EH (baseline)",
    "qrd": "QRD",
    "nnd": "NND",
    "csr": "CSR",
    "joint": "Joint (EH+QRD)",
}


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _results_dir(cfg):
    p = ROOT / cfg["output"]["results_dir"]
    if p.exists():
        return p
    return _SCRIPT_DIR


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


def _normalise_series(s):
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-12:
        return s * 0.0
    return (s - lo) / (hi - lo)


def plot_figure1(df, cfg, signals, figures_dir):
    """Signal means normalised to [0,1] vs epoch, overlaid with recall@10."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(12, 5))

    schedule = cfg["drift"]["schedule_gradual"]
    _shade_drift(ax, schedule)

    epochs = df["epoch"].values

    # Recall on right y-axis
    ax2 = ax.twinx()
    ax2.plot(epochs, df["recall"].values, color="black", linewidth=2.2,
             linestyle="--", label="Recall@10", zorder=5)
    ax2.set_ylabel("Recall@10", fontsize=11)
    ax2.set_ylim(0, 1.05)

    for sig in signals:
        if sig == "joint":
            # joint has no per-query mean — overlay normalised MMD² instead
            col = "joint_mmd_sq"
            if col not in df.columns:
                continue
            normed = _normalise_series(df[col].fillna(0))
            ax.plot(epochs, normed.values, color=SIGNAL_COLORS.get("joint", "purple"),
                    linewidth=1.8, marker="s", markersize=3, linestyle="-.",
                    label=SIGNAL_LABELS.get("joint", "Joint (EH+QRD)"), zorder=3)
            continue
        col = f"{sig}_mean"
        if col not in df.columns:
            continue
        normed = _normalise_series(df[col])
        ax.plot(epochs, normed.values, color=SIGNAL_COLORS.get(sig, "grey"),
                linewidth=1.8, marker="o", markersize=3,
                label=SIGNAL_LABELS.get(sig, sig), zorder=3)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Signal mean (normalised to [0,1])", fontsize=11)
    ax.set_title("Drift signals vs. recall@10 — gradual drift", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.5, len(schedule) - 0.5)
    ax.set_xticks(range(0, len(schedule), 2))

    # Annotate drift phase boundaries
    for x_annot, label in [(5, "drift\nstarts"), (19, "max\ndrift")]:
        ax.axvline(x_annot, color="grey", linestyle=":", linewidth=1.0, alpha=0.6)
        ax.text(x_annot + 0.15, 0.98, label, fontsize=7, color="grey",
                transform=ax.get_xaxis_transform(), va="top")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="lower left",
              framealpha=0.85, ncol=2)

    fig.tight_layout()
    out = figures_dir / "fig1_signals_vs_recall.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_figure2(df, cfg, signals, figures_dir):
    """Scatter: signal mean (x) vs recall@10 (y) per epoch; Pearson r annotated."""
    n_signals = len(signals)
    sns.set_theme(style="whitegrid", font_scale=0.95)
    fig, axes = plt.subplots(1, n_signals, figsize=(4.5 * n_signals, 4.5), sharey=True)
    if n_signals == 1:
        axes = [axes]

    epochs = df["epoch"].values
    recalls = df["recall"].values
    cmap = plt.cm.viridis

    for ax, sig in zip(axes, signals):
        if sig == "joint":
            col = "joint_mmd_sq"
        else:
            col = f"{sig}_mean"
        if col not in df.columns:
            ax.set_title(f"{sig} (no data)", fontsize=11)
            continue

        xvals = df[col].fillna(0).values if sig == "joint" else df[col].values
        r, pval = pearsonr(xvals, recalls)

        sc = ax.scatter(xvals, recalls, c=epochs, cmap=cmap, s=40, alpha=0.85,
                        edgecolors="none", zorder=3)
        xlabel = f"{SIGNAL_LABELS.get(sig, sig)} (MMD²)" if sig == "joint" else SIGNAL_LABELS.get(sig, sig)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Recall@10" if ax is axes[0] else "", fontsize=11)
        ax.set_title(f"{SIGNAL_LABELS.get(sig, sig)}", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.annotate(f"r = {r:.3f}", xy=(0.05, 0.08), xycoords="axes fraction",
                    fontsize=10, color="black",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    # Colourbar for epoch
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=epochs.min(), vmax=epochs.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("Epoch", fontsize=10)

    fig.suptitle("Signal mean vs. Recall@10 (one point per epoch)", fontsize=12,
                 fontweight="bold")
    fig.tight_layout()
    out = figures_dir / "fig2_scatter_signal_vs_recall.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_figure3(df, cfg, signals, figures_dir):
    """ROC curves: signal MMD² > threshold → drift declared, vs recall < 0.85 as ground truth."""
    roc_threshold = cfg.get("drift_threshold_for_roc", 0.85)
    y_true = (df["recall"].values < roc_threshold).astype(int)

    if y_true.sum() == 0:
        print("No epochs below recall threshold — ROC plot skipped.")
        return

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(7, 6))

    for sig in signals:
        col = f"{sig}_mmd_sq"
        if col not in df.columns:
            continue
        scores = df[col].fillna(0).values
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        ls = "--" if sig == "eh" else "-"
        ax.plot(fpr, tpr, color=SIGNAL_COLORS.get(sig, "grey"), linewidth=2.0,
                linestyle=ls, label=f"{SIGNAL_LABELS.get(sig, sig)}  AUC={roc_auc:.3f}",
                zorder=3 + (sig != "eh"))

    ax.plot([0, 1], [0, 1], color="grey", linewidth=1.0, linestyle=":", alpha=0.5)
    ax.set_xlabel("False positive rate", fontsize=11)
    ax.set_ylabel("True positive rate", fontsize=11)
    ax.set_title(
        f"ROC: signal MMD² as drift detector  (ground truth: recall < {roc_threshold})",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)

    fig.tight_layout()
    out = figures_dir / "fig3_roc_curves.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    cfg = _load_config()
    results_dir = _results_dir(cfg)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "signals.csv"
    if not csv_path.exists():
        print(f"No results found at {csv_path}. Run run_eval.py first.")
        return

    df = pd.read_csv(csv_path)
    signals = cfg.get("signals", ["eh", "qrd", "nnd", "csr"])

    plot_figure1(df, cfg, signals, figures_dir)
    plot_figure2(df, cfg, signals, figures_dir)
    plot_figure3(df, cfg, signals, figures_dir)


if __name__ == "__main__":
    main()
