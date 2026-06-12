"""Publication-quality figures for experiment 02 adaptive evaluation."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
# CSVs live alongside this script; fall back to results/ for cluster runs
_SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = _SCRIPT_DIR if (_SCRIPT_DIR / "gradual_all.csv").exists() \
    else ROOT / "results" / "02_adaptive_eval"
FIGURES_DIR = _SCRIPT_DIR / "figures"
CONFIG_PATH = _SCRIPT_DIR / "config.yaml"

COLOURS = {
    "static_hnsw":      "#d62728",
    "periodic_rebuild": "#ff7f0e",
    "adaptive":         "#2ca02c",
}
LABELS = {
    "static_hnsw":      "Static HNSW",
    "periodic_rebuild": "Periodic Rebuild",
    "adaptive":         "Adaptive (ours)",
}
METHODS = ["static_hnsw", "periodic_rebuild", "adaptive"]
PRIMARY_EF = 32
DPI = 300


# ── helpers ──────────────────────────────────────────────────────────────────

def load_data():
    gradual = pd.read_csv(RESULTS_DIR / "gradual_all.csv")
    sudden  = pd.read_csv(RESULTS_DIR / "sudden_all.csv")
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    return gradual, sudden, config


def _classify_phase(t):
    if t == 0.0:     return "pre"
    if t < 0.5:      return "ramp1"
    if t == 0.5:     return "stable"
    if t < 1.0:      return "ramp2"
    return "max"


_PHASE_META = {
    "pre":    dict(label="Pre-drift (t=0)",        color="#deebf7", alpha=0.0),
    "ramp1":  dict(label="Drift onset (t: 0→½)",   color="#9ecae1", alpha=0.30),
    "stable": dict(label="Partial drift (t=½)",    color="#fdae6b", alpha=0.30),
    "ramp2":  dict(label="Drift increase (t: \u00bd\u21921)", color="#fd8d3c", alpha=0.30),
    "max":    dict(label="Max drift (t=1)",         color="#e6550d", alpha=0.25),
}


def phase_spans(schedule):
    """Return list of (start, end, phase_key) for each contiguous block."""
    labels = [_classify_phase(t) for t in schedule]
    spans, cur, start = [], labels[0], 0
    for i, ph in enumerate(labels[1:], 1):
        if ph != cur:
            spans.append((start, i - 1, cur))
            cur, start = ph, i
    spans.append((start, len(labels) - 1, cur))
    return spans


def shade_phases(ax, spans, ymin, ymax):
    for start, end, ph in spans:
        meta = _PHASE_META[ph]
        if meta["alpha"] == 0.0:
            continue
        ax.axvspan(start - 0.5, end + 0.5, color=meta["color"],
                   alpha=meta["alpha"], zorder=0, linewidth=0)
        mid = (start + end) / 2
        ax.text(mid, ymax - (ymax - ymin) * 0.01, meta["label"],
                ha="center", va="top", fontsize=6, color="#555555",
                fontstyle="italic", clip_on=True)


def phase_legend_patches(spans):
    seen = {}
    for _, _, ph in spans:
        if ph not in seen and _PHASE_META[ph]["alpha"] > 0:
            seen[ph] = matplotlib.patches.Patch(
                facecolor=_PHASE_META[ph]["color"],
                alpha=_PHASE_META[ph]["alpha"],
                label=_PHASE_META[ph]["label"],
                edgecolor="none",
            )
    return list(seen.values())


# ── figure 1: gradual main result ─────────────────────────────────────────────

def fig1_gradual(gradual, config):
    schedule = config["drift"]["schedule_gradual"]
    spans = phase_spans(schedule)
    n_calibration = config["adaptation"]["n_calibration_epochs"]
    max_edges = config["adaptation"]["max_total_edges"]

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Gradual Drift — Adaptive vs Baselines", fontsize=13, fontweight="bold")

    # ── left: recall vs epoch ──
    ax = axes[0]
    ef32 = gradual[gradual["ef_search"] == PRIMARY_EF]
    ymin, ymax = 0.45, 1.01
    shade_phases(ax, spans, ymin, ymax)

    for method in METHODS:
        m = ef32[ef32["method"] == method].sort_values("epoch_idx")
        ax.plot(m["epoch_idx"], m["recall_at_k"],
                color=COLOURS[method], label=LABELS[method],
                linewidth=2.0, marker="o", markersize=4, zorder=3)

    # first genuine drift detection (skip first n_calibration epochs)
    adp = ef32[ef32["method"] == "adaptive"].sort_values("epoch_idx")
    post_cal = adp[adp["epoch_idx"] >= n_calibration]
    first_det = post_cal[post_cal["drift_detected"] == True]["epoch_idx"].min()
    if pd.notna(first_det):
        ax.axvline(first_det, color=COLOURS["adaptive"], linestyle="--",
                   linewidth=1.5, alpha=0.85, zorder=4,
                   label=f"First detection (epoch {int(first_det)})")

    method_handles = [
        matplotlib.lines.Line2D([0], [0], color=COLOURS[m], linewidth=2,
                                 marker="o", markersize=4, label=LABELS[m])
        for m in METHODS
    ]
    if pd.notna(first_det):
        method_handles.append(
            matplotlib.lines.Line2D([0], [0], color=COLOURS["adaptive"],
                                     linestyle="--", linewidth=1.5,
                                     label=f"First detection (epoch {int(first_det)})")
        )
    phase_handles = phase_legend_patches(spans)
    ax.legend(handles=method_handles + phase_handles,
              fontsize=8, loc="lower left", framealpha=0.85)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Recall@10", fontsize=11)
    ax.set_title(f"Recall@10 vs Epoch  (ef={PRIMARY_EF})", fontsize=11)
    ax.set_xlim(-0.5, 24.5)
    ax.set_ylim(0.45, 1.01)
    ax.set_xticks(range(0, 25, 2))

    # ── right: conjugate edge count ──
    ax2 = axes[1]
    adp_all = gradual[(gradual["method"] == "adaptive") &
                      (gradual["ef_search"] == PRIMARY_EF)].sort_values("epoch_idx")
    ax2.bar(adp_all["epoch_idx"], adp_all["n_conjugate_edges"],
            color=COLOURS["adaptive"], alpha=0.75, zorder=3)
    ax2.axhline(max_edges, color="black", linestyle="--", linewidth=1.5,
                label=f"Capacity limit ({max_edges:,})", zorder=4)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Conjugate edges", fontsize=11)
    ax2.set_title("Conjugate Graph Size Over Time", fontsize=11)
    ax2.set_xlim(-0.5, 24.5)
    ax2.set_xticks(range(0, 25, 2))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax2.legend(fontsize=9)

    fig.tight_layout()
    out = FIGURES_DIR / "main_result_gradual.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ── figure 2: sudden main result ──────────────────────────────────────────────

def fig2_sudden(sudden, config):
    max_edges = config["adaptation"]["max_total_edges"]
    n_calibration = config["adaptation"]["n_calibration_epochs"]
    # sudden drift jump epoch: first epoch where t==1.0 in sudden schedule
    schedule_s = config["drift"]["schedule_sudden"]
    drift_jump = next(i for i, t in enumerate(schedule_s) if t == 1.0)

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Sudden Drift — Adaptive vs Baselines", fontsize=13, fontweight="bold")

    # ── left: recall vs epoch ──
    ax = axes[0]
    ef32 = sudden[sudden["ef_search"] == PRIMARY_EF]
    ymin, ymax = 0.75, 1.01

    for method in METHODS:
        m = ef32[ef32["method"] == method].sort_values("epoch_idx")
        ax.plot(m["epoch_idx"], m["recall_at_k"],
                color=COLOURS[method], label=LABELS[method],
                linewidth=2.0, marker="o", markersize=4, zorder=3)

    ax.axvline(drift_jump - 0.5, color="black", linestyle="--", linewidth=1.8,
               alpha=0.7, zorder=4, label=f"Sudden drift (epoch {drift_jump})")

    # pre-drift / post-drift shading
    ax.axvspan(-0.5, drift_jump - 0.5, color="#deebf7", alpha=0.0, zorder=0)
    ax.axvspan(drift_jump - 0.5, 24.5,  color="#e6550d", alpha=0.20, zorder=0,
               label="Post-drift (t=1)")

    adp = ef32[ef32["method"] == "adaptive"].sort_values("epoch_idx")
    post_cal = adp[adp["epoch_idx"] >= n_calibration]
    first_det = post_cal[post_cal["drift_detected"] == True]["epoch_idx"].min()
    if pd.notna(first_det):
        ax.axvline(first_det, color=COLOURS["adaptive"], linestyle=":",
                   linewidth=1.8, alpha=0.85, zorder=4,
                   label=f"First detection (epoch {int(first_det)})")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Recall@10", fontsize=11)
    ax.set_title(f"Recall@10 vs Epoch  (ef={PRIMARY_EF})", fontsize=11)
    ax.set_xlim(-0.5, 24.5)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(range(0, 25, 2))
    ax.legend(fontsize=8, loc="lower left", framealpha=0.85)

    # ── right: conjugate edges ──
    ax2 = axes[1]
    adp_all = sudden[(sudden["method"] == "adaptive") &
                     (sudden["ef_search"] == PRIMARY_EF)].sort_values("epoch_idx")
    ax2.bar(adp_all["epoch_idx"], adp_all["n_conjugate_edges"],
            color=COLOURS["adaptive"], alpha=0.75, zorder=3)
    ax2.axhline(max_edges, color="black", linestyle="--", linewidth=1.5,
                label=f"Capacity limit ({max_edges:,})", zorder=4)
    ax2.axvline(drift_jump - 0.5, color="black", linestyle="--",
                linewidth=1.3, alpha=0.6, zorder=4)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Conjugate edges", fontsize=11)
    ax2.set_title("Conjugate Graph Size Over Time", fontsize=11)
    ax2.set_xlim(-0.5, 24.5)
    ax2.set_xticks(range(0, 25, 2))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax2.legend(fontsize=9)

    fig.tight_layout()
    out = FIGURES_DIR / "main_result_sudden.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ── figure 3: recall–latency tradeoff ─────────────────────────────────────────

def fig3_latency(gradual, sudden):
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    fig.suptitle("Recall–Latency Tradeoff (pre-drift vs post-drift)", fontsize=13,
                 fontweight="bold")

    for ax, (df, scenario) in zip(axes, [(gradual, "Gradual"), (sudden, "Sudden")]):
        for method in METHODS:
            mdf = df[df["method"] == method]
            # adaptive QPS is duplicated across ef values — deduplicate
            if method == "adaptive":
                ef_values = [PRIMARY_EF]
            else:
                ef_values = sorted(df["ef_search"].unique())

            pre_x, pre_y   = [], []
            post_x, post_y = [], []
            for ef in ef_values:
                edf = mdf[mdf["ef_search"] == ef]
                pre  = edf[edf["epoch_idx"] <= 4]
                post = edf[edf["epoch_idx"] >= 20]
                if len(pre) > 0:
                    pre_x.append(pre["queries_per_second"].mean())
                    pre_y.append(pre["recall_at_k"].mean())
                if len(post) > 0:
                    post_x.append(post["queries_per_second"].mean())
                    post_y.append(post["recall_at_k"].mean())

            c = COLOURS[method]
            label = LABELS[method]
            # pre-drift: solid filled marker
            ax.scatter(pre_x, pre_y, color=c, marker="o", s=100, zorder=4,
                       label=f"{label} (pre)")
            # post-drift: hollow marker
            ax.scatter(post_x, post_y, color=c, marker="o", s=60,
                       facecolors="none", edgecolors=c, linewidths=2.0, zorder=4,
                       label=f"{label} (post)")
            # connect pre→post per ef
            for px, py, ox, oy in zip(pre_x, pre_y, post_x, post_y):
                ax.annotate("", xy=(ox, oy), xytext=(px, py),
                            arrowprops=dict(arrowstyle="-|>", color=c, alpha=0.5,
                                            lw=1.2, mutation_scale=10))

        ax.set_xlabel("Queries per second (QPS)", fontsize=11)
        ax.set_ylabel("Recall@10", fontsize=11)
        ax.set_title(f"{scenario} drift", fontsize=11)
        # custom legend: method colours
        method_handles = [
            matplotlib.patches.Patch(facecolor=COLOURS[m], label=LABELS[m])
            for m in METHODS
        ]
        style_handles = [
            matplotlib.lines.Line2D([0], [0], marker="o", color="gray",
                                     markersize=8, linestyle="", label="Pre-drift"),
            matplotlib.lines.Line2D([0], [0], marker="o", color="gray",
                                     markersize=6, linestyle="",
                                     markerfacecolor="none", markeredgewidth=1.8,
                                     label="Post-drift"),
        ]
        ax.legend(handles=method_handles + style_handles, fontsize=8, framealpha=0.85)

        note = ("Note: adaptive rows are duplicated across ef values in results;\n"
                "QPS shown for primary ef=32 only.")
        ax.text(0.98, 0.02, note, transform=ax.transAxes, fontsize=6,
                ha="right", va="bottom", color="gray", fontstyle="italic")

    fig.tight_layout()
    out = FIGURES_DIR / "recall_latency_tradeoff.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ── figure 4: drift detection timing ──────────────────────────────────────────

def fig4_detection(gradual, config):
    n_calibration = config["adaptation"]["n_calibration_epochs"]
    schedule = config["drift"]["schedule_gradual"]
    spans = phase_spans(schedule)

    adp = (gradual[(gradual["method"] == "adaptive") &
                   (gradual["ef_search"] == PRIMARY_EF)]
           .sort_values("epoch_idx").reset_index(drop=True))

    # estimate threshold from calibration period (epochs 0 to n_calibration-1)
    calib_mmd = adp[adp["epoch_idx"] < n_calibration]["mmd_squared"].dropna()
    if len(calib_mmd) > 0:
        threshold = float(np.percentile(calib_mmd, 95))
    else:
        threshold = None

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(10, 4.5))

    ymin_mmd = 0.0
    ymax_mmd = adp["mmd_squared"].max() * 1.15
    shade_phases(ax, spans, ymin_mmd, ymax_mmd)

    # plot MMD² line coloured by drift_detected
    epochs = adp["epoch_idx"].values
    mmd    = adp["mmd_squared"].values
    det    = adp["drift_detected"].values.astype(bool)

    # draw segments
    for i in range(len(epochs) - 1):
        colour = "#d62728" if det[i] else "#2ca02c"
        ax.plot(epochs[i:i+2], mmd[i:i+2], color=colour, linewidth=2.5, zorder=3)
    # draw markers
    for i in range(len(epochs)):
        colour = "#d62728" if det[i] else "#2ca02c"
        ax.plot(epochs[i], mmd[i], "o", color=colour, markersize=6, zorder=4)

    if threshold is not None:
        ax.axhline(threshold, color="black", linestyle="--", linewidth=1.5,
                   alpha=0.75, zorder=4,
                   label=f"Calibration threshold (p95 = {threshold:.4f})")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("MMD²", fontsize=11, color="#333333")
    ax.set_xlim(-0.5, 24.5)
    ax.set_ylim(ymin_mmd, ymax_mmd)
    ax.set_xticks(range(0, 25, 2))
    ax.set_title("Drift Detector Signal Over Time  (Gradual schedule)", fontsize=12,
                 fontweight="bold")

    # secondary Y-axis: mean EH
    ax2 = ax.twinx()
    ax2.plot(adp["epoch_idx"], adp["mean_eh"], color="#9467bd",
             linewidth=1.5, linestyle="-.", marker="s", markersize=4,
             zorder=3, alpha=0.85, label="Mean EH")
    ax2.set_ylabel("Mean Escape Hardness (EH)", fontsize=10, color="#9467bd")
    ax2.tick_params(axis="y", labelcolor="#9467bd")

    # legend combining both axes
    handles = []
    handles.append(matplotlib.lines.Line2D(
        [0], [0], color="#2ca02c", linewidth=2.5, marker="o", markersize=6,
        label="MMD² — no drift"))
    handles.append(matplotlib.lines.Line2D(
        [0], [0], color="#d62728", linewidth=2.5, marker="o", markersize=6,
        label="MMD² — drift detected"))
    if threshold is not None:
        handles.append(matplotlib.lines.Line2D(
            [0], [0], color="black", linestyle="--", linewidth=1.5,
            label=f"Calibration threshold (p95 = {threshold:.4f})"))
    handles.append(matplotlib.lines.Line2D(
        [0], [0], color="#9467bd", linestyle="-.", linewidth=1.5,
        marker="s", markersize=4, label="Mean EH"))
    handles += phase_legend_patches(spans)
    ax.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.90)

    fig.tight_layout()
    out = FIGURES_DIR / "drift_detection_timing.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ── figure 5: adaptive benefit vs static per epoch ────────────────────────────

def fig5_ablation(gradual):
    """
    Show per-epoch recall gap (adaptive − static) and mark repair epochs.

    The naive before/after comparison is confounded under gradual drift: "after"
    epochs have higher t than "before" epochs, so drift alone depresses recall
    regardless of repairs. The right question is how much better adaptive is
    than static AT THE SAME EPOCH — eliminating the drift-increase confound.
    """
    ef32 = gradual[gradual["ef_search"] == PRIMARY_EF]
    adp = ef32[ef32["method"] == "adaptive"].sort_values("epoch_idx").set_index("epoch_idx")
    sta = ef32[ef32["method"] == "static_hnsw"].sort_values("epoch_idx").set_index("epoch_idx")

    common = adp.index.intersection(sta.index)
    if len(common) == 0:
        print("  WARNING: no common epochs between adaptive and static — skipping Figure 5.")
        return

    epochs = np.array(sorted(common))
    gap    = adp.loc[epochs, "recall_at_k"].values - sta.loc[epochs, "recall_at_k"].values
    repair_epochs = adp[adp["repair_happened"] == True].index.tolist()

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Adaptation Benefit: Adaptive vs Static  (Gradual drift, ef=32)",
                 fontsize=12, fontweight="bold")

    # ── left: per-epoch recall gap ──
    ax = axes[0]
    bar_colours = [COLOURS["adaptive"] if g >= 0 else COLOURS["static_hnsw"] for g in gap]
    ax.bar(epochs, gap, color=bar_colours, alpha=0.85, zorder=3)
    ax.axhline(0, color="black", linewidth=0.9, zorder=4)
    for ep in repair_epochs:
        ax.axvline(ep, color=COLOURS["adaptive"], linewidth=0.6, alpha=0.35, zorder=2)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Δ Recall@10  (adaptive − static)", fontsize=11)
    ax.set_title("Per-Epoch Recall Gain from Adaptation", fontsize=11)
    ax.set_xlim(-0.5, 24.5)
    ax.set_xticks(range(0, 25, 2))
    pos = int((gap >= 0).sum())
    ax.text(0.02, 0.98,
            f"{pos}/{len(gap)} epochs: adaptive ≥ static\n"
            f"Mean gap (post-drift): {gap[epochs >= 20].mean():+.3f}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    repair_patch = matplotlib.lines.Line2D(
        [0], [0], color=COLOURS["adaptive"], linewidth=1.5, alpha=0.5,
        label="Repair event")
    ax.legend(handles=[repair_patch], fontsize=9)

    # ── right: cumulative recall — both methods on the same axis ──
    ax2 = axes[1]
    for method, lbl in [("static_hnsw", "Static HNSW"), ("adaptive", "Adaptive (ours)")]:
        m = ef32[ef32["method"] == method].sort_values("epoch_idx")
        ax2.plot(m["epoch_idx"], m["recall_at_k"],
                 color=COLOURS[method], label=lbl, linewidth=2.0,
                 marker="o", markersize=4, zorder=3)
    # shade where adaptive is ahead
    adp_r = adp.loc[epochs, "recall_at_k"].values
    sta_r = sta.loc[epochs, "recall_at_k"].values
    ax2.fill_between(epochs, sta_r, adp_r,
                     where=(adp_r >= sta_r), alpha=0.15,
                     color=COLOURS["adaptive"], label="Adaptive advantage")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Recall@10", fontsize=11)
    ax2.set_title("Recall Trajectories", fontsize=11)
    ax2.set_xlim(-0.5, 24.5)
    ax2.set_xticks(range(0, 25, 2))
    ax2.legend(fontsize=9)

    fig.tight_layout()
    out = FIGURES_DIR / "ablation_repair_events.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ── figure 6: recall vs epoch for ef=64 and ef=128 ───────────────────────────

def fig6_recall_by_ef(gradual, sudden, config):
    extra_efs = [64, 128]
    scenarios = [("Gradual", gradual, config["drift"]["schedule_gradual"]),
                 ("Sudden",  sudden,  None)]

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(len(extra_efs), len(scenarios),
                             figsize=(12, 4.5 * len(extra_efs)),
                             sharex=True)
    fig.suptitle("Recall@10 vs Epoch — ef=64 and ef=128", fontsize=13, fontweight="bold")

    schedule_gradual = config["drift"]["schedule_gradual"]
    spans_gradual = phase_spans(schedule_gradual)
    drift_jump = next(i for i, t in enumerate(config["drift"]["schedule_sudden"]) if t == 1.0)

    for row, ef in enumerate(extra_efs):
        for col, (scenario_name, df, _) in enumerate(scenarios):
            ax = axes[row][col]
            ef_df = df[df["ef_search"] == ef]

            # y range: just below the minimum recall across all methods
            all_recall = ef_df["recall_at_k"].dropna()
            ymin = max(0.0, all_recall.min() - 0.03)
            ymax = 1.01

            if scenario_name == "Gradual":
                shade_phases(ax, spans_gradual, ymin, ymax)
            else:
                ax.axvspan(drift_jump - 0.5, 24.5, color="#e6550d", alpha=0.20, zorder=0)

            for method in METHODS:
                m = ef_df[ef_df["method"] == method].sort_values("epoch_idx")
                ax.plot(m["epoch_idx"], m["recall_at_k"],
                        color=COLOURS[method], label=LABELS[method],
                        linewidth=2.0, marker="o", markersize=4, zorder=3)

            if scenario_name == "Sudden":
                ax.axvline(drift_jump - 0.5, color="black", linestyle="--",
                           linewidth=1.5, alpha=0.7, zorder=4,
                           label=f"Sudden drift (epoch {drift_jump})")

            ax.set_xlim(-0.5, 24.5)
            ax.set_ylim(ymin, ymax)
            ax.set_xticks(range(0, 25, 2))
            ax.set_title(f"{scenario_name} drift  (ef={ef})", fontsize=11)
            if col == 0:
                ax.set_ylabel("Recall@10", fontsize=11)
            if row == len(extra_efs) - 1:
                ax.set_xlabel("Epoch", fontsize=11)
            ax.legend(fontsize=8, loc="lower left", framealpha=0.85)

    fig.tight_layout()
    out = FIGURES_DIR / "recall_by_ef.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ── LaTeX table ───────────────────────────────────────────────────────────────

def print_latex_table(gradual, sudden):
    rows = []
    for scenario, df in [("Gradual", gradual), ("Sudden", sudden)]:
        ef32 = df[df["ef_search"] == PRIMARY_EF]
        static_post = ef32[(ef32["method"] == "static_hnsw") &
                           (ef32["epoch_idx"] >= 20)]["recall_at_k"].mean()
        for method in METHODS:
            m = ef32[ef32["method"] == method]
            pre  = m[m["epoch_idx"] <= 4]["recall_at_k"].mean()
            post = m[m["epoch_idx"] >= 20]["recall_at_k"].mean()
            drop = pre - post
            rel  = (post - static_post) / static_post * 100 if method != "static_hnsw" else float("nan")
            pre_qps  = m[m["epoch_idx"] <= 4]["queries_per_second"].mean()
            post_qps = m[m["epoch_idx"] >= 20]["queries_per_second"].mean()
            rows.append(dict(
                scenario=scenario,
                method=LABELS[method],
                pre=pre,
                post=post,
                drop=drop,
                rel=rel,
                pre_qps=pre_qps,
                post_qps=post_qps,
            ))

    print()
    print("% ── LaTeX results table ─────────────────────────────────────────────")
    print(r"\begin{table}[t]")
    print(r"  \centering")
    print(r"  \caption{Recall@10 and throughput under gradual and sudden rotation drift")
    print(r"           (SIFT-5M, ef$_\text{search}$=32, $k$=10).}")
    print(r"  \label{tab:main_results}")
    print(r"  \begin{tabular}{llcccccc}")
    print(r"    \toprule")
    print(r"    Drift & Method & Pre R@10 & Post R@10 & Drop & "
          r"$\Delta$ vs static & Pre QPS & Post QPS \\")
    print(r"    \midrule")
    prev_scenario = None
    for r in rows:
        if r["scenario"] != prev_scenario:
            if prev_scenario is not None:
                print(r"    \midrule")
            prev_scenario = r["scenario"]
        rel_str = (f"{r['rel']:+.1f}\\%" if not np.isnan(r['rel']) else "--")
        pre_qps  = f"{r['pre_qps']:,.0f}"  if not np.isnan(r['pre_qps'])  else "--"
        post_qps = f"{r['post_qps']:,.0f}" if not np.isnan(r['post_qps']) else "--"
        print(
            f"    {r['scenario']} & {r['method']} & "
            f"{r['pre']:.3f} & {r['post']:.3f} & "
            f"{r['drop']:.3f} & {rel_str} & "
            f"{pre_qps} & {post_qps} \\\\"
        )
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import matplotlib.lines  # needed for Line2D in legend helpers

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    gradual, sudden, config = load_data()

    print("Generating figures …")
    fig1_gradual(gradual, config)
    fig2_sudden(sudden, config)
    fig3_latency(gradual, sudden)
    fig4_detection(gradual, config)
    fig5_ablation(gradual)
    fig6_recall_by_ef(gradual, sudden, config)
    print_latex_table(gradual, sudden)
    print(f"\nAll figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    # matplotlib.lines needed in helper functions before main() runs
    import matplotlib.lines
    main()
