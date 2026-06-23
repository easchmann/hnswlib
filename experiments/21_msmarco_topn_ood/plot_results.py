"""Experiment 21: compare top-N OOD pool sizes vs exp20 strict cluster."""

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

EXP20_DIR = ROOT / "experiments" / "20_msmarco_focused_ood"
EXP21_DIR = ROOT / "experiments" / "21_msmarco_topn_ood"
PLOTS_DIR = Path(__file__).parent / "plots"
CONFIG_PATH = Path(__file__).parent / "config.yaml"

# Colors per pool variant (exp20 strict cluster + exp21 top-N)
COLORS = {
    "exp20_strict167": "#2c7bb6",
    167: "#74add1",
    500: "#f46d43",
    1000: "#1a9641",
}
LABELS = {
    "exp20_strict167": "strict cluster (exp20, n=167)",
    167: "top-167 (centroid proximity)",
    500: "top-500",
    1000: "top-1000",
}


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _load_exp20(schedule, condition):
    p = EXP20_DIR / f"{schedule}_{condition}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Exp20 results not found: {p}")
    return pd.read_csv(p)


def _load_exp21(top_n, schedule, condition):
    p = EXP21_DIR / f"top{top_n}_{schedule}_{condition}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Exp21 results not found: {p}")
    return pd.read_csv(p)


def _t1_epochs(schedule_tag):
    """Return epoch indices where t=1.0 for each schedule."""
    if schedule_tag == "gradual":
        return list(range(18, 25))
    else:
        return list(range(13, 25))


def _shade_drift_epochs(ax, cfg, schedule_tag, alpha=0.08):
    t_vals = cfg["drift"][f"schedule_{schedule_tag}"]
    for i, t in enumerate(t_vals):
        if t > 0:
            ax.axvspan(i - 0.5, i + 0.5, color="orange", alpha=alpha, linewidth=0)


def figure1_recall_vs_epoch(cfg, schedule_tag):
    """Recall@10 vs epoch: exp20 strict + all exp21 top-N, adaptive + static."""
    primary_ef = cfg["eval"]["primary_ef"]
    top_n_values = cfg["focused_ood"]["top_n_values"]

    fig, ax = plt.subplots(figsize=(11, 5))
    _shade_drift_epochs(ax, cfg, schedule_tag)

    # exp20 strict cluster
    df_s = _load_exp20(schedule_tag, "static")
    df_a = _load_exp20(schedule_tag, "adaptive_mconj48")
    df_s = df_s[df_s["ef_search"] == primary_ef]
    df_a = df_a[df_a["ef_search"] == primary_ef]
    c = COLORS["exp20_strict167"]
    ax.plot(df_s["epoch_idx"], df_s["recall_at_k"], color=c, lw=1, linestyle="--", alpha=0.6)
    ax.plot(df_a["epoch_idx"], df_a["recall_at_k"], color=c, lw=2,
            label=LABELS["exp20_strict167"])

    # exp21 top-N variants
    for top_n in top_n_values:
        df_s = _load_exp21(top_n, schedule_tag, "static")
        df_a = _load_exp21(top_n, schedule_tag, "adaptive_mconj48")
        df_s = df_s[df_s["ef_search"] == primary_ef]
        df_a = df_a[df_a["ef_search"] == primary_ef]
        c = COLORS[top_n]
        ax.plot(df_s["epoch_idx"], df_s["recall_at_k"], color=c, lw=1, linestyle="--", alpha=0.6)
        ax.plot(df_a["epoch_idx"], df_a["recall_at_k"], color=c, lw=2, label=LABELS[top_n])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recall@10")
    title_tag = "gradual" if schedule_tag == "gradual" else "sudden"
    ax.set_title(f"MS MARCO focused OOD ({title_tag}) — recall@10 vs epoch (ef={primary_ef})")
    ax.legend(fontsize=8)

    # dashed-line legend note
    ax.plot([], [], color="gray", lw=1, linestyle="--", label="static baselines (dashed)")
    ax.legend(fontsize=8)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = PLOTS_DIR / f"figure{'1' if schedule_tag == 'gradual' else '2'}_{schedule_tag}_recall.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def figure3_recall_at_t1_vs_topn(cfg):
    """Adaptive recall@10 at t=1 vs pool size — the scaling plot."""
    primary_ef = cfg["eval"]["primary_ef"]
    top_n_values = cfg["focused_ood"]["top_n_values"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, schedule_tag in zip(axes, ["gradual", "sudden"]):
        t1_epochs = _t1_epochs(schedule_tag)

        # exp20 strict cluster
        df_a = _load_exp20(schedule_tag, "adaptive_mconj48")
        df_s = _load_exp20(schedule_tag, "static")
        df_a = df_a[df_a["ef_search"] == primary_ef]
        df_s = df_s[df_s["ef_search"] == primary_ef]

        adaptive_t1 = df_a[df_a["epoch_idx"].isin(t1_epochs)]["recall_at_k"]
        static_t1 = df_s[df_s["epoch_idx"].isin(t1_epochs)]["recall_at_k"]
        gain_exp20 = float((adaptive_t1 - static_t1.values).mean())
        recall_exp20 = float(adaptive_t1.mean())

        xs = [167]  # nominal size (actual strict cluster = 167)
        ys_recall = [recall_exp20]
        ys_gain = [gain_exp20]
        labels_x = ["strict\n167"]

        for top_n in top_n_values:
            df_a = _load_exp21(top_n, schedule_tag, "adaptive_mconj48")
            df_s = _load_exp21(top_n, schedule_tag, "static")
            df_a = df_a[df_a["ef_search"] == primary_ef]
            df_s = df_s[df_s["ef_search"] == primary_ef]
            adaptive_t1 = df_a[df_a["epoch_idx"].isin(t1_epochs)]["recall_at_k"]
            static_t1 = df_s[df_s["epoch_idx"].isin(t1_epochs)]["recall_at_k"]
            ys_recall.append(float(adaptive_t1.mean()))
            ys_gain.append(float((adaptive_t1 - static_t1.values).mean()))
            xs.append(top_n)
            labels_x.append(f"top-{top_n}")

        # offset exp20 point slightly so it doesn't overlap top-167
        x_plot = [x + (15 if i == 0 else 0) for i, x in enumerate(xs)]

        ax.plot(x_plot[1:], ys_recall[1:], "o-", color="darkorange", label="top-N (exp21)")
        ax.scatter([x_plot[0]], [ys_recall[0]], marker="*", s=120, color="#2c7bb6",
                   zorder=5, label="strict cluster (exp20)")
        ax2 = ax.twinx()
        ax2.plot(x_plot[1:], ys_gain[1:], "s--", color="steelblue", alpha=0.7,
                 label="adaptive gain (pp)")
        ax2.scatter([x_plot[0]], [ys_gain[0]], marker="*", s=120, color="#2c7bb6",
                    alpha=0.7, zorder=5)
        ax2.set_ylabel("Adaptive gain over static (pp)", color="steelblue")
        ax2.tick_params(axis="y", labelcolor="steelblue")

        ax.set_xticks(x_plot)
        ax.set_xticklabels(labels_x)
        ax.set_xlabel("OOD pool size")
        ax.set_ylabel("Mean adaptive recall@10 at t=1", color="darkorange")
        ax.tick_params(axis="y", labelcolor="darkorange")
        ax.set_title(f"{schedule_tag.capitalize()} schedule")

        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="lower right")

    fig.suptitle("Recall@10 at t=1 vs OOD pool size (ef=32)")
    fig.tight_layout()
    out = PLOTS_DIR / "figure3_recall_vs_topn.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def figure4_edge_accumulation(cfg):
    """n_conjugate_edges vs epoch, gradual, all N variants."""
    primary_ef = cfg["eval"]["primary_ef"]
    top_n_values = cfg["focused_ood"]["top_n_values"]

    fig, ax = plt.subplots(figsize=(10, 5))

    # exp20
    df = _load_exp20("gradual", "adaptive_mconj48")
    df = df[df["ef_search"] == primary_ef]
    ax.plot(df["epoch_idx"], df["n_conjugate_edges"], color=COLORS["exp20_strict167"],
            lw=2, label=LABELS["exp20_strict167"])

    for top_n in top_n_values:
        df = _load_exp21(top_n, "gradual", "adaptive_mconj48")
        df = df[df["ef_search"] == primary_ef]
        ax.plot(df["epoch_idx"], df["n_conjugate_edges"], color=COLORS[top_n],
                lw=2, label=LABELS[top_n])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("n_conjugate_edges")
    ax.set_title("Conjugate edge accumulation vs OOD pool size (gradual)")
    ax.legend(fontsize=8)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = PLOTS_DIR / "figure4_edge_accumulation.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    cfg = _load_config()
    figure1_recall_vs_epoch(cfg, "gradual")
    figure1_recall_vs_epoch(cfg, "sudden")
    figure3_recall_at_t1_vs_topn(cfg)
    figure4_edge_accumulation(cfg)


if __name__ == "__main__":
    main()
