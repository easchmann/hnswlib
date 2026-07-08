"""Positive-feedback-loop diagnostic plot for Exp35 SIFT1M (fix #4).

Direct, mechanistic test of the feedback-loop hypothesis: does the mean
distance from repair source nodes to the query that triggered them
(mean_source_query_dist) fall over epochs and converge toward the distance
achieved by a raw, non-adaptive HNSW top-k search?

Two references are plotted:
  - mean_static_topk_dist: raw top-k distance averaged over ALL queries in
    the epoch (easy and hard together). Under rotation drift this is
    confounded: as more of the population becomes "hard", this reference
    itself drifts toward the hard-query distribution, which can make the
    gap close even if the conjugate graph contributes nothing.
  - mean_hardonly_static_dist: raw top-k distance averaged over ONLY the
    same hard queries used for mean_source_query_dist. This is the
    deconfounded, apples-to-apples reference; convergence toward this one
    is the clean test of the conjugate graph's own contribution.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

COLOR_RECALL = "darkorange"
COLOR_SOURCE = "crimson"
COLOR_STATIC_REF = "grey"
COLOR_HARDONLY_REF = "steelblue"

DIAG_COLS = [
    "epoch", "mean_source_query_dist", "mean_static_topk_dist", "mean_hardonly_static_dist",
]


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def plot_schedule(df, schedule, cfg, plots_dir):
    ef_values = sorted(df["ef"].unique())
    primary_ef = cfg["primary_ef"]
    transitions = cfg.get(f"drift_transition_epochs_{schedule}", [])
    has_hardonly = "mean_hardonly_static_dist" in df.columns

    cols = DIAG_COLS if has_hardonly else DIAG_COLS[:-1]
    diag = df[df["ef"] == ef_values[0]].sort_values("epoch")[cols]
    recall = df[df["ef"] == primary_ef].sort_values("epoch")[["epoch", "recall"]]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    ax_top.plot(recall["epoch"], recall["recall"], color=COLOR_RECALL, linewidth=1.5,
                label=f"adaptive_improved recall@10 (ef={primary_ef})")
    ax_top.set_ylabel("Recall@10")
    ax_top.set_ylim(0, 1.05)
    ax_top.set_title(f"SIFT1M rotation drift — {schedule} — feedback-loop diagnostic")
    ax_top.legend(fontsize=8)

    ax_bot.plot(diag["epoch"], diag["mean_source_query_dist"], color=COLOR_SOURCE,
                linewidth=1.5, marker="o", markersize=3,
                label="repair source nodes → query")
    ax_bot.plot(diag["epoch"], diag["mean_static_topk_dist"], color=COLOR_STATIC_REF,
                linewidth=1.5, linestyle="--",
                label="static top-k → query (all-queries reference, confounded)")
    if has_hardonly:
        ax_bot.plot(diag["epoch"], diag["mean_hardonly_static_dist"], color=COLOR_HARDONLY_REF,
                    linewidth=1.5, linestyle=":", marker="s", markersize=3,
                    label="static top-k → query (hard-query-only reference, deconfounded)")
    ax_bot.set_xlabel("Epoch")
    ax_bot.set_ylabel("Mean L2 distance")
    ax_bot.legend(fontsize=7)

    for ax in (ax_top, ax_bot):
        for t in transitions:
            ax.axvline(t, color="black", linestyle=":", linewidth=0.8, alpha=0.4)

    fig.tight_layout()
    out = plots_dir / f"{schedule}_feedback_loop.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")

    # Console summary: does the gap close against each reference?
    if len(diag) > 1:
        first, last = diag.iloc[0], diag.iloc[-1]
        gap_first = first["mean_source_query_dist"] - first["mean_static_topk_dist"]
        gap_last = last["mean_source_query_dist"] - last["mean_static_topk_dist"]
        print(f"  [{schedule}] vs ALL-QUERIES reference (confounded): epoch {int(first['epoch'])}="
              f"{gap_first:.4f} -> epoch {int(last['epoch'])}={gap_last:.4f} "
              f"({'closing' if gap_last < gap_first else 'NOT closing'})")
        if has_hardonly and pd.notna(first["mean_hardonly_static_dist"]) and pd.notna(last["mean_hardonly_static_dist"]):
            hgap_first = first["mean_source_query_dist"] - first["mean_hardonly_static_dist"]
            hgap_last = last["mean_source_query_dist"] - last["mean_hardonly_static_dist"]
            print(f"  [{schedule}] vs HARD-ONLY reference (deconfounded): epoch {int(first['epoch'])}="
                  f"{hgap_first:.4f} -> epoch {int(last['epoch'])}={hgap_last:.4f} "
                  f"({'closing' if hgap_last < hgap_first else 'NOT closing'})")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "config_sift1m_feedback_diag.yaml"
    cfg = _load_config(config_path)
    plots_dir = ROOT / cfg["plots_dir"]
    plots_dir.mkdir(parents=True, exist_ok=True)

    for schedule, results_key in [("gradual", "results_gradual"), ("sudden", "results_sudden")]:
        path = ROOT / cfg[results_key]
        if not path.exists():
            print(f"Missing {path}, skipping {schedule}")
            continue
        df = pd.read_csv(path)
        df = df[df["condition"] == "adaptive_improved"]
        if df.empty or "mean_source_query_dist" not in df.columns:
            print(f"No adaptive_improved / diagnostic columns in {path}, skipping {schedule}")
            continue
        plot_schedule(df, schedule, cfg, plots_dir)


if __name__ == "__main__":
    main()
