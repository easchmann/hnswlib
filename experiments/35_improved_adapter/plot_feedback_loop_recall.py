"""Positive-feedback-loop diagnostic plot for Exp35 SIFT1M — recall-based version (v3).

Matched, same-query comparison (no population-composition confound, same
design as the v2 distance diagnostic): for the exact same hard-query indices
each epoch,
  - recall_adaptive_hardonly: recall@k of the conjugate-augmented search.
  - recall_static_hardonly:   recall@k of a fresh raw, non-adaptive search.

The feedback-loop signature is a GAP that starts near zero and GROWS over
epochs, tracking edge accumulation — not a gap that appears immediately and
then holds flat (that would indicate a one-shot benefit, not a compounding
loop) and not a gap that stays at zero (no measurable benefit at all).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

COLOR_ADAPTIVE = "darkorange"
COLOR_STATIC = "steelblue"
COLOR_GAP = "crimson"
COLOR_EDGES = "grey"

DIAG_COLS = [
    "epoch", "recall_adaptive_hardonly", "recall_static_hardonly", "n_hard_queries", "edge_count",
]


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def plot_schedule(df, schedule, cfg, plots_dir):
    ef_values = sorted(df["ef"].unique())
    transitions = cfg.get(f"drift_transition_epochs_{schedule}", [])

    diag = df[df["ef"] == ef_values[0]].sort_values("epoch")[DIAG_COLS].copy()
    diag["gap"] = diag["recall_adaptive_hardonly"] - diag["recall_static_hardonly"]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    ax_top.plot(diag["epoch"], diag["recall_adaptive_hardonly"], color=COLOR_ADAPTIVE,
                linewidth=1.5, marker="o", markersize=3,
                label="adaptive (conjugate-augmented), hard queries only")
    ax_top.plot(diag["epoch"], diag["recall_static_hardonly"], color=COLOR_STATIC,
                linewidth=1.5, linestyle="--", marker="s", markersize=3,
                label="raw static search, same hard queries")
    ax_top.set_ylabel("Recall@k")
    ax_top.set_ylim(0, 1.05)
    ax_top.set_title(f"SIFT1M rotation drift — {schedule} — matched-recall feedback-loop diagnostic")
    ax_top.legend(fontsize=8)

    ax_bot.plot(diag["epoch"], diag["gap"], color=COLOR_GAP, linewidth=1.5,
                marker="o", markersize=3, label="recall gap (adaptive − static, hard-only)")
    ax_bot.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax_bot.set_xlabel("Epoch")
    ax_bot.set_ylabel("Recall gap", color=COLOR_GAP)
    ax_bot.tick_params(axis="y", labelcolor=COLOR_GAP)

    ax_edges = ax_bot.twinx()
    ax_edges.plot(diag["epoch"], diag["edge_count"], color=COLOR_EDGES, linewidth=1.0,
                  linestyle=":", alpha=0.7, label="cumulative conjugate edges")
    ax_edges.set_ylabel("Cumulative edges", color=COLOR_EDGES)
    ax_edges.tick_params(axis="y", labelcolor=COLOR_EDGES)

    lines1, labels1 = ax_bot.get_legend_handles_labels()
    lines2, labels2 = ax_edges.get_legend_handles_labels()
    ax_bot.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")

    for ax in (ax_top, ax_bot):
        for t in transitions:
            ax.axvline(t, color="black", linestyle=":", linewidth=0.8, alpha=0.4)

    fig.tight_layout()
    out = plots_dir / f"{schedule}_feedback_loop_recall.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")

    # Console summary: does the gap grow (not just exist)?
    valid = diag.dropna(subset=["gap"])
    if len(valid) > 2:
        first_gap = valid.iloc[0]["gap"]
        last_gap = valid.iloc[-1]["gap"]
        n = len(valid)
        early_mean = valid.iloc[: n // 3]["gap"].mean()
        late_mean = valid.iloc[-n // 3 :]["gap"].mean()
        print(f"  [{schedule}] recall gap: epoch {int(valid.iloc[0]['epoch'])}={first_gap:.4f} "
              f"-> epoch {int(valid.iloc[-1]['epoch'])}={last_gap:.4f}")
        print(f"  [{schedule}] early-third mean gap={early_mean:.4f}  late-third mean gap={late_mean:.4f}  "
              f"({'growing' if late_mean > early_mean else 'NOT growing'})")
        print(f"  [{schedule}] mean n_hard_queries/epoch={valid['n_hard_queries'].mean():.1f}")
    else:
        print(f"  [{schedule}] not enough epochs with hard queries to assess trend")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "config_sift1m_feedback_diag_recall.yaml"
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
        if df.empty or "recall_adaptive_hardonly" not in df.columns:
            print(f"No adaptive_improved / recall diagnostic columns in {path}, skipping {schedule}")
            continue
        plot_schedule(df, schedule, cfg, plots_dir)


if __name__ == "__main__":
    main()
