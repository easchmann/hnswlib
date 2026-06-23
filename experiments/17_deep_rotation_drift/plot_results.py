"""Experiment 17 figures: Deep-image-96 rotation drift."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = _SCRIPT_DIR / "config.yaml"
DPI = 300
PRIMARY_EF = 32

CONDITION_COLORS = {
    "static": "grey",
    "periodic_rebuild": "darkorange",
    "adaptive_mconj48": "steelblue",
}
CONDITION_LABELS = {
    "static": "Static HNSW",
    "periodic_rebuild": "Periodic rebuild (every 5 ep.)",
    "adaptive_mconj48": r"Adaptive ($M_{\mathrm{conj}}=48$)",
}


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _results_dir(cfg):
    path = ROOT / cfg["output"]["results_dir"]
    if path.exists():
        return path
    return _SCRIPT_DIR


def _load_schedule_condition(results_dir, schedule, condition_name):
    p = results_dir / f"{schedule}_{condition_name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    return pd.read_csv(p)


def _load_all(cfg, results_dir):
    data = {}
    for schedule in ("gradual", "sudden"):
        for cond in cfg["conditions"]:
            name = cond["name"]
            try:
                df = _load_schedule_condition(results_dir, schedule, name)
                data[(schedule, name)] = df
            except FileNotFoundError as e:
                print(f"  WARNING: {e}")
    return data


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


def _plot_single(data, cfg, schedule, title, figures_dir, out_name):
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(10, 5))

    drift_schedule = cfg["drift"][f"schedule_{schedule}"]
    _shade_drift(ax, drift_schedule)

    for cond in cfg["conditions"]:
        name = cond["name"]
        key = (schedule, name)
        if key not in data:
            continue
        df = data[key]
        sub = df[df["ef_search"] == PRIMARY_EF].sort_values("epoch_idx")
        ax.plot(
            sub["epoch_idx"],
            sub["recall_at_k"],
            color=CONDITION_COLORS.get(name, "black"),
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            label=CONDITION_LABELS.get(name, name),
            zorder=3,
        )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Recall@10", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(-0.5, 24.5)
    ax.set_xticks(range(0, 25, 2))
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.85)

    fig.tight_layout()
    out = figures_dir / out_name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    cfg = _load_config()
    results_dir = _results_dir(cfg)
    figures_dir = _SCRIPT_DIR / "plots"
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = _load_all(cfg, results_dir)
    if not data:
        print("No results found. Run run_eval.py first.")
        return

    _plot_single(
        data, cfg,
        schedule="gradual",
        title="Deep-image-96 rotation drift (gradual) \u2014 recall@10 vs epoch",
        figures_dir=figures_dir,
        out_name="recall_vs_epoch_gradual.pdf",
    )
    _plot_single(
        data, cfg,
        schedule="sudden",
        title="Deep-image-96 rotation drift (sudden) \u2014 recall@10 vs epoch",
        figures_dir=figures_dir,
        out_name="recall_vs_epoch_sudden.pdf",
    )


if __name__ == "__main__":
    main()
