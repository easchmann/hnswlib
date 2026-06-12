"""Experiment 05 figures: M_conj sweep and cooldown interaction."""

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


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _results_dir(cfg):
    path = ROOT / cfg["output"]["results_dir"]
    if path.exists():
        return path
    return _SCRIPT_DIR


def _load_condition(results_dir, name):
    path = results_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    return pd.read_csv(path)


def _load_all(cfg):
    results_dir = _results_dir(cfg)
    frames = []
    for cond in cfg["conditions"]:
        df = _load_condition(results_dir, cond["name"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _load_ablation_baseline(cfg):
    path = ROOT / cfg["output"]["ablation_two_hop_path"]
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df[df["ef_search"] == PRIMARY_EF].sort_values("epoch_idx")


def _load_exp04_condition(cfg, name):
    """Try results/04_tuning/ then experiments/04_tuning/ for a named CSV."""
    candidates = [
        ROOT / cfg["output"]["exp04_results_dir"] / f"{name}.csv",
        ROOT / "experiments" / "04_tuning" / f"{name}.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    return None


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


def _mean_post_drift(df_ef32, condition, lo=20, hi=24):
    sub = df_ef32[(df_ef32["condition"] == condition) & (df_ef32["epoch_idx"] >= lo)]
    return sub["recall_at_k"].mean()


def plot_figure1(ef32, ablation_baseline, schedule, figures_dir):
    """Two subplots: M_conj sweep (left) and cooldown at M_conj=16 (right)."""
    sns.set_theme(style="whitegrid", font_scale=0.95)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    mconj_sweep = ["mconj_8_cd0", "mconj_16_cd0", "mconj_20_cd0", "mconj_24_cd0"]
    mconj_labels = {
        "mconj_8_cd0": "M_conj=8",
        "mconj_16_cd0": "M_conj=16",
        "mconj_20_cd0": "M_conj=20",
        "mconj_24_cd0": "M_conj=24",
    }

    cooldown_sweep = ["mconj_16_cd0", "mconj_16_cd2", "mconj_16_cd3", "mconj_16_cd5"]
    cooldown_labels = {
        "mconj_16_cd0": "cooldown=0",
        "mconj_16_cd2": "cooldown=2",
        "mconj_16_cd3": "cooldown=3",
        "mconj_16_cd5": "cooldown=5",
    }

    blues = sns.color_palette("Blues", len(mconj_sweep) + 2)[2:]
    greens = sns.color_palette("Greens", len(cooldown_sweep) + 2)[2:]

    for ax, conditions, labels, palette, title in [
        (ax_left,  mconj_sweep,    mconj_labels,    blues,  "M_conj sweep (cooldown=0)"),
        (ax_right, cooldown_sweep, cooldown_labels, greens, "repair_cooldown sweep (M_conj=16)"),
    ]:
        if schedule:
            _shade_drift(ax, schedule)

        if ablation_baseline is not None:
            ax.plot(
                ablation_baseline["epoch_idx"],
                ablation_baseline["recall_at_k"],
                color="grey",
                linestyle="--",
                linewidth=1.5,
                label="two_hop baseline (ablation)",
                zorder=2,
                alpha=0.8,
            )

        for color, cname in zip(palette, conditions):
            sub = ef32[ef32["condition"] == cname].sort_values("epoch_idx")
            if sub.empty:
                continue
            ax.plot(
                sub["epoch_idx"],
                sub["recall_at_k"],
                color=color,
                linewidth=1.8,
                marker="o",
                markersize=3.5,
                label=labels.get(cname, cname),
                zorder=3,
            )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Recall@10", fontsize=10)
        ax.set_xlim(-0.5, 24.5)
        ax.set_xticks(range(0, 25, 4))
        ax.legend(fontsize=8, loc="lower left", framealpha=0.85)

    fig.suptitle(
        f"Experiment 05: M_conj × cooldown sweep  (gradual drift, ef={PRIMARY_EF})",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = figures_dir / "mconj_cooldown_sweeps.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_figure2(ef32, exp04_best_guess, ablation_baseline, schedule, figures_dir):
    """Best-of comparison across key conditions plus exp04 best_guess reference."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(11, 5))

    featured = [
        "mconj_8_cd0",
        "mconj_16_cd0",
        "mconj_16_cd3",
        "mconj_20_cd3",
        "mconj_24_cd3",
    ]
    featured_labels = {
        "mconj_8_cd0": "mconj_8_cd0 (ablation baseline)",
        "mconj_16_cd0": "mconj_16_cd0",
        "mconj_16_cd3": "mconj_16_cd3",
        "mconj_20_cd3": "mconj_20_cd3",
        "mconj_24_cd3": "mconj_24_cd3",
    }

    palette = sns.color_palette("tab10", len(featured) + 1)

    if schedule:
        _shade_drift(ax, schedule)

    for color, cname in zip(palette, featured):
        sub = ef32[ef32["condition"] == cname].sort_values("epoch_idx")
        if sub.empty:
            continue
        ax.plot(
            sub["epoch_idx"],
            sub["recall_at_k"],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=4,
            label=featured_labels.get(cname, cname),
            zorder=3,
        )

    if exp04_best_guess is not None:
        sub = exp04_best_guess[exp04_best_guess["ef_search"] == PRIMARY_EF].sort_values("epoch_idx")
        if not sub.empty:
            ax.plot(
                sub["epoch_idx"],
                sub["recall_at_k"],
                color=palette[len(featured)],
                linewidth=2.0,
                marker="s",
                markersize=4,
                linestyle="--",
                label="best_guess (exp04)",
                zorder=3,
            )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Recall@10", fontsize=11)
    ax.set_title(
        f"Best-of comparison  (gradual drift, ef={PRIMARY_EF})",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlim(-0.5, 24.5)
    ax.set_xticks(range(0, 25, 2))
    ax.legend(fontsize=9, loc="lower left", framealpha=0.85)

    fig.tight_layout()
    out = figures_dir / "best_of_comparison.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def print_summary_table(ef32, cfg):
    conditions = [c["name"] for c in cfg["conditions"]]
    header = (
        f"{'Condition':<22}  {'Post-drift R@10':>15}  "
        f"{'Recall drop':>11}  {'Conj edges ep24':>15}  {'Repair events':>13}"
    )
    print("\n" + header)
    print("-" * len(header))
    for cname in conditions:
        sub = ef32[ef32["condition"] == cname]
        if sub.empty:
            continue
        pre = sub[sub["epoch_idx"] <= 4]["recall_at_k"].mean()
        post = sub[sub["epoch_idx"] >= 20]["recall_at_k"].mean()
        ep24 = sub[sub["epoch_idx"] == 24]["n_conjugate_edges"].values
        edges_val = int(ep24[0]) if len(ep24) > 0 else -1
        repair_count = int(sub["repair_happened"].sum()) if "repair_happened" in sub.columns else -1
        drop = pre - post
        print(
            f"{cname:<22}  {post:>15.4f}  {drop:>11.4f}  "
            f"{edges_val:>15d}  {repair_count:>13d}"
        )
    print()


def main():
    cfg = _load_config()
    results_dir = _results_dir(cfg)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    schedule = cfg.get("drift", {}).get("schedule_gradual")
    all_data = _load_all(cfg)
    ef32 = all_data[all_data["ef_search"] == PRIMARY_EF].copy()
    ablation_baseline = _load_ablation_baseline(cfg)
    exp04_best_guess = _load_exp04_condition(cfg, "best_guess")

    plot_figure1(ef32, ablation_baseline, schedule, figures_dir)
    plot_figure2(ef32, exp04_best_guess, ablation_baseline, schedule, figures_dir)
    print_summary_table(ef32, cfg)


if __name__ == "__main__":
    main()
