"""Experiment 06 figures: M_conj ceiling search."""

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

# Prior results to include in Figure 2 bar chart: (source_dir_key, condition_name, M_conj)
PRIOR_CONDITIONS = [
    ("exp04_results_dir", "mconj_12",    12),
    ("exp05_results_dir", "mconj_8_cd0",  8),
    ("exp05_results_dir", "mconj_16_cd0", 16),
    ("exp05_results_dir", "mconj_20_cd0", 20),
    ("exp05_results_dir", "mconj_24_cd0", 24),
]


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _results_dir(cfg):
    path = ROOT / cfg["output"]["results_dir"]
    if path.exists():
        return path
    return _SCRIPT_DIR


def _load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    return pd.read_csv(path)


def _load_all(cfg):
    results_dir = _results_dir(cfg)
    frames = []
    for cond in cfg["conditions"]:
        df = _load_csv(results_dir / f"{cond['name']}.csv")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _load_ablation_baseline(cfg):
    path = ROOT / cfg["output"]["ablation_two_hop_path"]
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df[df["ef_search"] == PRIMARY_EF].sort_values("epoch_idx")


def _mean_post_drift(df, condition, lo=20, hi=24):
    sub = df[(df["condition"] == condition) & (df["epoch_idx"] >= lo)]
    return sub["recall_at_k"].mean()


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


def plot_figure1(ef32, ablation_baseline, schedule, conditions, figures_dir):
    """Recall@10 vs epoch for all M_conj values in this experiment."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(11, 5))

    n = len(conditions)
    blues = sns.color_palette("Blues", n + 2)[2:]

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

    for color, cond in zip(blues, conditions):
        cname = cond["name"]
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
            label=f"M_conj={cond['M_conj']}",
            zorder=3,
        )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Recall@10", fontsize=11)
    ax.set_title(
        f"M_conj ceiling search  (gradual drift, ef={PRIMARY_EF})",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlim(-0.5, 24.5)
    ax.set_xticks(range(0, 25, 2))
    ax.legend(fontsize=9, loc="lower left", framealpha=0.85)

    fig.tight_layout()
    out = figures_dir / "mconj_ceiling_epochs.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _build_full_series(cfg, ef32):
    """Collect (M_conj, post_drift_recall, edges_ep24) across all experiments."""
    results_dir = _results_dir(cfg)
    rows = []

    # Prior experiments
    for dir_key, cname, mconj in PRIOR_CONDITIONS:
        exp_dir = ROOT / cfg["output"].get(dir_key, "")
        # Fall back to script sibling directories if results path missing
        candidates = [exp_dir / f"{cname}.csv"]
        if dir_key == "exp04_results_dir":
            candidates.append(ROOT / "experiments" / "04_tuning" / f"{cname}.csv")
        elif dir_key == "exp05_results_dir":
            candidates.append(ROOT / "experiments" / "05_mconj_cooldown" / f"{cname}.csv")
        df = None
        for p in candidates:
            if p.exists():
                df = pd.read_csv(p)
                break
        if df is None:
            continue
        sub = df[(df["ef_search"] == PRIMARY_EF) & (df["epoch_idx"] >= 20)]
        recall = sub["recall_at_k"].mean()
        ep24 = df[(df["ef_search"] == PRIMARY_EF) & (df["epoch_idx"] == 24)]["n_conjugate_edges"].values
        edges = int(ep24[0]) if len(ep24) > 0 else 0
        rows.append({"M_conj": mconj, "recall": recall, "edges": edges, "source": "prior"})

    # This experiment — use anchor mconj_24 only if no prior entry at 24
    prior_mconj = {r["M_conj"] for r in rows}
    for cond in cfg["conditions"]:
        mconj = cond["M_conj"]
        if mconj in prior_mconj:
            continue
        sub_ef = ef32[(ef32["condition"] == cond["name"]) & (ef32["epoch_idx"] >= 20)]
        recall = sub_ef["recall_at_k"].mean()
        ep24 = ef32[(ef32["condition"] == cond["name"]) & (ef32["epoch_idx"] == 24)]["n_conjugate_edges"].values
        edges = int(ep24[0]) if len(ep24) > 0 else 0
        rows.append({"M_conj": mconj, "recall": recall, "edges": edges, "source": "exp06"})

    # Always add exp06 anchor (mconj_24) as the authoritative rerun
    anchor_cond = next((c for c in cfg["conditions"] if c["name"] == "mconj_24"), None)
    if anchor_cond:
        sub_ef = ef32[(ef32["condition"] == "mconj_24") & (ef32["epoch_idx"] >= 20)]
        if not sub_ef.empty:
            recall = sub_ef["recall_at_k"].mean()
            ep24 = ef32[(ef32["condition"] == "mconj_24") & (ef32["epoch_idx"] == 24)]["n_conjugate_edges"].values
            edges = int(ep24[0]) if len(ep24) > 0 else 0
            # Replace any prior entry at M_conj=24 with the rerun
            rows = [r for r in rows if r["M_conj"] != 24]
            rows.append({"M_conj": 24, "recall": recall, "edges": edges, "source": "exp06"})

    rows.sort(key=lambda r: r["M_conj"])
    return rows


def plot_figure2(cfg, ef32, figures_dir):
    """Bar chart (recall) and scatter (edges) vs M_conj, spanning all experiments."""
    sns.set_theme(style="whitegrid", font_scale=0.95)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))

    series = _build_full_series(cfg, ef32)
    if not series:
        print("No data for Figure 2 — skipping.")
        plt.close(fig)
        return

    mconj_vals = [r["M_conj"] for r in series]
    recalls = [r["recall"] for r in series]
    edges = [r["edges"] for r in series]
    n = len(series)
    blues = sns.color_palette("Blues", n + 2)[2:]
    x = np.arange(n)

    # Left: recall bar chart
    bars = ax_left.bar(x, recalls, color=blues, width=0.6, zorder=3)
    for bar, val in zip(bars, recalls):
        ax_left.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=8,
        )
    ax_left.set_xticks(x)
    ax_left.set_xticklabels([str(m) for m in mconj_vals])
    ax_left.set_xlabel("M_conj", fontsize=10)
    ax_left.set_ylabel("Post-drift mean Recall@10  (epochs 20–24)", fontsize=10)
    ax_left.set_title("Recall vs M_conj", fontsize=11, fontweight="bold")
    ax_left.set_ylim(0, min(1.0, max(recalls) + 0.06))

    # Right: edges scatter + linear fit
    ax_right.scatter(mconj_vals, edges, color=blues, s=60, zorder=3)
    for mx, ev in zip(mconj_vals, edges):
        ax_right.annotate(
            f"{ev:,}",
            xy=(mx, ev),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    if len(mconj_vals) >= 2:
        fit = np.polyfit(mconj_vals, edges, 1)
        x_fit = np.linspace(min(mconj_vals) - 2, max(mconj_vals) + 2, 200)
        ax_right.plot(x_fit, np.polyval(fit, x_fit), color="steelblue", linewidth=1.5,
                      linestyle="--", alpha=0.7, label=f"linear fit  slope={fit[0]:.0f} edges/unit")
        ax_right.legend(fontsize=8, loc="upper left")
    ax_right.set_xlabel("M_conj", fontsize=10)
    ax_right.set_ylabel("Conjugate edges at epoch 24", fontsize=10)
    ax_right.set_title("Edge count vs M_conj", fontsize=11, fontweight="bold")

    fig.suptitle(
        f"M_conj scaling: recall and edge cost  (gradual drift, ef={PRIMARY_EF})",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = figures_dir / "mconj_scaling.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def print_summary_table(ef32, cfg):
    conditions = cfg["conditions"]
    header = (
        f"{'Condition':<14}  {'M_conj':>6}  {'Post-drift R@10':>15}  "
        f"{'Recall drop':>11}  {'Conj edges ep24':>15}  {'Repair events':>13}"
    )
    print("\n" + header)
    print("-" * len(header))
    for cond in conditions:
        cname = cond["name"]
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
            f"{cname:<14}  {cond['M_conj']:>6}  {post:>15.4f}  "
            f"{drop:>11.4f}  {edges_val:>15,}  {repair_count:>13d}"
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

    plot_figure1(ef32, ablation_baseline, schedule, cfg["conditions"], figures_dir)
    plot_figure2(cfg, ef32, figures_dir)
    print_summary_table(ef32, cfg)


if __name__ == "__main__":
    main()
