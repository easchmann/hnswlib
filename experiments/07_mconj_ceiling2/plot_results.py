"""Experiment 07 figures: full M_conj progression with saturation fit."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import yaml

ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = _SCRIPT_DIR / "config.yaml"
DPI = 300
PRIMARY_EF = 32

# (dir_key_in_output, csv_stem, M_conj)
PRIOR_CONDITIONS = [
    ("exp04_results_dir", "mconj_12",    12),
    ("exp05_results_dir", "mconj_8_cd0",  8),
    ("exp05_results_dir", "mconj_16_cd0", 16),
    ("exp05_results_dir", "mconj_20_cd0", 20),
    ("exp06_results_dir", "mconj_24",    24),
    ("exp06_results_dir", "mconj_28",    28),
    ("exp06_results_dir", "mconj_32",    32),
    ("exp06_results_dir", "mconj_40",    40),
    ("exp06_results_dir", "mconj_48",    48),
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


def _mean_post_drift(df, condition_col_val, lo=20, hi=24):
    sub = df[(df["condition"] == condition_col_val) & (df["epoch_idx"] >= lo)]
    return sub["recall_at_k"].mean()


def _build_full_series(cfg):
    """Collect (M_conj, post_drift_recall, pre_drift_recall, edges_ep24) from all experiments."""
    rows = []
    seen_mconj = set()

    for dir_key, csv_stem, mconj in PRIOR_CONDITIONS:
        exp_dir_key = cfg["output"].get(dir_key, "")
        candidates = [ROOT / exp_dir_key / f"{csv_stem}.csv"]
        # Fallback to experiment script directories
        exp_subdir_map = {
            "exp04_results_dir": "experiments/04_tuning",
            "exp05_results_dir": "experiments/05_mconj_cooldown",
            "exp06_results_dir": "experiments/06_mconj_ceiling",
        }
        if dir_key in exp_subdir_map:
            candidates.append(ROOT / exp_subdir_map[dir_key] / f"{csv_stem}.csv")

        df = None
        for p in candidates:
            if p.exists():
                df = pd.read_csv(p)
                break
        if df is None:
            print(f"  WARNING: no data for M_conj={mconj} ({csv_stem}) — skipping")
            continue

        ef_df = df[df["ef_search"] == PRIMARY_EF]
        pre = ef_df[ef_df["epoch_idx"] <= 4]["recall_at_k"].mean()
        post = ef_df[ef_df["epoch_idx"] >= 20]["recall_at_k"].mean()
        ep24 = ef_df[ef_df["epoch_idx"] == 24]["n_conjugate_edges"].values
        edges = int(ep24[0]) if len(ep24) > 0 else 0
        rows.append({"M_conj": mconj, "recall": post, "pre_recall": pre, "edges": edges})
        seen_mconj.add(mconj)

    # This experiment (56, 64)
    results_dir = _results_dir(cfg)
    for cond in cfg["conditions"]:
        mconj = cond["M_conj"]
        p = results_dir / f"{cond['name']}.csv"
        if not p.exists():
            print(f"  WARNING: no data for {cond['name']} — skipping")
            continue
        df = pd.read_csv(p)
        ef_df = df[df["ef_search"] == PRIMARY_EF]
        pre = ef_df[ef_df["epoch_idx"] <= 4]["recall_at_k"].mean()
        post = ef_df[ef_df["epoch_idx"] >= 20]["recall_at_k"].mean()
        ep24 = ef_df[ef_df["epoch_idx"] == 24]["n_conjugate_edges"].values
        edges = int(ep24[0]) if len(ep24) > 0 else 0
        rows.append({"M_conj": mconj, "recall": post, "pre_recall": pre, "edges": edges})

    rows.sort(key=lambda r: r["M_conj"])
    return rows


def _saturation_model(m, r_max, c):
    return r_max - c / m


def _fit_saturation(mconj_vals, recalls):
    """Fit R(m) = R_max - c/m; returns (r_max, c) or None on failure."""
    try:
        x = np.array(mconj_vals, dtype=float)
        y = np.array(recalls, dtype=float)
        p0 = [max(y) + 0.05, 1.0]
        popt, _ = curve_fit(_saturation_model, x, y, p0=p0, maxfev=10000)
        return float(popt[0]), float(popt[1])
    except Exception as e:
        print(f"  Saturation fit failed: {e}")
        return None


def plot_full_progression(series, figures_dir):
    """Bar chart of post-drift recall vs M_conj with saturation fit."""
    if not series:
        print("No data — skipping figure.")
        return

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(13, 5))

    mconj_vals = [r["M_conj"] for r in series]
    recalls = [r["recall"] for r in series]
    n = len(series)
    blues = sns.color_palette("Blues", n + 2)[2:]
    x = np.arange(n)

    bars = ax.bar(x, recalls, color=blues, width=0.6, zorder=3)
    for bar, val in zip(bars, recalls):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold",
        )

    # Saturation fit overlay
    fit = _fit_saturation(mconj_vals, recalls)
    if fit is not None:
        r_max, c = fit
        x_cont = np.linspace(min(mconj_vals) - 2, max(mconj_vals) + 10, 400)
        y_cont = _saturation_model(x_cont, r_max, c)
        # Map x_cont to bar x-axis positions via interpolation
        x_bar = np.interp(x_cont, mconj_vals, range(n))
        ax.plot(
            x_bar, y_cont,
            color="tomato", linewidth=2.0, linestyle="-",
            label=f"fit: $R_{{\\max}} - c/m$",
            zorder=4, alpha=0.85,
        )
        ax.axhline(
            r_max,
            color="tomato", linewidth=1.5, linestyle="--",
            label=f"$R_{{\\max}} = {r_max:.4f}$",
            zorder=4, alpha=0.7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in mconj_vals], fontsize=10)
    ax.set_xlabel("$M_{\\mathrm{conj}}$", fontsize=12)
    ax.set_ylabel("Post-drift mean Recall@10  (epochs 20–24)", fontsize=11)
    ax.set_title(
        f"Full $M_{{\\mathrm{{conj}}}}$ progression  (gradual drift, ef={PRIMARY_EF})",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(0, min(1.0, max(recalls) + 0.08))
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    out = figures_dir / "mconj_full_progression.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def print_summary_table(series):
    header = (
        f"{'M_conj':>7}  {'Post-drift R@10':>15}  {'Recall drop':>11}  {'Edges ep24':>12}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in series:
        drop = r.get("pre_recall", float("nan")) - r["recall"]
        print(
            f"{r['M_conj']:>7}  {r['recall']:>15.4f}  {drop:>11.4f}  {r['edges']:>12,}"
        )
    print()


def main():
    cfg = _load_config()
    results_dir = _results_dir(cfg)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    series = _build_full_series(cfg)
    plot_full_progression(series, figures_dir)
    print_summary_table(series)


if __name__ == "__main__":
    main()
