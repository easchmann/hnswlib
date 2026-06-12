"""Experiment 09 figures: latency decomposition and repair cost."""

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

METHOD_ORDER = ["batch_knn", "perquery_knn", "perquery_enhanced", "batch_knn_conj"]
METHOD_LABELS = {
    "batch_knn":           "Static (batch knn)",
    "perquery_knn":        "Per-query knn\n(no conjugate)",
    "perquery_enhanced":   "Current adaptive\n(per-query + scalar conj.)",
    "batch_knn_conj":      "Proposed: batch knn\n+ vectorised conj.",
}
METHOD_COLORS = {
    "batch_knn":           "grey",
    "perquery_knn":        "#aec7e8",
    "perquery_enhanced":   "tomato",
    "batch_knn_conj":      "steelblue",
}


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _out_dir(cfg):
    p = ROOT / cfg["output"]["results_dir"]
    if p.exists():
        return p
    return _SCRIPT_DIR


def _load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path)


def plot_figure1(lat_df, cfg, figures_dir):
    """2-panel: (left) latency decomposition at epoch-24, ef=32;
                (right) overhead factor vs ef for each method."""
    primary_ef = cfg["benchmark"]["primary_ef"]
    ef_values = cfg["benchmark"]["ef_search_values"]

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 5))

    # --- Left: latency decomposition at ep-24 CG, primary ef ---
    sub = lat_df[
        (lat_df["part"] == "latency_vs_ef") &
        (lat_df["ef_search"] == primary_ef)
    ]
    # Use largest edge count (ep-24)
    if sub.empty:
        sub = lat_df[
            (lat_df["part"] == "latency_vs_edges") &
            (lat_df["ef_search"] == primary_ef) &
            (lat_df["n_edges"] == lat_df["n_edges"].max())
        ]

    x = np.arange(len(METHOD_ORDER))
    bar_vals = []
    bar_errs = []
    for m in METHOD_ORDER:
        row = sub[sub["method"] == m]
        if row.empty:
            bar_vals.append(0.0)
            bar_errs.append(0.0)
        else:
            bar_vals.append(float(row["mean_ms_per_query"].iloc[0]))
            bar_errs.append(float(row["std_ms_per_query"].iloc[0]))

    bars = ax_left.bar(
        x, bar_vals,
        color=[METHOD_COLORS[m] for m in METHOD_ORDER],
        width=0.55,
        zorder=3,
        yerr=bar_errs,
        capsize=4,
        error_kw={"linewidth": 1.2, "zorder": 4},
    )
    for bar, val in zip(bars, bar_vals):
        ax_left.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(bar_errs) * 0.15 + 0.002,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold",
        )
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(
        [METHOD_LABELS[m] for m in METHOD_ORDER],
        fontsize=8.5,
    )
    ax_left.set_ylabel("Mean latency per query (ms)", fontsize=11)
    ax_left.set_title(
        f"Latency decomposition  (ef={primary_ef}, epoch-24 CG)",
        fontsize=11, fontweight="bold",
    )

    # --- Right: overhead factor vs ef ---
    ef_sub = lat_df[lat_df["part"] == "latency_vs_ef"].copy()
    if ef_sub.empty:
        ax_right.text(0.5, 0.5, "No data", ha="center", va="center",
                      transform=ax_right.transAxes)
    else:
        # normalise by batch_knn latency at each ef
        baseline = ef_sub[ef_sub["method"] == "batch_knn"][["ef_search", "mean_ms_per_query"]]
        baseline = baseline.set_index("ef_search")["mean_ms_per_query"].to_dict()

        for m in ["perquery_knn", "perquery_enhanced", "batch_knn_conj"]:
            m_sub = ef_sub[ef_sub["method"] == m].sort_values("ef_search")
            if m_sub.empty:
                continue
            factors = [
                row["mean_ms_per_query"] / baseline.get(row["ef_search"], 1.0)
                for _, row in m_sub.iterrows()
            ]
            ax_right.plot(
                m_sub["ef_search"], factors,
                color=METHOD_COLORS[m],
                marker="o", linewidth=2.0, markersize=6,
                label=METHOD_LABELS[m].replace("\n", " "),
                zorder=3,
            )

        ax_right.axhline(1.0, color="grey", linewidth=1.0, linestyle="--",
                         label="Static baseline (×1)", zorder=2)
        ax_right.set_xlabel("ef_search", fontsize=11)
        ax_right.set_ylabel("Latency overhead factor  (relative to batch knn)", fontsize=10)
        ax_right.set_title("Overhead factor vs ef_search  (epoch-24 CG)", fontsize=11,
                            fontweight="bold")
        ax_right.set_xticks(cfg["benchmark"]["ef_search_values"])
        ax_right.legend(fontsize=8.5, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    out = figures_dir / "latency_decomposition.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_figure2(lat_df, cfg, figures_dir):
    """Latency vs accumulated edge count (ef=primary, methods A/C/D)."""
    primary_ef = cfg["benchmark"]["primary_ef"]

    sub = lat_df[
        (lat_df["part"] == "latency_vs_edges") &
        (lat_df["ef_search"] == primary_ef)
    ].sort_values("n_edges")

    if sub.empty:
        print("No latency_vs_edges data — skipping Figure 2.")
        return

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(10, 5))

    for m in ["batch_knn", "perquery_enhanced", "batch_knn_conj"]:
        m_sub = sub[sub["method"] == m]
        if m_sub.empty:
            continue
        ax.plot(
            m_sub["n_edges"], m_sub["mean_ms_per_query"],
            color=METHOD_COLORS[m],
            marker="o", linewidth=2.0, markersize=6,
            label=METHOD_LABELS[m].replace("\n", " "),
            zorder=3,
        )
        ax.fill_between(
            m_sub["n_edges"],
            m_sub["mean_ms_per_query"] - m_sub["std_ms_per_query"],
            m_sub["mean_ms_per_query"] + m_sub["std_ms_per_query"],
            color=METHOD_COLORS[m], alpha=0.12, zorder=2,
        )

    ax.set_xlabel("Accumulated conjugate edges", fontsize=11)
    ax.set_ylabel("Mean latency per query (ms)", fontsize=11)
    ax.set_title(
        f"Query latency vs conjugate graph size  (ef={primary_ef})",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.set_xlim(left=-500)

    fig.tight_layout()
    out = figures_dir / "latency_vs_edges.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_figure3(repair_df, cfg, figures_dir):
    """Repair cost: per-event breakdown and amortised overhead per query."""
    if repair_df.empty:
        print("No repair data — skipping Figure 3.")
        return

    epoch_size = cfg["benchmark"]["epoch_size"]

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))

    components = ["compute_candidate_edges", "apply_repairs"]
    comp_labels = ["Search candidates\n(ef_repair=200, 500 nodes)", "Apply repairs\n(edge insertion + RNG)"]
    comp_colors = ["#4393c3", "#74c476"]

    means = []
    stds = []
    for comp in components:
        row = repair_df[repair_df["component"] == comp]
        means.append(float(row["mean_ms"].iloc[0]) if not row.empty else 0.0)
        stds.append(float(row["std_ms"].iloc[0]) if not row.empty else 0.0)

    x = np.arange(len(components))
    bars = ax_left.bar(x, means, color=comp_colors, width=0.5, zorder=3,
                       yerr=stds, capsize=5, error_kw={"linewidth": 1.2, "zorder": 4})
    for bar, val in zip(bars, means):
        ax_left.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{val:.0f} ms",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    total_row = repair_df[repair_df["component"] == "total_repair_event"]
    if not total_row.empty:
        total = float(total_row["mean_ms"].iloc[0])
        ax_left.axhline(total, color="tomato", linewidth=1.5, linestyle="--",
                        label=f"Total: {total:.0f} ms", zorder=4)
        ax_left.legend(fontsize=9)

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(comp_labels, fontsize=10)
    ax_left.set_ylabel("Wall-clock time per repair event (ms)", fontsize=10)
    ax_left.set_title("Repair event cost breakdown", fontsize=11, fontweight="bold")

    # Right: amortised cost comparison
    amortized_vals = {comp: float(repair_df[repair_df["component"] == comp]["amortized_ms_per_query"].iloc[0])
                      for comp in components if not repair_df[repair_df["component"] == comp].empty}

    # Also fetch batch_knn baseline from latency data for reference
    lat_path = figures_dir.parent / "latency_results.csv"
    baseline_ms = None
    if lat_path.exists():
        lat_df = pd.read_csv(lat_path)
        bk = lat_df[(lat_df["method"] == "batch_knn") & (lat_df["ef_search"] == cfg["benchmark"]["primary_ef"])]
        if not bk.empty:
            baseline_ms = float(bk["mean_ms_per_query"].mean())

    total_amort = sum(amortized_vals.values())
    bars_r_labels = list(amortized_vals.keys()) + ["total_repair"]
    bars_r_vals = list(amortized_vals.values()) + [total_amort]
    bars_r_colors = comp_colors + ["tomato"]
    bars_r_display = comp_labels + ["Total repair\n(amortised)"]

    x_r = np.arange(len(bars_r_labels))
    bars_r = ax_right.bar(x_r, bars_r_vals, color=bars_r_colors, width=0.5, zorder=3)
    for bar, val in zip(bars_r, bars_r_vals):
        ax_right.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0005,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    if baseline_ms is not None:
        ax_right.axhline(baseline_ms, color="grey", linewidth=1.5, linestyle="--",
                         label=f"Static knn: {baseline_ms:.4f} ms/q", zorder=4)
        ax_right.legend(fontsize=9)

    ax_right.set_xticks(x_r)
    ax_right.set_xticklabels(bars_r_display, fontsize=9)
    ax_right.set_ylabel(f"Amortised overhead per query (ms)\n[repair cost ÷ {epoch_size} queries/epoch]",
                        fontsize=9)
    ax_right.set_title("Amortised repair overhead per query", fontsize=11, fontweight="bold")

    fig.tight_layout()
    out = figures_dir / "repair_cost.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def print_summary_table(lat_df, repair_df, cfg):
    primary_ef = cfg["benchmark"]["primary_ef"]

    print("\n=== Latency summary (ef={}, epoch-24 CG) ===".format(primary_ef))
    sub = lat_df[(lat_df["part"] == "latency_vs_ef") & (lat_df["ef_search"] == primary_ef)]
    if sub.empty:
        sub = lat_df[
            (lat_df["part"] == "latency_vs_edges") &
            (lat_df["ef_search"] == primary_ef) &
            (lat_df["n_edges"] == lat_df["n_edges"].max())
        ]
    header = f"{'Method':<30}  {'ms/query':>9}  {'p95 ms':>9}  {'QPS':>8}  {'Overhead':>9}"
    print(header)
    print("-" * len(header))
    baseline_ms = None
    for m in METHOD_ORDER:
        row = sub[sub["method"] == m]
        if row.empty:
            continue
        ms = float(row["mean_ms_per_query"].iloc[0])
        p95 = float(row["p95_ms_per_query"].iloc[0])
        qps = 1000.0 / ms if ms > 0 else 0.0
        if m == "batch_knn":
            baseline_ms = ms
        overhead = f"{ms / baseline_ms:.1f}x" if baseline_ms else "-"
        print(f"  {METHOD_LABELS[m].replace(chr(10), ' '):<28}  {ms:>9.4f}  {p95:>9.4f}  {qps:>8.0f}  {overhead:>9}")

    if not repair_df.empty:
        print("\n=== Repair cost summary ===")
        for _, row in repair_df.iterrows():
            print(f"  {row['component']:<30}  {row['mean_ms']:>8.1f} ms  "
                  f"amortised={row['amortized_ms_per_query']:.4f} ms/q")


def main():
    cfg = _load_config()
    out_dir = _out_dir(cfg)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    lat_df = _load_csv(out_dir / "latency_results.csv")
    try:
        repair_df = _load_csv(out_dir / "repair_results.csv")
    except FileNotFoundError:
        repair_df = pd.DataFrame()

    plot_figure1(lat_df, cfg, figures_dir)
    plot_figure2(lat_df, cfg, figures_dir)
    plot_figure3(repair_df, cfg, figures_dir)
    print_summary_table(lat_df, repair_df, cfg)


if __name__ == "__main__":
    main()
