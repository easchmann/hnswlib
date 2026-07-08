"""Exp 39 plots: ef escalation on DEEP-96 cluster drift."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

COLORS = {
    "static":             "grey",
    "adaptive_eh":        "steelblue",
    "ef_escalation_2x":   "darkorange",
    "ef_escalation_4x":   "firebrick",
    "oracle_ef128":       "forestgreen",
}
LABELS = {
    "static":             "Static",
    "adaptive_eh":        "Adaptive EH (conjugate)",
    "ef_escalation_2x":   "EH-gated escalation ×2",
    "ef_escalation_4x":   "EH-gated escalation ×4",
    "oracle_ef128":       "Oracle ef=128 (all queries)",
}
# order for legend
COND_ORDER = [
    "static",
    "adaptive_eh",
    "ef_escalation_2x",
    "ef_escalation_4x",
    "oracle_ef128",
]


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _plateau_recall(df, condition, ef, plateau_start=20, plateau_end=24):
    sub = df[
        (df["condition"] == condition)
        & (df["ef"] == ef)
        & (df["epoch"] >= plateau_start)
        & (df["epoch"] <= plateau_end)
    ]
    return float(sub["recall"].mean()) if len(sub) > 0 else np.nan


def plot_schedule(df, schedule, cfg, plots_dir):
    primary_ef = cfg["primary_ef"]
    ef_values = sorted(df["ef"].unique())
    transitions = cfg.get(f"drift_transition_epochs_{schedule}", [])
    present = set(df["condition"].unique())

    # Figure 1: recall@10 vs epoch at primary_ef for all conditions
    fig, ax = plt.subplots(figsize=(8, 4))
    for cond in COND_ORDER:
        if cond not in present:
            continue
        # For oracle_ef128, use ef=oracle_ef; for others use primary_ef
        ef_to_plot = cfg["ef_oracle"] if cond == "oracle_ef128" else primary_ef
        sub = df[(df["condition"] == cond) & (df["ef"] == ef_to_plot)].sort_values("epoch")
        if sub.empty:
            continue
        label = LABELS[cond]
        if cond == "oracle_ef128":
            label += f" (ef={cfg['ef_oracle']})"
        elif cond in ("ef_escalation_2x", "ef_escalation_4x"):
            factor = int(cond.split("_")[-1][:-1])
            label += f" (base ef={primary_ef}→{primary_ef * factor})"
        ax.plot(sub["epoch"], sub["recall"], color=COLORS[cond], label=label, linewidth=1.5)
    for t in transitions:
        ax.axvline(t, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Recall@{cfg['k']}")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"DEEP-96 cluster drift — ef escalation ({schedule})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = plots_dir / f"{schedule}_recall_primary.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")

    # Figure 2: ef-scaling plot at plateau (epochs 20-24)
    # x-axis: ef_base (32, 64, 128); static curve + ef_escalation_4x + oracle_ef128 point
    fig, ax = plt.subplots(figsize=(6, 4))

    ef_axis = sorted(e for e in ef_values if e != cfg["ef_oracle"])
    if not ef_axis:
        ef_axis = ef_values

    # static: plateau recall at ef=32/64/128
    static_recalls = [_plateau_recall(df, "static", ef) for ef in ef_axis]
    ax.plot(ef_axis, static_recalls, color=COLORS["static"], marker="o",
            label=LABELS["static"], linewidth=1.5)

    # ef_escalation_4x: plateau recall at ef=32/64/128 (with escalation)
    if "ef_escalation_4x" in present:
        esc4_recalls = [_plateau_recall(df, "ef_escalation_4x", ef) for ef in ef_axis]
        ax.plot(ef_axis, esc4_recalls, color=COLORS["ef_escalation_4x"], marker="s",
                label=LABELS["ef_escalation_4x"], linewidth=1.5)

    # oracle_ef128: single plateau recall at ef=oracle_ef — horizontal dashed line
    if "oracle_ef128" in present:
        oracle_recall = _plateau_recall(df, "oracle_ef128", cfg["ef_oracle"])
        ax.axhline(oracle_recall, color=COLORS["oracle_ef128"], linestyle="--",
                   linewidth=1.5, label=f"Oracle ef={cfg['ef_oracle']} (all queries)")

    ax.set_xlabel("ef_base")
    ax.set_ylabel(f"Mean recall@{cfg['k']} (plateau ep20–24)")
    ax.set_xticks(ef_axis)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"ef-scaling at plateau — {schedule} drift")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = plots_dir / f"{schedule}_ef_scaling.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")

    # Figure 3: n_escalated vs epoch for escalation conditions
    escalation_conds = [c for c in present if c.startswith("ef_escalation")]
    if escalation_conds:
        fig, ax = plt.subplots(figsize=(6, 3))
        for cond in escalation_conds:
            sub = df[(df["condition"] == cond) & (df["ef"] == primary_ef)].sort_values("epoch")
            if "n_escalated" in sub.columns:
                ax.plot(sub["epoch"], sub["n_escalated"], color=COLORS[cond],
                        label=LABELS[cond], linewidth=1.5)
        for t in transitions:
            ax.axvline(t, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Queries escalated / epoch")
        ax.set_title(f"Escalation rate — {schedule} drift")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = plots_dir / f"{schedule}_escalation_rate.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"  saved {out}")


def main():
    config_path = (
        sys.argv[1] if len(sys.argv) > 1
        else Path(__file__).parent / "config.yaml"
    )
    cfg = _load_config(config_path)
    plots_dir = ROOT / cfg["plots_dir"]
    plots_dir.mkdir(parents=True, exist_ok=True)

    for schedule, results_key in [
        ("gradual", "results_gradual"),
        ("sudden",  "results_sudden"),
    ]:
        path = ROOT / cfg[results_key]
        if not path.exists():
            print(f"Missing {path}, skipping {schedule}")
            continue
        df = pd.read_csv(path)
        plot_schedule(df, schedule, cfg, plots_dir)


if __name__ == "__main__":
    main()
