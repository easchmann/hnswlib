"""Mmd2gate comparison plots: adaptive_improved (no gate) vs adaptive_improved_mmd2gate."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).parent

COLORS = {
    "static":                     "grey",
    "adaptive_current":           "steelblue",
    "adaptive_improved":          "darkorange",
    "adaptive_improved_mmd2gate": "firebrick",
}
LABELS = {
    "static":                     "Static",
    "adaptive_current":           "Adaptive EH (current)",
    "adaptive_improved":          "Adaptive improved (no gate)",
    "adaptive_improved_mmd2gate": "Adaptive improved + MMD² gate",
}
LINE_STYLES = {
    "static":                     "-",
    "adaptive_current":           "-",
    "adaptive_improved":          "--",
    "adaptive_improved_mmd2gate": "-",
}


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def plot_schedule(schedule, cfg, cfg_gate, plots_dir):
    primary_ef = cfg["primary_ef"]
    transitions = cfg.get(f"drift_transition_epochs_{schedule}", [])

    # load no-gate results (hardness/) — filter to ef=primary_ef
    path_nogate = ROOT / cfg["results_" + schedule]
    # load gated results (hardness_mmd2gate/)
    path_gate = ROOT / cfg_gate["results_" + schedule]

    dfs = {}
    for label, path in [("nogate", path_nogate), ("gate", path_gate)]:
        if not path.exists():
            print(f"Missing {path}, skipping")
            return
        dfs[label] = pd.read_csv(path)

    df_nogate = dfs["nogate"]
    df_gate = dfs["gate"]

    # Figure 1: recall vs epoch at primary_ef
    fig, ax = plt.subplots(figsize=(8, 4))

    # static (from no-gate file — same in both)
    for df, label in [(df_nogate, "nogate")]:
        sub_static = df[(df["condition"] == "static") & (df["ef"] == primary_ef)].sort_values("epoch")
        if not sub_static.empty:
            ax.plot(sub_static["epoch"], sub_static["recall"],
                    color=COLORS["static"], linestyle=LINE_STYLES["static"],
                    label=LABELS["static"], linewidth=1.5)
            break

    # adaptive_current (from no-gate file)
    sub_cur = df_nogate[(df_nogate["condition"] == "adaptive_current") & (df_nogate["ef"] == primary_ef)].sort_values("epoch")
    if not sub_cur.empty:
        ax.plot(sub_cur["epoch"], sub_cur["recall"],
                color=COLORS["adaptive_current"], linestyle=LINE_STYLES["adaptive_current"],
                label=LABELS["adaptive_current"], linewidth=1.5)

    # adaptive_improved without gate (from no-gate file)
    sub_imp = df_nogate[(df_nogate["condition"] == "adaptive_improved") & (df_nogate["ef"] == primary_ef)].sort_values("epoch")
    if not sub_imp.empty:
        ax.plot(sub_imp["epoch"], sub_imp["recall"],
                color=COLORS["adaptive_improved"], linestyle=LINE_STYLES["adaptive_improved"],
                label=LABELS["adaptive_improved"], linewidth=1.5)

    # adaptive_improved_mmd2gate (from gate file)
    sub_gate = df_gate[(df_gate["condition"] == "adaptive_improved_mmd2gate") & (df_gate["ef"] == primary_ef)].sort_values("epoch")
    if not sub_gate.empty:
        ax.plot(sub_gate["epoch"], sub_gate["recall"],
                color=COLORS["adaptive_improved_mmd2gate"],
                linestyle=LINE_STYLES["adaptive_improved_mmd2gate"],
                label=LABELS["adaptive_improved_mmd2gate"], linewidth=1.5)

    for t in transitions:
        ax.axvline(t, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Recall@{cfg['k']}")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Hardness drift — MMD² gate comparison ({schedule}, ef={primary_ef})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = plots_dir / f"{schedule}_recall_mmd2gate.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")

    # Figure 2: cumulative edge count vs epoch
    fig, ax = plt.subplots(figsize=(7, 3.5))

    sub_cur_ec = df_nogate[(df_nogate["condition"] == "adaptive_current") & (df_nogate["ef"] == primary_ef)].sort_values("epoch")
    if not sub_cur_ec.empty:
        ax.plot(sub_cur_ec["epoch"], sub_cur_ec["edge_count"],
                color=COLORS["adaptive_current"], label=LABELS["adaptive_current"], linewidth=1.5)

    sub_imp_ec = df_nogate[(df_nogate["condition"] == "adaptive_improved") & (df_nogate["ef"] == primary_ef)].sort_values("epoch")
    if not sub_imp_ec.empty:
        ax.plot(sub_imp_ec["epoch"], sub_imp_ec["edge_count"],
                color=COLORS["adaptive_improved"], linestyle="--",
                label=LABELS["adaptive_improved"], linewidth=1.5)

    sub_gate_ec = df_gate[(df_gate["condition"] == "adaptive_improved_mmd2gate") & (df_gate["ef"] == primary_ef)].sort_values("epoch")
    if not sub_gate_ec.empty:
        ax.plot(sub_gate_ec["epoch"], sub_gate_ec["edge_count"],
                color=COLORS["adaptive_improved_mmd2gate"],
                label=LABELS["adaptive_improved_mmd2gate"], linewidth=1.5)

    for t in transitions:
        ax.axvline(t, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cumulative edges added")
    ax.set_title(f"Edge count — {schedule} drift (mmd2gate saves pre-drift budget)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = plots_dir / f"{schedule}_edges_mmd2gate.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


def main():
    # Two configs: no-gate (hardness/) and gated (hardness_mmd2gate/)
    if len(sys.argv) > 1:
        config_gate_path = sys.argv[1]
    else:
        config_gate_path = str(EXP_DIR / "config_hardness_mmd2gate.yaml")

    cfg_gate = _load_config(config_gate_path)

    # load the original (no-gate) hardness config for its result paths
    cfg_nogate = _load_config(str(EXP_DIR / "config_hardness.yaml"))

    plots_dir = ROOT / cfg_gate["plots_dir"]
    plots_dir.mkdir(parents=True, exist_ok=True)

    for schedule in ["gradual", "sudden"]:
        plot_schedule(schedule, cfg_nogate, cfg_gate, plots_dir)


if __name__ == "__main__":
    main()
