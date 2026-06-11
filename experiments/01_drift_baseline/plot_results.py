import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DIR = Path(__file__).parent
OUT = DIR / "plots"
OUT.mkdir(exist_ok=True)

gradual = pd.read_csv(DIR / "results_gradual.csv", usecols=["epoch", "t", "ef_search", "recall"])
sudden  = pd.read_csv(DIR / "results_sudden.csv",  usecols=["epoch", "t", "ef_search", "recall"])

ef_values = sorted(gradual["ef_search"].unique())
colors = plt.cm.viridis([i / (len(ef_values) - 1) for i in range(len(ef_values))])


# Figure 1: recall@10 vs epoch, gradual and sudden side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for ef, c in zip(ef_values, colors):
    sub = gradual[gradual["ef_search"] == ef].sort_values("epoch")
    ax1.plot(sub["epoch"], sub["recall"], label=f"ef={ef}", color=c)

# shade drift phases on gradual
ax1.axvspan(0, 5, alpha=0.05, color="blue", label="pre-drift")
ax1.axvspan(5, 15, alpha=0.08, color="orange",label="ramp 0→0.5")
ax1.axvspan(15, 20, alpha=0.08, color="red", label="ramp 0.5→1")
ax1.axvspan(20, 24, alpha=0.05, color="darkred")
ax1.set_xlabel("epoch")
ax1.set_ylabel("recall@10")
ax1.set_title("gradual drift")
ax1.legend(fontsize=7)

for ef, c in zip(ef_values, colors):
    sub = sudden[sudden["ef_search"] == ef].sort_values("epoch")
    ax2.plot(sub["epoch"], sub["recall"], color=c)

ax2.axvline(12.5, color="red", linestyle="--", linewidth=1, label="drift onset")
ax2.set_xlabel("epoch")
ax2.set_title("sudden drift")
ax2.legend(fontsize=7)

fig.tight_layout()
fig.savefig(OUT / "fig1_recall_vs_epoch.png", dpi=150)
print("saved fig1")


# Figure 2: drift diagnostics vs epoch (from diagnostics.json)
diag_path = DIR / "dataset_gradual" / "diagnostics.json"
if diag_path.exists():
    diag = pd.DataFrame(json.loads(diag_path.read_text()))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, col, label in zip(axes,
            ["ood_distance", "mmd_squared", "mean_nn_distance"],
            ["OOD distance", "MMD²", "mean NN distance"]):
        ax.plot(diag["epoch_idx"], diag[col])
        ax.set_xlabel("epoch")
        ax.set_title(label)

    fig.tight_layout()
    fig.savefig(OUT / "fig2_drift_diagnostics.png", dpi=150)
    print("saved fig2")
else:
    print(f"skipping fig2 — {diag_path} not found (sync dataset_gradual/ from cluster)")


# Figure 3:recall@10 vs drift magnitude t, coloured by ef
fig, ax = plt.subplots(figsize=(6, 4))
for ef, c in zip(ef_values, colors):
    sub = gradual[gradual["ef_search"] == ef]
    ax.scatter(sub["t"], sub["recall"], label=f"ef={ef}", color=c, s=15, alpha=0.7)

ax.set_xlabel("drift parameter t")
ax.set_ylabel("recall@10")
ax.set_title("recall vs drift magnitude")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "fig3_recall_vs_t.png", dpi=150)
print("saved fig3")
