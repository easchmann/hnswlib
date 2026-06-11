# Analyze whether queries in the same hardness bucket are spatially closer to
# each other than queries in other buckets.
#
# Runs two bucketing strategies back-to-back:
#   recall      — equal-count buckets by hardness_score (1 - recall@k at ef_hard)
#   dist_comps  — equal-count buckets by base_dist_comps (requires metadata.csv)
#                 this matches the signal the HardnessAdaptiveController acts on
#
# Metrics per strategy:
#   - Mean intra vs inter-bucket L2 distance heatmap + intra/inter ratio
#   - Per-bucket silhouette scores (sklearn, vectorized)
#   - Optional PCA scatter (--pca)
#
# Usage:
#   python experiments/analysis/analyse_sift_hardness_clustering.py \
#       --data_dir data/sift_hardness \
#       --n_buckets 10 \
#       --sample_per_bucket 300

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True)
parser.add_argument("--n_buckets", type=int, default=10)
parser.add_argument("--sample_per_bucket", type=int, default=300)
parser.add_argument("--pca", action="store_true")
parser.add_argument("--out_dir", default=None)
args = parser.parse_args()

out_dir = args.out_dir or args.data_dir
os.makedirs(out_dir, exist_ok=True)

rng = np.random.default_rng(42)

# load
print("Loading data...")
queries  = np.load(os.path.join(args.data_dir, "queries_by_hardness.npy"))
hardness = np.load(os.path.join(args.data_dir, "hardness_scores.npy"))
meta_path = os.path.join(args.data_dir, "metadata.csv")
meta = pd.read_csv(meta_path) if os.path.exists(meta_path) else None

N = len(queries)
print(f"  {N:,} queries  dim={queries.shape[1]}  "
      f"hardness range [{hardness.min():.3f}, {hardness.max():.3f}]")


# core analysis 
def run_analysis(queries, bucket_ids, bucket_edges, bucket_labels, label):
    """Run spatial clustering analysis for one bucketing strategy.

    label         — short string used in filenames and print headers ("recall" or "dist_comps")
    bucket_ids    — (N,) int array, values in [0, n_buckets)
    bucket_edges  — list of (lo, hi) signal-value tuples per bucket
    bucket_labels — list of tick label strings per bucket
    """
    n_buckets = len(bucket_edges)
    print(f"\n{'='*60}")
    print(f"Bucketing by: {label}")
    print(f"{'='*60}")
    for b, (lo, hi) in enumerate(bucket_edges):
        print(f"  B{b}: [{lo:.3g}, {hi:.3g}]  n={(bucket_ids==b).sum():,}")

    # sample per bucket
    samples = []
    for b in range(n_buckets):
        idx    = np.where(bucket_ids == b)[0]
        chosen = rng.choice(idx, min(args.sample_per_bucket, len(idx)), replace=False)
        samples.append(queries[chosen])

    # distance matrix
    print(f"\nComputing distance matrix ({n_buckets}×{n_buckets}, "
          f"{args.sample_per_bucket} samples/bucket)...")
    mean_dist = np.zeros((n_buckets, n_buckets))
    for i in range(n_buckets):
        for j in range(i, n_buckets):
            d = cdist(samples[i], samples[j], metric="euclidean")
            if i == j:
                mask = ~np.eye(len(d), dtype=bool)
                mean_dist[i, j] = d[mask].mean()
            else:
                mean_dist[i, j] = d.mean()
            mean_dist[j, i] = mean_dist[i, j]

    intra = np.array([mean_dist[b, b] for b in range(n_buckets)])
    inter = np.array([np.mean([mean_dist[b, j] for j in range(n_buckets) if j != b])
                      for b in range(n_buckets)])
    ratio = intra / inter

    print(f"\n  {'Bucket':<8} {'Intra':>10} {'Inter':>10} {'Ratio':>8}")
    for b in range(n_buckets):
        flag = "  << clustered" if ratio[b] < 1.0 else ""
        print(f"  B{b:<7} {intra[b]:>10.2f} {inter[b]:>10.2f} {ratio[b]:>8.3f}{flag}")

    # silhouette
    all_vecs    = np.concatenate(samples)
    all_buckets = np.concatenate([np.full(len(s), b) for b, s in enumerate(samples)])

    print("\nComputing silhouette scores...")
    sil_scores = silhouette_samples(all_vecs, all_buckets, metric="euclidean")
    print(f"  Mean: {sil_scores.mean():.4f}  (>0 → spatial clustering)")
    for b in range(n_buckets):
        print(f"  B{b}: {sil_scores[all_buckets == b].mean():.4f}")

    # save CSV
    sil_df = pd.DataFrame({
        "bucket":            all_buckets,
        "silhouette":        sil_scores,
        "intra_dist":        [mean_dist[b, b] for b in all_buckets],
        "inter_dist":        [inter[b] for b in all_buckets],
        "intra_inter_ratio": [ratio[b] for b in all_buckets],
    })
    csv_path = os.path.join(out_dir, f"silhouette_scores_{label}.csv")
    sil_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # plot: distance heatmap + intra/inter ratio
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    im = axes[0].imshow(mean_dist, cmap="viridis_r")
    axes[0].set_xticks(range(n_buckets)); axes[0].set_xticklabels(bucket_labels, fontsize=8)
    axes[0].set_yticks(range(n_buckets)); axes[0].set_yticklabels(bucket_labels, fontsize=8)
    axes[0].set_title(f"Mean L2 distance between buckets [{label}]\n(darker = closer)")
    plt.colorbar(im, ax=axes[0])
    for i in range(n_buckets):
        for j in range(n_buckets):
            axes[0].text(j, i, f"{mean_dist[i,j]:.0f}", ha="center", va="center",
                         color="white" if mean_dist[i,j] < mean_dist.max()*0.6 else "black",
                         fontsize=7)

    colors = ["tab:green" if r < 1 else "tab:red" for r in ratio]
    axes[1].bar(range(n_buckets), ratio, color=colors, alpha=0.8, edgecolor="black")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1, label="ratio=1")
    axes[1].set_xticks(range(n_buckets)); axes[1].set_xticklabels(bucket_labels, fontsize=8)
    axes[1].set_ylabel("Intra / Inter distance ratio")
    axes[1].set_title(f"Spatial clustering [{label}]\n(green < 1 → same-bucket queries closer)")
    axes[1].legend()
    plt.tight_layout()
    cluster_path = os.path.join(out_dir, f"hardness_clustering_{label}.png")
    plt.savefig(cluster_path, dpi=150)
    print(f"  Saved: {cluster_path}")
    plt.close()

    # plot: silhouette boxplot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([sil_scores[all_buckets == b] for b in range(n_buckets)],
               labels=bucket_labels, patch_artist=True,
               boxprops=dict(facecolor="steelblue", alpha=0.6))
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1, label="silhouette=0")
    ax.axhline(sil_scores.mean(), color="tab:orange", linestyle="--", linewidth=1,
               label=f"global mean={sil_scores.mean():.3f}")
    ax.set_ylabel("Silhouette score")
    ax.set_title(f"Per-bucket silhouette [{label}]\n(>0 → queries cluster within bucket)")
    ax.legend()
    plt.tight_layout()
    sil_plot_path = os.path.join(out_dir, f"hardness_silhouette_{label}.png")
    plt.savefig(sil_plot_path, dpi=150)
    print(f"  Saved: {sil_plot_path}")
    plt.close()

    # optional PCA scatter
    if args.pca:
        from sklearn.decomposition import PCA
        print("  Computing PCA...")
        pca  = PCA(n_components=2, random_state=42)
        proj = pca.fit_transform(all_vecs)
        fig, ax = plt.subplots(figsize=(7, 6))
        cmap = plt.get_cmap("RdYlGn_r", n_buckets)
        for b in range(n_buckets):
            mask = all_buckets == b
            ax.scatter(proj[mask, 0], proj[mask, 1], s=6, alpha=0.5, color=cmap(b),
                       label=f"B{b} [{bucket_edges[b][0]:.2g},{bucket_edges[b][1]:.2g}]")
        ax.set_title(f"PCA colored by bucket [{label}]")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.legend(fontsize=7, markerscale=3)
        plt.tight_layout()
        pca_path = os.path.join(out_dir, f"hardness_pca_{label}.png")
        plt.savefig(pca_path, dpi=150)
        print(f"  Saved: {pca_path}")
        plt.close()


# ── Strategy 1: recall-based bucketing ───────────────────────────────────────
bucket_ids_recall = np.floor(np.linspace(0, args.n_buckets, N, endpoint=False)).astype(int)
edges_recall  = [(hardness[bucket_ids_recall == b].min(),
                  hardness[bucket_ids_recall == b].max())
                 for b in range(args.n_buckets)]
labels_recall = [f"B{b}\n[{lo:.2f},{hi:.2f}]" for b, (lo, hi) in enumerate(edges_recall)]

run_analysis(queries, bucket_ids_recall, edges_recall, labels_recall, label="recall")

# ── Strategy 2: dist_comps-based bucketing ────────────────────────────────────
if meta is not None and "base_dist_comps" in meta.columns:
    signal     = meta["base_dist_comps"].to_numpy()
    thresholds = np.quantile(signal, np.linspace(0, 1, args.n_buckets + 1))
    bucket_ids_dc = np.clip(np.digitize(signal, thresholds[1:-1]), 0, args.n_buckets - 1)
    edges_dc  = [(signal[bucket_ids_dc == b].min(), signal[bucket_ids_dc == b].max())
                 for b in range(args.n_buckets)]
    labels_dc = [f"B{b}\n[{lo:.0f},{hi:.0f}]" for b, (lo, hi) in enumerate(edges_dc)]

    run_analysis(queries, bucket_ids_dc, edges_dc, labels_dc, label="dist_comps")
else:
    print("\nSkipping dist_comps bucketing: metadata.csv not found or missing base_dist_comps column.")

print("\nDone.")
