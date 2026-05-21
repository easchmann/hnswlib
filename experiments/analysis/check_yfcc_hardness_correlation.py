# Check whether query hardness correlates with temporal drift in YFCC.
#
# Drift proxy: shift_sigma (displacement along the 2007->2013 mean direction).
# Hardness proxy: (1 - recall@k) at a low ef, and base_dist_comps at the same ef.
# If drift correlates with hardness, the two concepts are interchangeable as
# dataset construction strategies.
#
# Usage:
#   python check_yfcc_hardness_correlation.py \
#       --embeddings_path ../../data/yfcc/embeddings.npy \
#       --metadata_path   ../../data/yfcc/metadata.csv

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))

import hnswlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

parser = argparse.ArgumentParser()
parser.add_argument("--embeddings_path", required=True)
parser.add_argument("--metadata_path", required=True)
parser.add_argument("--n_index", type=int, default=500_000)
parser.add_argument("--n_queries", type=int, default=1_000)
parser.add_argument("--shift_sigmas", type=float, nargs="+", default=[0, 1, 2, 3, 4, 6, 8, 10, 12, 16])
parser.add_argument("--ef_hard", type=int, default=20)
parser.add_argument("--M", type=int, default=16)
parser.add_argument("--ef_construction", type=int, default=200)
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--out_dir", default="results/yfcc_hardness_check")
args = parser.parse_args()

dim = 384
k = args.k
os.makedirs(args.out_dir, exist_ok=True)


def load_data():
    print("loading embeddings...")
    emb = np.load(args.embeddings_path, mmap_mode="r")
    meta = pd.read_csv(args.metadata_path)
    assert len(emb) == len(meta)
    print(f"  {len(emb):,} embeddings  dim={emb.shape[1]}")
    return emb, meta


def compute_drift_direction(emb, meta):
    m07 = np.array(emb[(meta["year"] == 2007).values], dtype=np.float32).mean(axis=0)
    m13 = np.array(emb[(meta["year"] == 2013).values], dtype=np.float32).mean(axis=0)
    d = m13 - m07
    mag = float(np.linalg.norm(d))
    d /= mag
    print(f"drift direction magnitude (2007->2013): {mag:.4f}")
    return d.astype(np.float32), mag


def build_index(index_vecs):
    print(f"building index on {len(index_vecs):,} vectors...")
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.init_index(max_elements=len(index_vecs), ef_construction=args.ef_construction, M=args.M)
    idx.add_items(index_vecs, np.arange(len(index_vecs), dtype=np.int32))
    return idx


def compute_gt(index_vecs, queries):
    import faiss
    flat = faiss.IndexFlatL2(dim)
    flat.add(np.ascontiguousarray(index_vecs, dtype=np.float32))
    _, nbrs = flat.search(np.ascontiguousarray(queries, dtype=np.float32), k)
    return nbrs.astype(np.int32)


def measure_hardness(idx, queries, gt):
    idx.set_ef(args.ef_hard)
    hard_recall, hard_comps, bl_entry = [], [], []
    for i, q in enumerate(queries):
        lbls, _ = idx.knn_query(q.reshape(1, -1), k=k)
        s = hnswlib.get_last_query_stats()
        recall = len(set(lbls[0]) & set(gt[i])) / k
        hard_recall.append(1.0 - recall)
        hard_comps.append(int(s["base_layer_distance_computations"]))
        bl_entry.append(float(s["base_layer_entry_distance"]))
    return (np.array(hard_recall, dtype=np.float32), np.array(hard_comps, dtype=np.float32), np.array(bl_entry, dtype=np.float32))


def main():
    print(f"YFCC hardness-drift correlation check")
    print(f"n_index={args.n_index:,}  n_queries={args.n_queries}  ef_hard={args.ef_hard}  k={k}\n")

    emb, meta = load_data()
    direction, magnitude = compute_drift_direction(emb, meta)

    n_total = len(emb)
    n_index = min(args.n_index, n_total - args.n_queries)
    if n_index < args.n_index:
        print(f"  capping n_index to {n_index:,} (dataset has {n_total:,} vectors)")
    chosen = rng.choice(n_total, n_index, replace=False)
    chosen.sort()
    index_vecs = np.array(emb[chosen], dtype=np.float32)

    q_idx = rng.choice(n_total, args.n_queries, replace=False)
    q_idx.sort()
    query_base = np.array(emb[q_idx], dtype=np.float32)
    del emb

    idx = build_index(index_vecs)

    rows = []
    all_sigma, all_hard_recall, all_hard_comps, all_bl = [], [], [], []

    for sigma in args.shift_sigmas:
        shift = direction * sigma * magnitude
        queries = query_base + shift

        print(f"\nsigma={sigma}  |shift|={np.linalg.norm(shift):.4f}")
        gt = compute_gt(index_vecs, queries)
        hard_r, hard_c, bl = measure_hardness(idx, queries, gt)

        print(f"  mean recall={1-hard_r.mean():.4f}  mean_comps={hard_c.mean():.1f}  mean_bl={bl.mean():.4f}")

        for i in range(len(queries)):
            rows.append({
                "shift_sigma": sigma,
                "query_id": i,
                "hardness_recall": float(hard_r[i]),
                "base_dist_comps": float(hard_c[i]),
                "bl_entry_dist": float(bl[i]),
                "recall": 1.0 - float(hard_r[i]),
            })
        all_sigma.extend([sigma] * len(queries))
        all_hard_recall.extend(hard_r.tolist())
        all_hard_comps.extend(hard_c.tolist())
        all_bl.extend(bl.tolist())

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "per_query_hardness.csv"), index=False)

    # correlations
    all_sigma = np.array(all_sigma, dtype=np.float32)
    print("\n--- sigma vs. hardness correlations ---")
    for name, arr in [("1-recall@k", np.array(all_hard_recall)),
                      ("base_dist_comps", np.array(all_hard_comps)),
                      ("bl_entry_dist", np.array(all_bl))]:
        pr, pp = stats.pearsonr(all_sigma, arr)
        sr, sp = stats.spearmanr(all_sigma, arr)
        print(f"\n  {name}:")
        print(f"    Pearson  r={pr:+.4f}  p={pp:.2e}")
        print(f"    Spearman r={sr:+.4f}  p={sp:.2e}")

    corr_rows = []
    for name, arr in [("1-recall@k", np.array(all_hard_recall)),
                      ("base_dist_comps", np.array(all_hard_comps)),
                      ("bl_entry_dist", np.array(all_bl))]:
        pr, pp = stats.pearsonr(all_sigma, arr)
        sr, sp = stats.spearmanr(all_sigma, arr)
        corr_rows.append({"metric": name, "pearson_r": round(float(pr), 4), "pearson_p": float(pp),
                          "spearman_r": round(float(sr), 4), "spearman_p": float(sp)})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(os.path.join(args.out_dir, "correlations.csv"), index=False)
    print(f"\n{corr_df.to_string(index=False)}")

    summary = (df.groupby("shift_sigma")
                 .agg(mean_recall=("recall", "mean"),
                      mean_hardness=("hardness_recall", "mean"),
                      mean_comps=("base_dist_comps", "mean"),
                      mean_bl=("bl_entry_dist", "mean"))
                 .round(4))
    print(f"\n{summary.to_string()}")
    summary.to_csv(os.path.join(args.out_dir, "summary_by_sigma.csv"))

    
    

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    col_map = {"1-recall@k": "hardness_recall", "base_dist_comps": "base_dist_comps", "bl_entry_dist": "bl_entry_dist"}
    for ax, (title, col) in zip(axes, col_map.items()):
        arr = np.array(all_hard_recall if "recall" in col else (all_hard_comps if "comps" in col else all_bl))
        jitter = rng.uniform(-0.15, 0.15, size=len(all_sigma))
        ax.scatter(all_sigma + jitter, arr, alpha=0.15, s=4, rasterized=True)
        means = df.groupby("shift_sigma")[col].mean()
        ax.plot(means.index, means.values, "r-o", linewidth=2, markersize=5, label="mean")
        ax.set_xlabel("shift_sigma")
        ax.set_ylabel(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "hardness_correlation.png"), dpi=150)
    print(f"\nsaved hardness_correlation.png")
  

    print(f"\ndone — results in {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
