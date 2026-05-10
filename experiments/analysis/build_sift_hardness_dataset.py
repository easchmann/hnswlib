# Build a hardness-sorted query dataset from SIFT1B (bigann_base.bvecs).
#
# Loads n_index vectors as the HNSW index, scores n_queries vectors for
# hardness (1 - recall@k at ef_hard), sorts them easy->hard.
#
# --query_source after_index  load queries from the bvecs file right after
#                             the index block (disjoint, better hardness range)
#               from_index    sample from index vectors (find themselves,
#                             all score hardness=0 — mostly for testing)
#
# Download:
#   wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz
#   gunzip bigann_base.bvecs.gz
#
# Usage:
#   python build_sift_hardness_dataset.py \
#       --bvecs_path ../../data/sift1b/bigann_base.bvecs \
#       --n_index 10000000 --n_queries 5000000 \
#       --out_dir ../../data/sift_hardness

import os
import sys
import json
import struct
import argparse
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))

import hnswlib

rng = np.random.default_rng(42)

parser = argparse.ArgumentParser()
parser.add_argument("--bvecs_path", required=True)
parser.add_argument("--n_index", type=int, default=10_000_000)
parser.add_argument("--n_queries", type=int, default=5_000_000)
parser.add_argument("--query_source", choices=["after_index", "from_index"], default="after_index")
parser.add_argument("--M", type=int, default=16)
parser.add_argument("--ef_construction", type=int, default=200)
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--ef_hard", type=int, default=20)
parser.add_argument("--out_dir", default="data/sift_hardness")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)


def read_bvecs_range(path, start, count):
    # bvecs: [dim:int32][dim bytes:uint8] per record, dim=128 for SIFT
    print(f"  reading {count:,} vectors at offset {start:,}...")
    with open(path, "rb") as f:
        raw = f.read(4)
        dim = struct.unpack("<i", raw)[0]
        f.seek(start * (4 + dim))
        vecs = []
        for i in range(count):
            raw = f.read(4)
            if not raw:
                break
            d = struct.unpack("<i", raw)[0]
            vecs.append(np.frombuffer(f.read(d), dtype=np.uint8).astype(np.float32))
            if (i + 1) % 500_000 == 0:
                print(f"    {i+1:,}...")
    arr = np.stack(vecs)
    print(f"  loaded {len(arr):,} vectors  dim={arr.shape[1]}")
    return arr


def compute_gt(index_vecs, queries):
    import faiss
    print("  computing ground truth with faiss...")
    flat = faiss.IndexFlatL2(index_vecs.shape[1])
    flat.add(np.ascontiguousarray(index_vecs, dtype=np.float32))
    _, nbrs = flat.search(np.ascontiguousarray(queries, dtype=np.float32), args.k)
    return nbrs.astype(np.int32)


def score_hardness(hnsw, queries, gt):
    n = len(queries)
    hardness = np.zeros(n, dtype=np.float32)
    comps = np.zeros(n, dtype=np.float32)
    bl_dists = np.zeros(n, dtype=np.float32)
    hnsw.set_ef(args.ef_hard)
    for i, q in enumerate(queries):
        lbls, _ = hnsw.knn_query(q.reshape(1, -1), k=args.k)
        s = hnswlib.get_last_query_stats()
        recall = len(set(lbls[0]) & set(gt[i])) / args.k
        hardness[i] = 1.0 - recall
        comps[i] = float(s["base_layer_distance_computations"])
        bl_dists[i] = float(s["base_layer_entry_distance"])
        if (i + 1) % 100_000 == 0:
            print(f"    {i+1:,}/{n:,}  mean_hardness={hardness[:i+1].mean():.4f}")
    return hardness, comps, bl_dists


def main():
    print(f"SIFT1B hardness dataset  n_index={args.n_index:,}  n_queries={args.n_queries:,}  query_source={args.query_source}\n")

    print("loading index vectors...")
    index_vecs = read_bvecs_range(args.bvecs_path, start=0, count=args.n_index)
    dim = index_vecs.shape[1]

    if args.query_source == "after_index":
        print("loading query vectors...")
        query_vecs = read_bvecs_range(args.bvecs_path, start=args.n_index, count=args.n_queries)
    else:
        print("sampling queries from index vectors...")
        idx = rng.choice(args.n_index, args.n_queries, replace=False)
        query_vecs = index_vecs[idx]

    print(f"\nbuilding HNSW index on {len(index_vecs):,} vectors...")
    hnsw = hnswlib.Index(space="l2", dim=dim)
    hnsw.init_index(max_elements=len(index_vecs), ef_construction=args.ef_construction, M=args.M)
    hnsw.add_items(index_vecs, np.arange(len(index_vecs), dtype=np.int32))
    hnsw.save_index(os.path.join(args.out_dir, "hnsw_index.bin"))
    print(f"  saved hnsw_index.bin")

    print("\ncomputing ground truth...")
    gt = compute_gt(index_vecs, query_vecs)

    print(f"\nscoring hardness at ef={args.ef_hard}...")
    hardness, comps, bl_dists = score_hardness(hnsw, query_vecs, gt)
    print(f"\nhardness: mean={hardness.mean():.4f}  std={hardness.std():.4f}  frac>0={(hardness > 0).mean():.3f}")

    order = np.argsort(hardness * 1e6 + comps, kind="stable")
    queries_sorted  = query_vecs[order]
    hardness_sorted = hardness[order]
    gt_sorted       = gt[order]

    print("\nsaving...")
    np.save(os.path.join(args.out_dir, "index_vectors.npy"),       index_vecs)
    np.save(os.path.join(args.out_dir, "queries_by_hardness.npy"), queries_sorted)
    np.save(os.path.join(args.out_dir, "hardness_scores.npy"),     hardness_sorted)
    np.save(os.path.join(args.out_dir, "ground_truth.npy"),        gt_sorted)

    pd.DataFrame({
        "hardness_rank":    np.arange(len(queries_sorted)),
        "hardness_score":   hardness_sorted,
        "base_dist_comps":  comps[order],
        "bl_entry_dist":    bl_dists[order],
        "recall_at_ef_hard": 1.0 - hardness_sorted,
    }).to_csv(os.path.join(args.out_dir, "metadata.csv"), index=False)

    with open(os.path.join(args.out_dir, "params.json"), "w") as f:
        json.dump({
            "source": "SIFT1B", "dim": int(dim),
            "n_index": int(len(index_vecs)), "n_queries": int(len(queries_sorted)),
            "query_source": args.query_source, "M": args.M,
            "ef_construction": args.ef_construction, "ef_hard": args.ef_hard,
            "k": args.k, "rng_seed": 42,
        }, f, indent=2)

    print(f"  index_vectors.npy       {index_vecs.shape}")
    print(f"  queries_by_hardness.npy {queries_sorted.shape}")
    print(f"  ground_truth.npy        {gt_sorted.shape}")
    print(f"\ndone — {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
