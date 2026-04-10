# Distribution shift experiment on SIFT-1B dataset.
#
# The idea: build an HNSW index on the first N vectors of SIFT-1B, 
# then query it with vectors projected progressively further from the construction distribution along the first principal component. 
# This creates a drift similar to the synthetic continuous drift experiment but on real 128-dim image descriptors.
#
# Why PCA rather than a random direction? The first PC of SIFT captures the dominant axis of variance in the descriptor space.
# vectors with high PC1 scores are genuinely different from vectors with low PC1 scores in terms of
# the underlying image patches they represent. 
# Shifting along PC1 is therefore semantically meaningful, not just geometric.
#
# Setup:
#   -> downloaded SIFT-1B from http://corpus-texmex.irisa.fr/ (bigann_base.bvecs etc.)
#
# data saved in summary.csv, per_query.csv, lb_traces.csv, params.csv
# plot_continuous_drift.py to generate figures (same format as synthetic).
#
#   python experiment_sift_drift.py --sift_path /path/to/sift1b --n_index 10000000

import os
import sys
import argparse
import struct
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import hnswlib

rng = np.random.default_rng(42)

parser = argparse.ArgumentParser()
parser.add_argument("--sift_path", required=True,
                    help="directory containing bigann_base.bvecs (or .fvecs/.hdf5)")
parser.add_argument("--n_index", type=int, default=5_000_000,
                    help="number of vectors to build the index on (default: 5M)")
parser.add_argument("--n_queries", type=int, default=1_000,
                    help="queries per shift level (default: 1000)")
parser.add_argument("--n_shift_levels", type=int, default=10,
                    help="number of shift levels along PC1 (default: 10)")
parser.add_argument("--ef_sweep", type=int, nargs="+",
                    default=[10, 20, 50, 100, 200, 500])
parser.add_argument("--M", type=int, default=16)
parser.add_argument("--ef_construction", type=int, default=200)
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--out_dir", default=None)
args = parser.parse_args()

out_dir = args.out_dir or f"results_sift_drift_{args.n_index // 1_000_000}M_pastPC1"
os.makedirs(out_dir, exist_ok=True)

dim = 128


# --- loading SIFT data -------------------------------------------------------

def load_bvecs(path, n=None):
    # bvecs format: each vector is [dim (4 bytes int)] + [dim bytes uint8]
    with open(path, "rb") as f:
        d = struct.unpack("i", f.read(4))[0]
        assert d == 128, f"expected dim=128, got {d}"
        f.seek(0)
        record_size = 4 + d  # 4 bytes for dim + d bytes for values
        if n is not None:
            data = np.frombuffer(f.read(n * record_size), dtype=np.uint8)
        else:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        # reshape and strip the 4-byte dim header from each record
        data = data.reshape(-1, record_size)[:, 4:].astype(np.float32)
    return data



def load_sift(path, n):
    if os.path.isdir(path):
        # texmex directory — look for bigann_base.bvecs
        vec_path = os.path.join(vec_path, "bigann_base.bvecs")
       
        if os.path.exists(vec_path):
            print(f"  loading from {vec_path}")
            return load_bvecs(path, n)
    
    elif path.endswith(".bvecs"):
        return load_bvecs(path, n)
    else:
        sys.exit(f"unrecognised file format: {path}")


# --- PCA-based shift ---------------------------------------------------------

def compute_pca_direction(data, n_sample=100_000):
    # fit PCA on a sample to keep memory reasonable
    sample = data[rng.choice(len(data), min(n_sample, len(data)), replace=False)]
    pca = PCA(n_components=1)
    pca.fit(sample)
    direction = pca.components_[0].astype(np.float32)  # shape (128,)
    direction /= np.linalg.norm(direction)
    return direction, pca



def make_shifted_queries(data, direction, pca, n_queries, n_levels):
    # project all index vectors onto PC1 to find the range
    projections = data @ direction  # shape (n_index,)
    pc1_min, pc1_max = projections.min(), projections.max()
    pc1_std = projections.std()

    print(f"  PC1 range: [{pc1_min:.1f}, {pc1_max:.1f}]  std={pc1_std:.1f}")

    # shift from data centre to 2x the data radius
    shift_values = np.linspace(0, pc1_max * 2, n_levels + 1)

    query_sets = {}
    for shift in shift_values:
        # sample queries from the region of the data near this PC1 value
        # this creates genuine out-of-distribution queries, not just
        # translated vectors, which is more realistic than the synthetic case
        center_vec = direction * shift
        # add Gaussian noise in the full space around the target PC1 value
        noise = rng.standard_normal((n_queries, dim)).astype(np.float32)
        # scale noise to be small relative to the inter-vector spacing
        noise *= pc1_std * 0.5
        queries = center_vec + noise
        query_sets[round(float(shift), 2)] = queries.astype(np.float32)

    return query_sets, shift_values, pc1_std


# --- ground truth ------------------------------------------------------------

def compute_gt(data, queries):
    import faiss

    index = faiss.IndexFlatL2(dim)
    if faiss.get_num_gpus() > 0:
        print(f"  faiss: {faiss.get_num_gpus()} GPU(s)")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    else:
        print("  faiss: no GPU, using CPU")

    index.add(np.ascontiguousarray(data, dtype=np.float32))
    _, neighbours = index.search(
        np.ascontiguousarray(queries, dtype=np.float32), args.k
    )
    return neighbours.astype(np.int32)


# --- index + querying --------------------------------------------------------

def build_index(data):
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.init_index(
        max_elements=len(data),
        ef_construction=args.ef_construction,
        M=args.M
    )
    idx.add_items(data, np.arange(len(data), dtype=np.int32))
    return idx



def recall(predicted, ground_truth):
    hits = sum(len(set(p) & set(g)) for p, g in zip(predicted, ground_truth))
    return hits / (len(ground_truth) * ground_truth.shape[1])


def query_index(idx, queries, gt):
    results = []
    for i, q in enumerate(queries):
        pred, _ = idx.knn_query(q.reshape(1, -1), k=args.k)
        s = hnswlib.get_last_query_stats()
        results.append({
            "recall":               recall(pred, gt[i:i+1]),
            "ep_dist":              float(s["entry_point_distance"]),
            "bl_entry_dist":        float(s["base_layer_entry_distance"]),
            "ul_dist_comps":        int(s["upper_layer_distance_computations"]),
            "layer_visits":         list(s["layer_visit_counts"]),
            "base_visited":         int(s["base_layer_visited_count"]),
            "base_dist_comps":      int(s["base_layer_distance_computations"]),
            "candidates_remaining": int(s["candidates_remaining_at_termination"]),
            "lb_trace":             list(s["lowerbound_trace"]),
        })
    return results


# --- saving ------------------------------------------------------------------

def save_params(path):
    rows = [
        {"param": "n_index",         "value": args.n_index},
        {"param": "n_queries",        "value": args.n_queries},
        {"param": "n_shift_levels",   "value": args.n_shift_levels},
        {"param": "ef_sweep",         "value": str(args.ef_sweep)},
        {"param": "M",                "value": args.M},
        {"param": "ef_construction",  "value": args.ef_construction},
        {"param": "k",                "value": args.k},
        {"param": "dim",              "value": dim},
        {"param": "dataset",          "value": "SIFT-1B"},
        {"param": "split_method",     "value": "PCA_PC1"},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  saved {path}")


def save_summary(all_results, path):
    rows = []
    for (shift, ef), results in sorted(all_results.items()):
        def avg(field):
            return float(np.nanmean([r[field] for r in results]))
        l1 = [r["layer_visits"][1] for r in results if len(r["layer_visits"]) > 1]
        rows.append({
            "shift":                        shift,
            "ef_search":                    ef,
            "mean_recall":                  avg("recall"),
            "mean_ep_distance":             avg("ep_dist"),
            "mean_bl_entry_distance":       avg("bl_entry_dist"),
            "mean_ul_dist_comps":           avg("ul_dist_comps"),
            "mean_layer1_visits":           float(np.mean(l1)) if l1 else float("nan"),
            "mean_base_visited":            avg("base_visited"),
            "mean_base_dist_comps":         avg("base_dist_comps"),
            "mean_candidates_remaining":    avg("candidates_remaining"),
            "n_queries":                    len(results),
        })

    df = pd.DataFrame(rows).round(4)
    df.to_csv(path, index=False)
    print(f"  saved {path}")

    pivot = df.pivot(index="shift", columns="ef_search",
                     values="mean_recall").round(3)
    print("\nRecall@k (rows=shift, cols=ef):")
    print(pivot.to_string())


def save_per_query(all_results, path):
    rows = []
    for (shift, ef), results in sorted(all_results.items()):
        for i, r in enumerate(results):
            row = {
                "shift":                shift,
                "ef_search":            ef,
                "query_id":             i,
                "recall":               r["recall"],
                "ep_dist":              r["ep_dist"],
                "bl_entry_dist":        r["bl_entry_dist"],
                "ul_dist_comps":        r["ul_dist_comps"],
                "base_visited":         r["base_visited"],
                "base_dist_comps":      r["base_dist_comps"],
                "candidates_remaining": r["candidates_remaining"],
                "lb_trace_final":       r["lb_trace"][-1] if r["lb_trace"] else float("nan"),
                "lb_trace_len":         len(r["lb_trace"]),
            }
            for layer, visits in enumerate(r["layer_visits"]):
                row[f"layer{layer}_visits"] = visits
            rows.append(row)
    pd.DataFrame(rows).round(6).to_csv(path, index=False)
    print(f"  saved {path}")


def save_lb_traces(all_results, path):
    rows = []
    for (shift, ef), results in sorted(all_results.items()):
        traces = [r["lb_trace"] for r in results if r["lb_trace"]]
        if not traces:
            continue
        min_len = min(len(t) for t in traces)
        mean_trace = np.mean([t[:min_len] for t in traces], axis=0)
        for iteration, value in enumerate(mean_trace):
            rows.append({
                "shift":           shift,
                "ef_search":       ef,
                "iteration":       iteration,
                "mean_lowerbound": round(float(value), 6),
            })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  saved {path}")


# --- main --------------------------------------------------------------------

def main():
    print(f"SIFT drift experiment")
    print(f"n_index={args.n_index:,}  n_queries={args.n_queries}  k={args.k}")
    print(f"ef_sweep={args.ef_sweep}  M={args.M}  ef_construction={args.ef_construction}\n")

    # load SIFT vectors
    print(f"loading {args.n_index:,} SIFT vectors...")
    data = load_sift(args.sift_path, args.n_index)
    assert data.shape == (args.n_index, dim), f"unexpected shape {data.shape}"
    print(f"  loaded: {data.shape}  dtype={data.dtype}")
    print(f"  value range: [{data.min():.1f}, {data.max():.1f}]")

    # fit PCA on a sample and compute shift direction
    print("\nfitting PCA to find shift direction...")
    direction, pca = compute_pca_direction(data)
    print(f"  PC1 direction norm: {np.linalg.norm(direction):.4f}")

    # generate query sets at different points along PC1
    print(f"\ngenerating {args.n_shift_levels + 1} query sets along PC1...")
    query_sets, shift_values, pc1_std = make_shifted_queries(
        data, direction, pca, args.n_queries, args.n_shift_levels
    )
    for shift, qs in sorted(query_sets.items()):
        proj = float(qs @ direction.reshape(-1, 1)).item() if args.n_queries == 1 \
               else float((qs @ direction).mean())
        print(f"  shift={shift:>8.2f}  mean PC1 projection: {proj:.2f}")

    # build index
    print(f"\nbuilding index on {args.n_index:,} vectors...")
    idx = build_index(data)
    print(f"  done  (M={args.M}, ef_construction={args.ef_construction})")

    # ground truth — computed once, reused across all ef values
    print(f"\ncomputing ground truth (faiss)...")
    ground_truths = {}
    for shift, queries in sorted(query_sets.items()):
        ground_truths[shift] = compute_gt(data, queries)
    print("  done")

    # query sweep
    print(f"\nquerying: {len(query_sets)} shift levels x {len(args.ef_sweep)} ef values")
    all_results = {}

    for ef in args.ef_sweep:
        idx.set_ef(ef)
        print(f"\nef={ef}")
        for shift, queries in sorted(query_sets.items()):
            results = query_index(idx, queries, ground_truths[shift])
            all_results[(shift, ef)] = results
            mean_r  = np.mean([r["recall"] for r in results])
            mean_ep = np.nanmean([r["ep_dist"] for r in results])
            mean_bl = np.nanmean([r["bl_entry_dist"] for r in results])
            print(f"  shift={shift:>8.2f}  recall={mean_r:.4f}  "
                  f"ep={mean_ep:.1f}  bl_entry={mean_bl:.1f}")

    # save
    save_params(os.path.join(out_dir, "params.csv"))
    save_summary(all_results, os.path.join(out_dir, "summary.csv"))
    save_per_query(all_results, os.path.join(out_dir, "per_query.csv"))
    save_lb_traces(all_results, os.path.join(out_dir, "lb_traces.csv"))
    print(f"\ndone — results in {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()