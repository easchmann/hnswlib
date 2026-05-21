# Distribution shift experiment on real SIFT-1B descriptors.
#
# Split design:
#   - load all N vectors and project onto PC1 (first principal component)
#   - build the index on the bottom 50% by PC1 value (low PC1 = one mode
#     of the SIFT descriptor distribution)
#   - generate query sets at increasing quantiles of the top 50%:
#     level 0 samples just above the median (slight overlap with index),
#     level n samples from the top quantile (far OOD)
#   - this gives a hard geometric split combined with a continuous drift curve


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
                    help="path to bigann_base_10M.bvecs (or .fvecs/.hdf5)")
parser.add_argument("--n_total", type=int, default=10_000_000,
                    help="total vectors to load before splitting (default: 10M)")
parser.add_argument("--n_queries", type=int, default=1_000,
                    help="queries per shift level (default: 1000)")
parser.add_argument("--n_shift_levels", type=int, default=10,
                    help="number of shift levels across the top 50 percent (default: 10)")
parser.add_argument("--ef_sweep", type=int, nargs="+",
                    default=[10, 20, 50, 100, 200, 500])
parser.add_argument("--M", type=int, default=16)
parser.add_argument("--ef_construction", type=int, default=200)
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--out_dir", default=None)
args = parser.parse_args()

out_dir = args.out_dir or f"results_sift_drift_{args.n_total // 1_000_000}M_pca"
os.makedirs(out_dir, exist_ok=True)

dim = 128


# --- loading -----------------------------------------------------------------

def load_bvecs(path, n=None):
    # bvecs: each record is [dim:int32][values:uint8 x dim]
    with open(path, "rb") as f:
        d = struct.unpack("i", f.read(4))[0]
        assert d == 128, f"expected dim=128, got {d}"
        f.seek(0)
        record_size = 4 + d
        raw = f.read(n * record_size) if n else f.read()
        data = np.frombuffer(raw, dtype=np.uint8).reshape(-1, record_size)
        return data[:, 4:].astype(np.float32)



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



# --- PC1 split ---------------------------------------------------------------

def pc1_split(data, n_shift_levels, n_queries):
    # fit PCA on a random sample — no need to use all 10M for this
    sample_idx = rng.choice(len(data), min(200_000, len(data)), replace=False)
    pca = PCA(n_components=1)
    pca.fit(data[sample_idx])
    direction = pca.components_[0].astype(np.float32)
    direction /= np.linalg.norm(direction)

    projections = data @ direction
    median = np.median(projections)
    print(f"  PC1 range: [{projections.min():.1f}, {projections.max():.1f}]")
    print(f"  PC1 median: {median:.1f}")

    # index vectors: bottom 50% by PC1
    index_mask = projections <= median
    index_vecs = data[index_mask]
    print(f"  index vectors (PC1 <= median): {len(index_vecs):,}")

    # query pool: top 50% by PC1
    query_pool = data[~index_mask]
    query_projections = projections[~index_mask]
    print(f"  query pool (PC1 > median):     {len(query_pool):,}")

    # divide the query pool into n_shift_levels quantile bands
    # level 0 = just above median (least shifted)
    # level n = top quantile (most shifted, furthest from index)
    quantile_edges = np.linspace(0, 1, n_shift_levels + 2)  # n+1 bands
    q_values = np.quantile(query_projections, quantile_edges)

    query_sets = {}
    shift_labels = []
    for i in range(n_shift_levels + 1):
        lo, hi = q_values[i], q_values[i + 1]
        band_mask = (query_projections >= lo) & (query_projections < hi)
        band_vecs = query_pool[band_mask]

        if len(band_vecs) < n_queries:
            print(f"  warning: level {i} has only {len(band_vecs)} vectors, "
                  f"sampling with replacement")
            chosen = rng.choice(len(band_vecs), n_queries, replace=True)
        else:
            chosen = rng.choice(len(band_vecs), n_queries, replace=False)

        # use the midpoint PC1 value as the shift label
        shift_label = round(float((lo + hi) / 2), 2)
        query_sets[shift_label] = band_vecs[chosen]
        shift_labels.append(shift_label)
        print(f"  level {i:>2}  PC1=[{lo:.1f}, {hi:.1f}]  "
              f"label={shift_label:.2f}  n_band={len(band_vecs):,}")

    return index_vecs, query_sets, direction


# --- ground truth ------------------------------------------------------------

def compute_gt(index_vecs, queries):
    
    import faiss

    index = faiss.IndexFlatL2(dim)
    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        print(f"  faiss GPU")
    else:
        print(f"  faiss CPU")

    index.add(np.ascontiguousarray(index_vecs, dtype=np.float32))
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
        {"param": "n_total",         "value": args.n_total},
        {"param": "n_queries",        "value": args.n_queries},
        {"param": "n_shift_levels",   "value": args.n_shift_levels},
        {"param": "ef_sweep",         "value": str(args.ef_sweep)},
        {"param": "M",                "value": args.M},
        {"param": "ef_construction",  "value": args.ef_construction},
        {"param": "k",                "value": args.k},
        {"param": "dim",              "value": dim},
        {"param": "dataset",          "value": "SIFT-1B"},
        {"param": "split_method",     "value": "PC1_quantile_bands"},
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
    print("\nRecall@k (rows=PC1 shift label, cols=ef):")
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
    print(f"SIFT drift experiment (PC1 quantile split)")
    print(f"n_total={args.n_total:,}  n_queries={args.n_queries}  k={args.k}")
    print(f"n_shift_levels={args.n_shift_levels}  ef_sweep={args.ef_sweep}\n")

    print(f"loading {args.n_total:,} SIFT vectors...")
    data = load_sift(args.sift_path, args.n_total)
    assert data.shape[1] == dim, f"expected dim=128, got {data.shape[1]}"
    print(f"  loaded: {data.shape}  range=[{data.min():.1f}, {data.max():.1f}]")

    # split by PC1: index on bottom half, queries from top half quantile bands
    print("\nsplitting by PC1...")
    index_vecs, query_sets, direction = pc1_split(
        data, args.n_shift_levels, args.n_queries
    )

    # free the full data array — we only need index_vecs and query_sets now
    del data

    print(f"\nbuilding index on {len(index_vecs):,} vectors...")
    idx = build_index(index_vecs)
    print(f"  done  (M={args.M}, ef_construction={args.ef_construction})")

    # ground truth against index vectors only
    print(f"\ncomputing ground truth (faiss)...")
    ground_truths = {}
    for shift, queries in sorted(query_sets.items()):
        ground_truths[shift] = compute_gt(index_vecs, queries)
    print("  done")

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

   
    save_params(os.path.join(out_dir, "params.csv"))
    save_summary(all_results, os.path.join(out_dir, "summary.csv"))
    save_per_query(all_results, os.path.join(out_dir, "per_query.csv"))
    save_lb_traces(all_results, os.path.join(out_dir, "lb_traces.csv"))
    print(f"\ndone — results in {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()