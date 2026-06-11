# No-adaptation baseline on the SIFT hardness dataset.
#
# Parallelised with ThreadPoolExecutor: each thread runs its own per-query
# loop and collects per-query stats independently. This is safe because:
#   - HNSW knn_query is thread-safe for concurrent reads.
#   - last_query_stats is thread_local in hnswalg.h, so each thread's stats
#     are independent even though they share the same index object.
#   - The GIL is released during C++ extension calls, giving real parallelism
#     on query computation without needing multiple index copies in memory.
#
# The index is loaded once and shared across all ef values and all threads.
# Results are saved to out_dir and loaded by
# experiment_sift_hardness_poolAndRewire.py for the recall delta.
#
# Usage:
#   python experiment_sift_hardness_baseline_parallel.py \
#       --data_dir data/sift_hardness \
#       --n_bins 10 --ef_sweep 10 20 50 100 200 500 --num_threads 16

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, ".."))

import hnswlib

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",    required=True)
parser.add_argument("--n_bins",      type=int, default=10)
parser.add_argument("--ef_sweep",    type=int, nargs="+", default=[10, 20, 50, 100, 200, 500])
parser.add_argument("--k",           type=int, default=10)
parser.add_argument("--num_threads", type=int, default=16,
                    help="parallel query threads; last_query_stats is thread_local so stats are safe")
parser.add_argument("--out_dir",     default="results_sift_hardness_baseline")
args = parser.parse_args()

k = args.k
os.makedirs(args.out_dir, exist_ok=True)


def load_dataset():
    print(f"loading dataset from {args.data_dir}...")
    queries = np.load(os.path.join(args.data_dir, "queries_by_hardness.npy"))
    gt      = np.load(os.path.join(args.data_dir, "ground_truth.npy"))
    scores  = np.load(os.path.join(args.data_dir, "hardness_scores.npy"))
    assert len(queries) == len(gt) == len(scores)
    print(f"  {len(queries):,} queries  dim={queries.shape[1]}")
    print(f"  hardness range [{scores.min():.3f}, {scores.max():.3f}]  mean={scores.mean():.3f}")
    return queries, gt, scores


def load_index(n_elements, dim):
    path = os.path.join(args.data_dir, "hnsw_index.bin")
    print(f"  loading index from {path}...")
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.load_index(path, max_elements=n_elements)
    return idx


def _query_chunk(idx, queries_chunk, gt_chunk, ef):
    """Run one thread's share of queries. Returns list of per-query result dicts.

    last_query_stats is thread_local in hnswalg.h so each thread's stats
    are independent even though they share the same index object.
    """
    idx.set_ef(ef)
    results = []
    for i, q in enumerate(queries_chunk):
        pred, _ = idx.knn_query(q.reshape(1, -1), k=k)
        s = hnswlib.get_last_query_stats()
        results.append({
            "recall":               len(set(pred[0]) & set(gt_chunk[i])) / k,
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


def run_bin_parallel(idx, queries, gt, ef):
    """Split queries across num_threads threads and merge results in order."""
    chunks = np.array_split(np.arange(len(queries)), args.num_threads)
    futures = {}
    with ThreadPoolExecutor(max_workers=args.num_threads) as pool:
        for chunk_idx in chunks:
            if len(chunk_idx) == 0:
                continue
            f = pool.submit(_query_chunk, idx,
                            queries[chunk_idx], gt[chunk_idx], ef)
            futures[f] = int(chunk_idx[0])

    # reassemble in original query order
    ordered = sorted([(start, f.result()) for f, start in futures.items()])
    return [r for _, chunk in ordered for r in chunk]


def save_params(n_index, n_queries_total, dim, bin_edges):
    rows = [
        {"param": "dataset",            "value": "SIFT1B-hardness"},
        {"param": "experiment_type",    "value": "sift_hardness_baseline_parallel"},
        {"param": "dim",                "value": dim},
        {"param": "n_index",            "value": n_index},
        {"param": "n_queries_total",    "value": n_queries_total},
        {"param": "n_bins",             "value": args.n_bins},
        {"param": "hardness_bin_edges", "value": str(list(bin_edges.round(4)))},
        {"param": "ef_sweep",           "value": str(args.ef_sweep)},
        {"param": "k",                  "value": args.k},
        {"param": "num_threads",        "value": args.num_threads},
        {"param": "drift_direction",    "value": "hardness_sorted_easy_to_hard"},
        {"param": "rng_seed",           "value": 42},
    ]
    path = os.path.join(args.out_dir, "params.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  saved {path}")


def save_summary(all_results, bin_edges):
    rows = []
    for (ef, b), results in sorted(all_results.items()):
        def avg(f): return float(np.nanmean([r[f] for r in results]))
        l1 = [r["layer_visits"][1] for r in results if len(r["layer_visits"]) > 1]
        rows.append({
            "ef_search":                 ef,
            "hardness_bin":              b,
            "hardness_bin_lo":           float(bin_edges[b]),
            "hardness_bin_hi":           float(bin_edges[b + 1]),
            "mean_recall":               avg("recall"),
            "mean_ep_distance":          avg("ep_dist"),
            "mean_bl_entry_distance":    avg("bl_entry_dist"),
            "mean_ul_dist_comps":        avg("ul_dist_comps"),
            "mean_layer1_visits":        float(np.mean(l1)) if l1 else float("nan"),
            "mean_base_visited":         avg("base_visited"),
            "mean_base_dist_comps":      avg("base_dist_comps"),
            "mean_candidates_remaining": avg("candidates_remaining"),
            "n_queries":                 len(results),
        })

    df = pd.DataFrame(rows).round(4)
    path = os.path.join(args.out_dir, "summary.csv")
    df.to_csv(path, index=False)
    print(f"  saved {path}")

    pivot = df.pivot(index="hardness_bin", columns="ef_search", values="mean_recall").round(3)
    print(f"\nno_adaptation recall@{k}:")
    print(pivot.to_string())


def save_per_query(all_results):
    rows = []
    for (ef, b), results in sorted(all_results.items()):
        for i, r in enumerate(results):
            row = {
                "strategy":             "no_adaptation",
                "ef_search":            ef,
                "hardness_bin":         b,
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
    path = os.path.join(args.out_dir, "per_query.csv")
    pd.DataFrame(rows).round(6).to_csv(path, index=False)
    print(f"  saved {path}")


def main():
    print(f"SIFT hardness baseline experiment (parallel)")
    print(f"n_bins={args.n_bins}  ef_sweep={args.ef_sweep}  k={k}  num_threads={args.num_threads}\n")

    queries, gt, scores = load_dataset()
    dim = queries.shape[1]

    bins = np.array_split(np.arange(len(queries)), args.n_bins)
    bin_edges = np.array(
        [scores[b[0]] for b in bins] + [scores[bins[-1][-1]]], dtype=np.float32
    )
    print(f"bin sizes: {[len(b) for b in bins]}")
    print(f"hardness bin edges: {bin_edges.round(3)}\n")

    eval_bins = bins[1:]

    params_path = os.path.join(args.data_dir, "params.json")
    with open(params_path) as f:
        n_index = json.load(f)["n_index"]

    all_results = {}

    print(f"{'='*60}")
    print("strategy: no_adaptation  (parallel threads, single index load)")
    print(f"{'='*60}")
    idx = load_index(n_index, dim)
    for ef in args.ef_sweep:
        cumulative_query = 0
        print(f"\n  ef={ef}")
        for b, bin_idx in enumerate(eval_bins, start=1):
            results = run_bin_parallel(idx, queries[bin_idx], gt[bin_idx], ef)
            all_results[(ef, b)] = results
            mean_r  = np.mean([r["recall"]       for r in results])
            mean_bl = np.mean([r["bl_entry_dist"] for r in results])
            cumulative_query += len(results)
            print(f"    bin={b}  hardness=[{bin_edges[b]:.3f},{bin_edges[b+1]:.3f}]  "
                  f"recall={mean_r:.4f}  bl_entry={mean_bl:.2f}  cumulative_q={cumulative_query}")

    print("\nsaving results...")
    save_params(n_index, len(queries), dim, bin_edges)
    save_summary(all_results, bin_edges)
    save_per_query(all_results)
    print(f"\ndone — {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
