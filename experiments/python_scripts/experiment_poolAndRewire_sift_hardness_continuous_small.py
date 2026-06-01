# PoolAndRewire adaptation experiment on the SIFT1B hardness dataset (small/fast variant).
#
# Identical to the full continuous experiment but caps each bin to
# --n_queries_per_bin queries (default 5000). This reduces the total query
# count by ~100x while keeping enough queries per bin for statistical significance:
#   SE(recall) = sqrt(p*(1-p)/n) ≈ 0.006 at n=5000, p=0.8
#   → detects deltas >= ~0.012 at 95% confidence.
#   (YFCC deltas were 0.015–0.07, so this is sufficient.)
#
# Strategies compared:
#   no_adaptation  — standard HNSW search
#   poolAndRewire  — PoolAndRewireController
#
# Usage:
#   python experiment_poolAndRewire_sift_hardness_continuous_small.py \
#       --data_dir data/sift_hardness \
#       --n_bins 10 --n_queries_per_bin 5000 --ef_sweep 10 50 100

import os
import sys
import json
import argparse
import time
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, ".."))

import hnswlib
from experiments.controllers.poolAndRewire import PoolAndRewireController, run_query_batch

rng = np.random.default_rng(42)

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",           required=True)
parser.add_argument("--n_bins",             type=int,   default=10,
                    help="number of hardness bins (bin 0 = warmup, rest = eval)")
parser.add_argument("--n_queries_per_bin",  type=int,   default=5000,
                    help="cap queries taken from each bin; 5000 gives SE(recall)~0.006")
parser.add_argument("--ef_sweep",           type=int,   nargs="+",
                    default=[10, 50, 100])
parser.add_argument("--k",                  type=int,   default=10)
parser.add_argument("--alpha",              type=float, default=1.1)
parser.add_argument("--max_layer",          type=int,   default=3)
parser.add_argument("--queries_per_rewire", type=int,   default=200)
parser.add_argument("--cooldown",           type=int,   default=50)
parser.add_argument("--window_size",        type=int,   default=200)
parser.add_argument("--max_pool_size",      type=int,   default=50)
parser.add_argument("--out_dir", default="results_poolAndRewire_sift_hardness_continuous_small")
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


def run_queries_plain(idx, queries, gt, ef):
    idx.set_ef(ef)
    results = []
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        pred, _ = idx.knn_query(q.reshape(1, -1), k=k)
        t_query_ms = (time.perf_counter() - t0) * 1000
        s = hnswlib.get_last_query_stats()
        results.append({
            "recall":               len(set(pred[0]) & set(gt[i])) / k,
            "ep_dist":              float(s["entry_point_distance"]),
            "bl_entry_dist":        float(s["base_layer_entry_distance"]),
            "ul_dist_comps":        int(s["upper_layer_distance_computations"]),
            "layer_visits":         list(s["layer_visit_counts"]),
            "base_visited":         int(s["base_layer_visited_count"]),
            "base_dist_comps":      int(s["base_layer_distance_computations"]),
            "candidates_remaining": int(s["candidates_remaining_at_termination"]),
            "lb_trace":             list(s["lowerbound_trace"]),
            "t_query_ms":           t_query_ms,
        })
    return results


def save_params(n_index, n_queries_total, dim, bin_edges):
    rows = [
        {"param": "dataset",                "value": "SIFT1B-hardness"},
        {"param": "experiment_type",        "value": "continuous_hardness_drift"},
        {"param": "dim",                    "value": dim},
        {"param": "n_index",                "value": n_index},
        {"param": "n_queries_total",        "value": n_queries_total},
        {"param": "n_bins",                 "value": args.n_bins},
        {"param": "hardness_bin_edges",     "value": str(list(bin_edges.round(4)))},
        {"param": "ef_sweep",               "value": str(args.ef_sweep)},
        {"param": "k",                      "value": args.k},
        {"param": "alpha",                  "value": args.alpha},
        {"param": "max_layer",              "value": args.max_layer},
        {"param": "queries_per_rewire",     "value": args.queries_per_rewire},
        {"param": "cooldown",               "value": args.cooldown},
        {"param": "window_size",            "value": args.window_size},
        {"param": "max_pool_size",          "value": args.max_pool_size},
        {"param": "n_queries_per_bin",       "value": args.n_queries_per_bin},
        {"param": "drift_direction",        "value": "hardness_sorted_easy_to_hard"},
        {"param": "rng_seed",               "value": 42},
    ]
    path = os.path.join(args.out_dir, "params.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  saved {path}")


def save_summary(all_results, bin_edges, adapt_log=None):
    rows = []
    for (strategy, ef, b), results in sorted(all_results.items()):
        def avg(f): return float(np.nanmean([r[f] for r in results]))
        l1 = [r["layer_visits"][1] for r in results if len(r["layer_visits"]) > 1]
        rows.append({
            "strategy":               strategy,
            "ef_search":              ef,
            "hardness_bin":           b,
            "hardness_bin_lo":        float(bin_edges[b]),
            "hardness_bin_hi":        float(bin_edges[b + 1]),
            "mean_recall":            avg("recall"),
            "mean_ep_distance":       avg("ep_dist"),
            "mean_bl_entry_distance": avg("bl_entry_dist"),
            "mean_ul_dist_comps":     avg("ul_dist_comps"),
            "mean_layer1_visits":     float(np.mean(l1)) if l1 else float("nan"),
            "mean_base_visited":      avg("base_visited"),
            "mean_base_dist_comps":   avg("base_dist_comps"),
            "mean_candidates_remaining": avg("candidates_remaining"),
            "n_queries":              len(results),
            "mean_t_query_ms":        avg("t_query_ms"),
            "mean_t_pool_scan_ms":    avg("t_pool_scan_ms") if "t_pool_scan_ms" in results[0] else float("nan"),
            "mean_t_pool_knn_ms":     avg("t_pool_knn_ms")  if "t_pool_knn_ms"  in results[0] else float("nan"),
            "mean_t_orig_knn_ms":     avg("t_orig_knn_ms")  if "t_orig_knn_ms"  in results[0] else float("nan"),
            "mean_t_adapt_ms":        avg("t_adapt_ms")     if "t_adapt_ms"     in results[0] else float("nan"),
        })

    df = pd.DataFrame(rows).round(4)
    path = os.path.join(args.out_dir, "summary.csv")
    df.to_csv(path, index=False)
    print(f"  saved {path}")

    for strategy in df["strategy"].unique():
        sub = df[df["strategy"] == strategy]
        pivot = sub.pivot(index="hardness_bin", columns="ef_search",
                          values="mean_recall").round(3)
        print(f"\n{strategy} recall@{k}:")
        print(pivot.to_string())

    base = df[df["strategy"] == "no_adaptation"].set_index(
        ["hardness_bin", "ef_search"])["mean_recall"]
    for strategy in df["strategy"].unique():
        if strategy == "no_adaptation":
            continue
        adap  = df[df["strategy"] == strategy].set_index(
            ["hardness_bin", "ef_search"])["mean_recall"]
        delta = (adap - base).round(3).unstack("ef_search")
        print(f"\nrecall delta ({strategy} - no_adaptation):")
        print(delta.to_string())

    # Overhead summary: mean query latency and overhead ratio per ef.
    print("\n--- query latency overhead ---")
    base_t = df[df["strategy"] == "no_adaptation"].groupby("ef_search")["mean_t_query_ms"].mean()
    for strategy in df["strategy"].unique():
        if strategy == "no_adaptation":
            continue
        adap_t = df[df["strategy"] == strategy].groupby("ef_search")["mean_t_query_ms"].mean()
        overhead = (adap_t / base_t).round(2)
        print(f"\n{strategy} vs no_adaptation (mean ms / overhead ratio):")
        for ef in sorted(base_t.index):
            print(f"  ef={ef:4d}  no_adapt={base_t[ef]:.3f}ms  "
                  f"{strategy}={adap_t[ef]:.3f}ms  overhead={overhead[ef]:.2f}x")


def save_per_query(all_results):
    rows = []
    for (strategy, ef, b), results in sorted(all_results.items()):
        for i, r in enumerate(results):
            row = {
                "strategy":             strategy,
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
                "t_query_ms":           r.get("t_query_ms", float("nan")),
                "t_pool_scan_ms":       r.get("t_pool_scan_ms", float("nan")),
                "t_pool_knn_ms":        r.get("t_pool_knn_ms", float("nan")),
                "t_orig_knn_ms":        r.get("t_orig_knn_ms", float("nan")),
                "t_adapt_ms":           r.get("t_adapt_ms", float("nan")),
            }
            for layer, visits in enumerate(r["layer_visits"]):
                row[f"layer{layer}_visits"] = visits
            rows.append(row)
    path = os.path.join(args.out_dir, "per_query.csv")
    pd.DataFrame(rows).round(6).to_csv(path, index=False)
    print(f"  saved {path}")


def save_adapt_log(adapt_log):
    if not adapt_log:
        return
    path = os.path.join(args.out_dir, "adapt_log.csv")
    pd.DataFrame(adapt_log).to_csv(path, index=False)
    print(f"  saved {path} ({len(adapt_log)} entries)")


def main():
    print(f"SIFT hardness poolAndRewire continuous-drift experiment")
    print(f"n_bins={args.n_bins}  ef_sweep={args.ef_sweep}  k={k}")
    print(f"alpha={args.alpha}  max_layer={args.max_layer}  cooldown={args.cooldown}  "
          f"window_size={args.window_size}  max_pool_size={args.max_pool_size}")
    print(f"continuous drift mode: index/controller persists across hardness bins\n")

    queries, gt, scores = load_dataset()
    dim = queries.shape[1]

    bins = np.array_split(np.arange(len(queries)), args.n_bins)
    bin_edges = np.array(
        [scores[b[0]] for b in bins] + [scores[bins[-1][-1]]], dtype=np.float32
    )
    print(f"bin sizes: {[len(b) for b in bins]}")
    print(f"hardness bin edges: {bin_edges.round(3)}")
    print(f"capping each bin to {args.n_queries_per_bin} queries\n")

    # Take the first n_queries_per_bin from each bin; queries are hardness-sorted
    # so this preserves the hardness range of each bin.
    bins = [b[:args.n_queries_per_bin] for b in bins]

    warmup_queries = queries[bins[0]]
    eval_bins      = bins[1:]

    params_path = os.path.join(args.data_dir, "params.json")
    if os.path.exists(params_path):
        with open(params_path) as f:
            n_index = json.load(f)["n_index"]
    else:
        n_index = int(input("n_index not found in params.json, enter manually: "))

    all_results = {}
    adapt_log   = []

    # no_adaptation baseline: one fresh index per ef, bins fed in hardness order
    print(f"{'='*60}")
    print("strategy: no_adaptation  (continuous — one run per ef)")
    print(f"{'='*60}")
    for ef in args.ef_sweep:
        idx = load_index(n_index, dim)
        cumulative_query = 0
        print(f"\n  ef={ef}")
        for b, bin_idx in enumerate(eval_bins, start=1):
            results = run_queries_plain(idx, queries[bin_idx], gt[bin_idx], ef)
            all_results[("no_adaptation", ef, b)] = results
            mean_r  = np.mean([r["recall"]       for r in results])
            mean_bl = np.mean([r["bl_entry_dist"] for r in results])
            cumulative_query += len(results)
            print(f"    bin={b}  hardness=[{bin_edges[b]:.3f},{bin_edges[b+1]:.3f}]  "
                  f"recall={mean_r:.4f}  bl_entry={mean_bl:.2f}  cumulative_q={cumulative_query}")

    # poolAndRewire: one fresh index per ef, controller persists across bins
    print(f"\n{'='*60}")
    print("strategy: poolAndRewire  (continuous — one run per ef)")
    print(f"{'='*60}")
    for ef in args.ef_sweep:
        idx  = load_index(n_index, dim)
        ctrl = PoolAndRewireController(
            idx,
            reference_queries=warmup_queries,
            window_size=args.window_size,
            alpha=args.alpha,
            max_layer=args.max_layer,
            queries_per_rewire=args.queries_per_rewire,
            cooldown=args.cooldown,
            max_pool_size=args.max_pool_size,
        )
        cumulative_query = 0
        print(f"\n  ef={ef}  (controller initialized on warmup bin)")
        for b, bin_idx in enumerate(eval_bins, start=1):
            results = run_query_batch(idx, queries[bin_idx], gt[bin_idx],
                                      k=k, ef=ef, controller=ctrl, use_pool=True)
            all_results[("poolAndRewire", ef, b)] = results
            mean_r  = np.mean([r["recall"]       for r in results])
            mean_bl = np.mean([r["bl_entry_dist"] for r in results])
            cumulative_query += len(results)
            print(f"    bin={b}  hardness=[{bin_edges[b]:.3f},{bin_edges[b+1]:.3f}]  "
                  f"recall={mean_r:.4f}  bl_entry={mean_bl:.2f}  "
                  f"updates={ctrl.update_count}  cumulative_q={cumulative_query}")
        n_eval_queries = sum(len(b) for b in eval_bins)
        amortized_ms = ctrl.total_adapt_time_ms / max(n_eval_queries, 1)
        print(f"  adapt events={ctrl.update_count}  total_adapt={ctrl.total_adapt_time_ms:.0f}ms  "
              f"amortized={amortized_ms:.4f}ms/query")
        for entry in ctrl.update_log:
            entry["ef"] = ef
            adapt_log.append(entry)

    print("\nsaving results...")
    save_params(n_index, len(queries), dim, bin_edges)
    save_summary(all_results, bin_edges)
    save_per_query(all_results)
    save_adapt_log(adapt_log)
    print(f"\ndone — {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
