# Pool routing experiment on synthetic data with continuous drift.
#
# Unlike the per-sigma version, the index and pool are built once and
# reused across all shift-sigma levels in order. The pool is seeded from
# sigma=0 warmup queries (the initial distribution), then eval batches for
# sigma=0,1,2,... are run sequentially on the same live index and pool.
# This mirrors real continuous drift: the pool represents the query
# distribution at deployment time, not a freshly-tuned oracle per shift.
#
# Strategies compared:
#   no_adaptation  — standard HNSW, no modification
#   pool_<N>       — pool of N nodes built from sigma=0 warmup queries,
#                    each eval query routed to nearest pool node
#
# For each ef value one full continuous-drift run is performed (fresh load).
#
# Usage:
#   python experiment_pool_routing_synthetic_continuous.py --mode small
#   python experiment_pool_routing_synthetic_continuous.py --mode large --n_init 5000000

import os
import sys
import argparse
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, ".."))

import hnswlib
from pool_router import PoolRouter

rng = np.random.default_rng(42)

configs = {
    "small": {
        "dim": 128,
        "n_init": 10_000,
        "n_clusters": 1,
        "cluster_std": 1.0,
        "shift_sigmas": [0, 1, 2, 3, 4, 6, 8, 10, 12, 16],
        "n_queries": 500,
        "n_warmup": 100,
        "M": 16,
        "ef_construction": 200,
        "ef_sweep": [10, 20, 50, 100, 200],
        "k": 10,
        "pool_sizes": [100, 500, 1000],
        "ef_build": 500,
        "k_build": 5,
        "gt_backend": "numpy",
    },
    "large": {
        "dim": 128,
        "n_init": 5_000_000,
        "n_clusters": 500,
        "cluster_std": 1.0,
        "cluster_spread": 100.0,
        "shift_sigmas": [0, 1, 2, 3, 4, 6, 8, 10, 12, 16],
        "n_queries": 1_000,
        "n_warmup": 200,
        "M": 16,
        "ef_construction": 200,
        "ef_sweep": [10, 20, 50, 100, 200, 500],
        "k": 10,
        "pool_sizes": [100, 500, 1000],
        "ef_build": 2000,
        "k_build": 5,
        "gt_backend": "faiss",
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["small", "large"], default="small")
parser.add_argument("--n_init", type=int, default=None)
parser.add_argument("--n_clusters", type=int, default=None)
parser.add_argument("--cluster_std", type=float, default=None)
parser.add_argument("--cluster_spread", type=float, default=None)
parser.add_argument("--shift_sigmas", type=float, nargs="+", default=None)
parser.add_argument("--n_queries", type=int, default=None)
parser.add_argument("--n_warmup", type=int, default=None)
parser.add_argument("--M", type=int, default=None)
parser.add_argument("--ef_construction", type=int, default=None)
parser.add_argument("--ef_sweep", type=int, nargs="+", default=None)
parser.add_argument("--k", type=int, default=None)
parser.add_argument("--pool_sizes", type=int, nargs="+", default=None)
parser.add_argument("--ef_build", type=int, default=None)
parser.add_argument("--k_build", type=int, default=None)
parser.add_argument("--out_dir", default=None)
args = parser.parse_args()

cfg = dict(configs[args.mode])
for key in ["n_init", "n_clusters", "cluster_std", "cluster_spread", "shift_sigmas",
            "n_queries", "n_warmup", "M", "ef_construction", "ef_sweep", "k",
            "pool_sizes", "ef_build", "k_build"]:
    val = getattr(args, key, None)
    if val is not None:
        cfg[key] = val

out_dir = args.out_dir or f"results_pool_routing_synthetic_continuous_{args.mode}"
os.makedirs(out_dir, exist_ok=True)

dim            = cfg["dim"]
n_init         = cfg["n_init"]
n_clusters     = cfg.get("n_clusters", 1)
cluster_std    = cfg["cluster_std"]
cluster_spread = cfg.get("cluster_spread", 1.0)
shift_sigmas   = cfg["shift_sigmas"]
n_queries      = cfg["n_queries"]
n_warmup       = cfg["n_warmup"]
M              = cfg["M"]
ef_construction = cfg["ef_construction"]
ef_sweep       = cfg["ef_sweep"]
k              = cfg["k"]
pool_sizes     = cfg["pool_sizes"]
ef_build       = cfg["ef_build"]
k_build        = cfg["k_build"]
gt_backend     = cfg["gt_backend"]


if n_clusters > 1:
    _centers = rng.standard_normal((n_clusters, dim)).astype(np.float32) * cluster_spread
else:
    _centers = None


def make_blob(n, center=None):
    offset = center if center is not None else np.zeros(dim, dtype=np.float32)
    if n_clusters == 1:
        return rng.standard_normal((n, dim)).astype(np.float32) * cluster_std + offset
    assignments = rng.integers(0, n_clusters, size=n)
    noise = rng.standard_normal((n, dim)).astype(np.float32) * cluster_std
    return (_centers[assignments] + noise + offset).astype(np.float32)


def compute_gt(data, queries):
    if gt_backend == "faiss":
        import faiss
        idx = faiss.IndexFlatL2(dim)
        idx.add(np.ascontiguousarray(data, dtype=np.float32))
        _, nbrs = idx.search(np.ascontiguousarray(queries, dtype=np.float32), k)
        return nbrs.astype(np.int32)
    results = []
    for i in range(0, len(queries), 50):
        batch = queries[i:i+50]
        d = np.sum((batch[:, None, :] - data[None, :, :]) ** 2, axis=-1)
        results.append(np.argsort(d, axis=1)[:, :k])
    return np.vstack(results).astype(np.int32)


def build_index(data):
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.init_index(max_elements=len(data), ef_construction=ef_construction, M=M)
    idx.add_items(data, np.arange(len(data), dtype=np.int32))
    return idx


def load_fresh(path):
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.load_index(path, max_elements=n_init)
    return idx


def collect_stats(pred, gt_row):
    s = hnswlib.get_last_query_stats()
    return {
        "recall":               len(set(pred) & set(gt_row)) / k,
        "ep_dist":              float(s["entry_point_distance"]),
        "bl_entry_dist":        float(s["base_layer_entry_distance"]),
        "ul_dist_comps":        int(s["upper_layer_distance_computations"]),
        "layer_visits":         list(s["layer_visit_counts"]),
        "base_visited":         int(s["base_layer_visited_count"]),
        "base_dist_comps":      int(s["base_layer_distance_computations"]),
        "candidates_remaining": int(s["candidates_remaining_at_termination"]),
        "lb_trace":             list(s["lowerbound_trace"]),
    }


def run_queries_plain(idx, queries, gt, ef):
    idx.set_ef(ef)
    results = []
    for i, q in enumerate(queries):
        pred, _ = idx.knn_query(q.reshape(1, -1), k=k)
        results.append(collect_stats(pred[0], gt[i]))
    return results


def run_queries_routed(router, queries, gt, ef):
    results = []
    for i, q in enumerate(queries):
        labels, _ = router.query(q, k=k, ef=ef, adapt=False)
        results.append(collect_stats(labels[0], gt[i]))
    return results


def save_params(direction):
    rows = [
        {"param": "mode",            "value": args.mode},
        {"param": "experiment_type", "value": "continuous_drift"},
        {"param": "dim",             "value": dim},
        {"param": "n_init",          "value": n_init},
        {"param": "n_clusters",      "value": n_clusters},
        {"param": "cluster_std",     "value": cluster_std},
        {"param": "cluster_spread",  "value": cluster_spread},
        {"param": "shift_sigmas",    "value": str(shift_sigmas)},
        {"param": "n_queries",       "value": n_queries},
        {"param": "n_warmup",        "value": n_warmup},
        {"param": "M",               "value": M},
        {"param": "ef_construction", "value": ef_construction},
        {"param": "ef_sweep",        "value": str(ef_sweep)},
        {"param": "k",               "value": k},
        {"param": "pool_sizes",      "value": str(pool_sizes)},
        {"param": "ef_build",        "value": ef_build},
        {"param": "k_build",         "value": k_build},
        {"param": "gt_backend",      "value": gt_backend},
        {"param": "rng_seed",        "value": 42},
    ]
    path = os.path.join(out_dir, "params.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  saved {path}")


def save_summary(all_results):
    rows = []
    for (strategy, ef, sigma), results in sorted(all_results.items()):
        def avg(field):
            return float(np.nanmean([r[field] for r in results]))
        l1 = [r["layer_visits"][1] for r in results if len(r["layer_visits"]) > 1]
        rows.append({
            "strategy":                  strategy,
            "ef_search":                 ef,
            "shift_sigma":               sigma,
            "shift_absolute":            round(sigma * cluster_std, 4),
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
    path = os.path.join(out_dir, "summary.csv")
    df.to_csv(path, index=False)
    print(f"  saved {path}")

    for strategy in df["strategy"].unique():
        sub = df[df["strategy"] == strategy]
        pivot = sub.pivot(index="shift_sigma", columns="ef_search",
                          values="mean_recall").round(3)
        print(f"\n{strategy} recall@{k}:")
        print(pivot.to_string())

    base = df[df["strategy"] == "no_adaptation"].set_index(
        ["shift_sigma", "ef_search"])["mean_recall"]
    for strategy in df["strategy"].unique():
        if strategy == "no_adaptation":
            continue
        adap = df[df["strategy"] == strategy].set_index(
            ["shift_sigma", "ef_search"])["mean_recall"]
        delta = (adap - base).round(3).unstack("ef_search")
        print(f"\nrecall delta ({strategy} - no_adaptation):")
        print(delta.to_string())


def save_per_query(all_results):
    rows = []
    for (strategy, ef, sigma), results in sorted(all_results.items()):
        for i, r in enumerate(results):
            row = {
                "strategy":             strategy,
                "ef_search":            ef,
                "shift_sigma":          sigma,
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
    path = os.path.join(out_dir, "per_query.csv")
    pd.DataFrame(rows).round(6).to_csv(path, index=False)
    print(f"  saved {path}")


def save_pool_log(pool_log):
    if not pool_log:
        return
    path = os.path.join(out_dir, "pool_build_log.csv")
    pd.DataFrame(pool_log).to_csv(path, index=False)
    print(f"  saved {path} ({len(pool_log)} entries)")


def main():
    if n_clusters > 1:
        print(f"mode={args.mode}  n_init={n_init:,}  n_clusters={n_clusters}  "
              f"cluster_std={cluster_std}  cluster_spread={cluster_spread}")
    else:
        print(f"mode={args.mode}  n_init={n_init:,}  cluster_std={cluster_std}")
    print(f"n_queries={n_queries}  n_warmup={n_warmup}  k={k}")
    print(f"pool_sizes={pool_sizes}  ef_build={ef_build}  k_build={k_build}")
    print(f"shift_sigmas={shift_sigmas}")
    print(f"ef_sweep={ef_sweep}")
    print(f"continuous drift mode: pool built once from sigma={shift_sigmas[0]} warmup, "
          f"reused across all sigmas\n")

    print("generating index data...")
    init_vecs = make_blob(n_init)
    direction = rng.standard_normal(dim).astype(np.float32)
    direction /= np.linalg.norm(direction)

    print("generating query sets...")
    query_sets = {}
    for sigma in shift_sigmas:
        center = direction * sigma * cluster_std
        query_sets[sigma] = make_blob(n_queries + n_warmup, center=center)
        print(f"  sigma={sigma:>5}  |center|={np.linalg.norm(center):.2f}")

    print(f"\nbuilding index on {n_init:,} vectors...")
    base_idx = build_index(init_vecs)
    index_path = os.path.join(out_dir, "base_index.bin")
    base_idx.save_index(index_path)
    print(f"  saved to {index_path}")

    print(f"\ncomputing ground truth ({gt_backend})...")
    ground_truths = {}
    for sigma in shift_sigmas:
        print(f"  sigma={sigma}")
        ground_truths[sigma] = compute_gt(init_vecs, query_sets[sigma])
    del init_vecs

    all_results = {}
    pool_log    = []

    # no_adaptation baseline: one fresh index per ef, queries fed in sigma order
    print(f"\n{'='*60}")
    print("strategy: no_adaptation  (continuous — one run per ef)")
    print(f"{'='*60}")
    for ef in ef_sweep:
        idx = load_fresh(index_path)
        cumulative_query = 0
        print(f"\n  ef={ef}")
        for sigma in shift_sigmas:
            eval_queries = query_sets[sigma][n_warmup:]
            eval_gt      = ground_truths[sigma][n_warmup:]
            results = run_queries_plain(idx, eval_queries, eval_gt, ef)
            all_results[("no_adaptation", ef, sigma)] = results
            mean_r  = np.mean([r["recall"]       for r in results])
            mean_bl = np.mean([r["bl_entry_dist"] for r in results])
            cumulative_query += len(results)
            print(f"    sigma={sigma:>5}  recall={mean_r:.4f}  bl_entry={mean_bl:.2f}"
                  f"  cumulative_q={cumulative_query}")

    # pool routing: pool built once from sigma=0, reused across all sigmas
    for pool_size in pool_sizes:
        strategy = f"pool_{pool_size}"
        print(f"\n{'='*60}")
        print(f"strategy: {strategy}  (continuous — pool from sigma={shift_sigmas[0]})")
        print(f"{'='*60}")

        for ef in ef_sweep:
            idx = load_fresh(index_path)
            router = PoolRouter(idx, pool_size=pool_size, ef_build=ef_build, k_build=k_build)

            warmup_queries = query_sets[shift_sigmas[0]][:n_warmup]
            print(f"\n  ef={ef}  building pool from {n_warmup} warmup queries at sigma={shift_sigmas[0]}...")
            n_collected = router.build_pool(warmup_queries)
            print(f"  pool built: {n_collected} nodes")

            pool_log.append({
                "strategy":            strategy,
                "ef":                  ef,
                "pool_size_requested": pool_size,
                "pool_size_achieved":  n_collected,
                "warmup_sigma":        shift_sigmas[0],
                "n_warmup":            n_warmup,
                "ef_build":            ef_build,
            })

            cumulative_query = 0
            for sigma in shift_sigmas:
                eval_queries = query_sets[sigma][n_warmup:]
                eval_gt      = ground_truths[sigma][n_warmup:]
                results = run_queries_routed(router, eval_queries, eval_gt, ef)
                all_results[(strategy, ef, sigma)] = results
                mean_r  = np.mean([r["recall"]       for r in results])
                mean_bl = np.mean([r["bl_entry_dist"] for r in results])
                cumulative_query += len(results)
                print(f"    sigma={sigma:>5}  recall={mean_r:.4f}  bl_entry={mean_bl:.2f}"
                      f"  cumulative_q={cumulative_query}")

    print("\nsaving results...")
    save_params(direction)
    save_summary(all_results)
    save_per_query(all_results)
    save_pool_log(pool_log)
    print(f"\ndone - results in {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
