# Thesis adaptation strategy experiment on synthetic continuous drift.
#
# Strategies compared:
#   no_adaptation  — standard HNSW, no modification
#   thesis         — ThesisAdaptationController: query-driven upper-layer
#                    rewiring + bounded entry-point pool with LRU eviction
#
# For each shift_sigma level a fresh copy of the index is loaded.
# A warmup phase (n_warmup queries) calibrates the baseline bl_entry_dist
# and seeds the first adaptation step if drift is already present.
# Eval queries are then run with online adaptation active.
#
# Usage:
#   python experiment_thesis_adaptation_synthetic.py --mode small
#   python experiment_thesis_adaptation_synthetic.py --mode large --n_init 5000000

import os
import sys
import argparse
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, ".."))

import hnswlib
from poolAndRewire import PoolAndRewireController, run_query_batch

rng = np.random.default_rng(42)

configs = {
    "small": {
        "dim": 128,
        "n_init": 10_000,
        "n_clusters": 1,
        "cluster_std": 1.0,
        "shift_sigmas": [0, 1, 2, 3, 4, 6, 8, 10, 12, 16],
        "n_queries": 500,
        "n_warmup": 200,
        "M": 16,
        "ef_construction": 200,
        "ef_sweep": [10, 20, 50, 100, 200, 500],
        "k": 10,
        "gt_backend": "numpy",
        # controller params
        "alpha": 1.5,
        "max_layer": 1,
        "queries_per_rewire": 10,
        "cooldown": 50,
        "window_size": 100,
        "max_pool_size": 5,
        "ef_build": 500,
    },
    "large": {
        "dim": 128,
        "n_init": 5_000_000,
        "n_clusters": 1,
        "cluster_std": 1.0,
        "shift_sigmas": [0, 1, 2, 3, 4, 6, 8, 10, 12, 16],
        "n_queries": 1_000,
        "n_warmup": 200,
        "M": 16,
        "ef_construction": 200,
        "ef_sweep": [10, 20, 50, 100, 200, 500],
        "k": 10,
        "gt_backend": "faiss",
        "alpha": 1.5,
        "max_layer": 2,
        "queries_per_rewire": 20,
        "cooldown": 100,
        "window_size": 200,
        "max_pool_size": 10,
        "ef_build": 200,
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["small", "large"], default="small")
parser.add_argument("--n_init", type=int, default=None)
parser.add_argument("--shift_sigmas", type=float, nargs="+", default=None)
parser.add_argument("--n_queries", type=int, default=None)
parser.add_argument("--n_warmup", type=int, default=None)
parser.add_argument("--ef_sweep", type=int, nargs="+", default=None)
parser.add_argument("--alpha", type=float, default=None)
parser.add_argument("--max_layer", type=int, default=None)
parser.add_argument("--max_pool_size", type=int, default=None)
parser.add_argument("--cooldown", type=int, default=None)
parser.add_argument("--window_size", type=int, default=None)
parser.add_argument("--out_dir", default=None)
args = parser.parse_args()

cfg = dict(configs[args.mode])
for key in ["n_init", "shift_sigmas", "n_queries", "n_warmup", "ef_sweep",
            "alpha", "max_layer", "max_pool_size", "cooldown", "window_size"]:
    val = getattr(args, key, None)
    if val is not None:
        cfg[key] = val

out_dir = args.out_dir or f"results_thesis_synthetic_{args.mode}"
os.makedirs(out_dir, exist_ok=True)

dim            = cfg["dim"]
n_init         = cfg["n_init"]
n_clusters     = cfg.get("n_clusters", 1)
cluster_std    = cfg["cluster_std"]
shift_sigmas   = cfg["shift_sigmas"]
n_queries      = cfg["n_queries"]
n_warmup       = cfg["n_warmup"]
M              = cfg["M"]
ef_construction = cfg["ef_construction"]
ef_sweep       = cfg["ef_sweep"]
k              = cfg["k"]
gt_backend     = cfg["gt_backend"]
alpha          = cfg["alpha"]
max_layer      = cfg["max_layer"]
queries_per_rewire = cfg["queries_per_rewire"]
cooldown       = cfg["cooldown"]
window_size    = cfg["window_size"]
max_pool_size  = cfg["max_pool_size"]
ef_build       = cfg["ef_build"]


# data generation

def make_blob(n, center=None):
    offset = center if center is not None else np.zeros(dim, dtype=np.float32)
    return (rng.standard_normal((n, dim)).astype(np.float32) * cluster_std + offset)


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


def save_summary(all_results):
    rows = []
    for (strategy, sigma, ef), results in sorted(all_results.items()):
        def avg(field):
            return float(np.nanmean([r[field] for r in results]))
        rows.append({
            "strategy":              strategy,
            "shift_sigma":           sigma,
            "ef_search":             ef,
            "mean_recall":           avg("recall"),
            "mean_ep_distance":      avg("ep_dist"),
            "mean_bl_entry_dist":    avg("bl_entry_dist"),
            "mean_ul_dist_comps":    avg("ul_dist_comps"),
            "mean_base_visited":     avg("base_visited"),
            "mean_base_dist_comps":  avg("base_dist_comps"),
            "n_queries":             len(results),
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
    for (strategy, sigma, ef), results in sorted(all_results.items()):
        for i, r in enumerate(results):
            row = {
                "strategy":    strategy,
                "shift_sigma": sigma,
                "ef_search":   ef,
                "query_id":    i,
                "recall":      r["recall"],
                "ep_dist":     r["ep_dist"],
                "bl_entry_dist": r["bl_entry_dist"],
                "base_visited":  r["base_visited"],
            }
            rows.append(row)
    path = os.path.join(out_dir, "per_query.csv")
    pd.DataFrame(rows).round(6).to_csv(path, index=False)
    print(f"  saved {path}")


def save_adapt_log(adapt_log):
    if not adapt_log:
        return
    path = os.path.join(out_dir, "adapt_log.csv")
    pd.DataFrame(adapt_log).to_csv(path, index=False)
    print(f"  saved {path} ({len(adapt_log)} entries)")


def main():
    print(f"mode={args.mode}  n_init={n_init:,}  cluster_std={cluster_std}")
    print(f"n_queries={n_queries}  n_warmup={n_warmup}  k={k}")
    print(f"alpha={alpha}  max_layer={max_layer}  cooldown={cooldown}  "
          f"window_size={window_size}  max_pool_size={max_pool_size}")
    print(f"shift_sigmas={shift_sigmas}")
    print(f"ef_sweep={ef_sweep}\n")

    print("generating data...")
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
    adapt_log   = []

    # no_adaptation baseline
    print(f"\n{'='*60}")
    print("strategy: no_adaptation")
    print(f"{'='*60}")
    for sigma in shift_sigmas:
        idx = load_fresh(index_path)
        eval_q  = query_sets[sigma][n_warmup:]
        eval_gt = ground_truths[sigma][n_warmup:]
        print(f"\n  sigma={sigma}")
        for ef in ef_sweep:
            results = run_queries_plain(idx, eval_q, eval_gt, ef)
            all_results[("no_adaptation", sigma, ef)] = results
            mean_r  = np.mean([r["recall"]        for r in results])
            mean_bl = np.mean([r["bl_entry_dist"]  for r in results])
            print(f"    ef={ef:>4}  recall={mean_r:.4f}  bl_entry={mean_bl:.2f}")

    # adaptation strategy using poolAndRewire
    print(f"\n{'='*60}")
    print("strategy: poolAndRewire  (rewiring + entry-pool)")
    print(f"{'='*60}")
    for sigma in shift_sigmas:
        warmup_q = query_sets[sigma][:n_warmup]
        eval_q   = query_sets[sigma][n_warmup:]
        eval_gt  = ground_truths[sigma][n_warmup:]
        print(f"\n  sigma={sigma}  building controller...")

        for ef in ef_sweep:
            idx = load_fresh(index_path)
            ctrl = PoolAndRewireController(
                idx,
                reference_queries=warmup_q,
                window_size=window_size,
                alpha=alpha,
                max_layer=max_layer,
                queries_per_rewire=queries_per_rewire,
                cooldown=cooldown,
                max_pool_size=max_pool_size,
            )
            results = run_query_batch(idx, eval_q, eval_gt, k=k, ef=ef,
                                      controller=ctrl, use_pool=True)
            all_results[("poolAndRewire", sigma, ef)] = results
            mean_r  = np.mean([r["recall"]       for r in results])
            mean_bl = np.mean([r["bl_entry_dist"] for r in results])
            print(f"    ef={ef:>4}  recall={mean_r:.4f}  bl_entry={mean_bl:.2f}"
                  f"  updates={ctrl.update_count}")
            for entry in ctrl.update_log:
                entry["sigma"] = sigma
                entry["ef"] = ef
                adapt_log.append(entry)

    print("\nsaving results...")
    save_summary(all_results)
    save_per_query(all_results)
    save_adapt_log(adapt_log)
    print(f"\ndone — results in {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
