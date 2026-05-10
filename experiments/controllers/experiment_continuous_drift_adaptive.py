# AdaptiveHNSW experiment on synthetic continuous drift.
#
# Mirrors experiment_continuous_drift.py (Gaussian blob, fixed drift direction, sigma sweep) but tests the adaptation strategy:
# AdaptiveHNSW wrapper: auto-detects drift via sliding window on base_layer_entry_distance, then repairs base-layer edges
# and promotes a shortcut node into the upper layers
#
# For each sigma:
#   1. Load a fresh copy of the base index (saved once to disk at startup).
#   2. (adaptive only) Inject the pre-measured baseline, run n_warmup queries
#      with adapt=True so drift detection and adaptation fire automatically.
#   3. Evaluate the remaining n_queries - n_warmup queries at each ef value.


import os
import sys
import argparse
import tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hnswlib
from adaptive_hnsw import AdaptiveHNSW

rng = np.random.default_rng(42)

# configurations

configs = {
    "small": {
        # Single Gaussian blob, small n.  ef/n is high so recall is good.
        "dim":              128,
        "n_init":           10_000,
        "n_clusters":       1,  # 1 = single blob
        "cluster_std":      1.0,
        "shift_sigmas":     [0, 1, 2, 3, 4, 6, 8, 10, 12, 16],
        "n_queries":        500,
        "n_warmup":         100,
        "M":                16,
        "ef_construction":  200,
        "ef_sweep":         [10, 20, 50, 100, 200],
        "k":                10,
        "k_repair_nodes":   50,
        "ef_repair":        500,
        "n_repair_rounds":  3,
        "window_size":      50,
        "threshold_factor": 1.5,
        "gt_backend":       "numpy",
    },
    "large": {
        # Mixture-of-Gaussians so that HNSW has real structure to exploit at large n.  With n_clusters=500 and n_init=5M, each cluster holds 10k points.  
        # Within-cluster distances are ~sqrt(128)*1 ≈ 11 (much smaller than cross-cluster distances of ~sqrt(128)*cluster_spread ≈ 1130), so true k-NN are within the same cluster and ef=500 is sufficient.
        # The drift direction shifts queries out of the indexed cluster region,reducing recall in the same way as in the single-blob case.
        "dim":              128,
        "n_init":           5_000_000,
        "n_clusters":       500, # mixture components
        "cluster_std":      1.0, # within-cluster spread
        "cluster_spread":   100.0, # std of cluster center placement
        "shift_sigmas":     [0, 1, 2, 3, 4, 6, 8, 10, 12, 16],
        "n_queries":        1_000,
        "n_warmup":         200,
        "M":                16,
        "ef_construction":  200,
        "ef_sweep":         [10, 20, 50, 100, 200, 500],
        "k":                10,
        "k_repair_nodes":   50,
        "ef_repair":        2000,
        "n_repair_rounds":  3,
        "window_size":      200,
        "threshold_factor": 1.5,
        "gt_backend":       "faiss",
    },
}

# argument parsing

parser = argparse.ArgumentParser()
parser.add_argument("--mode",             choices=["small", "large"], default="small")
parser.add_argument("--dim",              type=int,   default=None)
parser.add_argument("--n_init",           type=int,   default=None)
parser.add_argument("--n_clusters",       type=int,   default=None)
parser.add_argument("--cluster_std",      type=float, default=None)
parser.add_argument("--cluster_spread",   type=float, default=None)
parser.add_argument("--shift_sigmas",     type=float, nargs="+", default=None)
parser.add_argument("--n_queries",        type=int,   default=None)
parser.add_argument("--n_warmup",         type=int,   default=None)
parser.add_argument("--M",                type=int,   default=None)
parser.add_argument("--ef_construction",  type=int,   default=None)
parser.add_argument("--ef_sweep",         type=int,   nargs="+", default=None)
parser.add_argument("--k",                type=int,   default=None)
parser.add_argument("--k_repair_nodes",   type=int,   default=None)
parser.add_argument("--ef_repair",        type=int,   default=None)
parser.add_argument("--n_repair_rounds",  type=int,   default=None)
parser.add_argument("--window_size",      type=int,   default=None)
parser.add_argument("--threshold_factor", type=float, default=None)
parser.add_argument("--out_dir",          default=None)
args = parser.parse_args()

# CLI overrides take precedence over config defaults
cfg = dict(configs[args.mode])
for key in ["dim", "n_init", "n_clusters", "cluster_std", "cluster_spread",
            "shift_sigmas", "n_queries", "n_warmup", "M", "ef_construction",
            "ef_sweep", "k", "k_repair_nodes", "ef_repair", "n_repair_rounds",
            "window_size", "threshold_factor"]:
    cli_val = getattr(args, key, None)
    if cli_val is not None:
        cfg[key] = cli_val

out_dir = args.out_dir or f"results_continuous_drift_adaptive_{args.mode}"
os.makedirs(out_dir, exist_ok=True)

dim             = cfg["dim"]
n_init          = cfg["n_init"]
n_clusters      = cfg.get("n_clusters", 1)
cluster_std     = cfg["cluster_std"]
cluster_spread  = cfg.get("cluster_spread", 1.0)
shift_sigmas    = cfg["shift_sigmas"]
n_queries       = cfg["n_queries"]
n_warmup        = cfg["n_warmup"]
M               = cfg["M"]
ef_construction = cfg["ef_construction"]
ef_sweep        = cfg["ef_sweep"]
k               = cfg["k"]
k_repair_nodes  = cfg["k_repair_nodes"]
ef_repair       = cfg["ef_repair"]
n_repair_rounds = cfg["n_repair_rounds"]
window_size     = cfg["window_size"]
threshold_factor = cfg["threshold_factor"]
gt_backend      = cfg["gt_backend"]


# data generation
# For n_clusters > 1 the cluster centers are fixed at module load so that ndex data and query sets share the same mixture geometry.
if n_clusters > 1:
    _cluster_centers = (rng.standard_normal((n_clusters, dim)).astype(np.float32) * cluster_spread)
else:
    _cluster_centers = None


# Generate n vectors.
# Single-blob mode (n_clusters == 1): isotropic Gaussian with cluster_std, optionally offset by `center`.
# Mixture mode (n_clusters > 1): each point is drawn from one of the pre-generated cluster centers (uniform assignment) with within-cluster
# noise cluster_std.  
# `center` is added as a global offset so that query sets for different sigma values shift the entire mixture.
def make_blob(n, center=None):
    offset = center if center is not None else np.zeros(dim, dtype=np.float32)
    if n_clusters == 1:
        return rng.standard_normal((n, dim)).astype(np.float32) * cluster_std + offset
    # mixture: pick a random cluster for each point, add within-cluster noise
    assignments = rng.integers(0, n_clusters, size=n)
    centers_for_points = _cluster_centers[assignments]  # (n, dim)
    noise = rng.standard_normal((n, dim)).astype(np.float32) * cluster_std
    return (centers_for_points + noise + offset).astype(np.float32)


# ground truth

def compute_gt_numpy(data, queries):
    results = []
    for start in range(0, len(queries), 50):
        batch = queries[start:start + 50]
        dists = np.sum((batch[:, None, :] - data[None, :, :]) ** 2, axis=-1)
        results.append(np.argsort(dists, axis=1)[:, :k])
    return np.vstack(results).astype(np.int32)


def compute_gt_faiss(data, queries):
    import faiss
    index = faiss.IndexFlatL2(dim)
    index.add(np.ascontiguousarray(data, dtype=np.float32))
    _, neighbours = index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
    return neighbours.astype(np.int32)


def compute_gt(data, queries):
    if gt_backend == "faiss":
        return compute_gt_faiss(data, queries)
    return compute_gt_numpy(data, queries)


# index helpers

def build_index(data):
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.init_index(max_elements=len(data), ef_construction=ef_construction, M=M)
    idx.add_items(data, np.arange(len(data), dtype=np.int32))
    return idx


def load_fresh_index(path):
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.load_index(path, max_elements=n_init)
    return idx


# query helpers

def collect_stats(pred, gt_row):
    s = hnswlib.get_last_query_stats()
    return {
        "recall": len(set(pred) & set(gt_row)) / k,
        "ep_dist": float(s["entry_point_distance"]),
        "bl_entry_dist": float(s["base_layer_entry_distance"]),
        "ul_dist_comps": int(s["upper_layer_distance_computations"]),
        "layer_visits": list(s["layer_visit_counts"]),
        "base_visited": int(s["base_layer_visited_count"]),
        "base_dist_comps": int(s["base_layer_distance_computations"]),
        "candidates_remaining": int(s["candidates_remaining_at_termination"]),
        "lb_trace": list(s["lowerbound_trace"]),
    }


def run_query_batch(raw_index, queries, gt, ef):
    raw_index.set_ef(ef)
    results = []
    for i, q in enumerate(queries):
        pred, _ = raw_index.knn_query(q.reshape(1, -1), k=k)
        results.append(collect_stats(pred[0], gt[i]))
    return results


def run_warmup_adaptive(adaptive, queries, gt, ef=200):
    adaptive.index.set_ef(ef)
    for i, q in enumerate(queries):
        adaptive.knn_query(q.reshape(1, -1), k=k, adapt=True)


def run_eval_adaptive(adaptive, queries, gt, ef):
    adaptive.index.set_ef(ef)
    results = []
    for i, q in enumerate(queries):
        pred, _ = adaptive.index.knn_query(q.reshape(1, -1), k=k)
        results.append(collect_stats(pred[0], gt[i]))
    return results


# output helpers

def save_params(direction):
    rows = [
        {"param": "mode", "value": args.mode},
        {"param": "dim",  "value": dim},
        {"param": "n_init",   "value": n_init},
        {"param": "n_clusters",  "value": n_clusters},
        {"param": "cluster_std", "value": cluster_std},
        {"param": "cluster_spread",   "value": cluster_spread},
        {"param": "shift_sigmas", "value": str(shift_sigmas)},
        {"param": "n_queries", "value": n_queries},
        {"param": "n_warmup",  "value": n_warmup},
        {"param": "M",   "value": M},
        {"param": "ef_construction",  "value": ef_construction},
        {"param": "ef_sweep", "value": str(ef_sweep)},
        {"param": "k", "value": k},
        {"param": "k_repair_nodes",  "value": k_repair_nodes},
        {"param": "ef_repair", "value": ef_repair},
        {"param": "n_repair_rounds",  "value": n_repair_rounds},
        {"param": "window_size",  "value": window_size},
        {"param": "threshold_factor", "value": threshold_factor},
        {"param": "gt_backend",  "value": gt_backend},
        {"param": "rng_seed", "value": 42},
    ]
    path = os.path.join(out_dir, "params.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f" saved {path}")


def save_summary(all_results):
    rows = []
    for (strategy, sigma, ef), results in sorted(all_results.items()):
        def avg(field):
            return float(np.nanmean([r[field] for r in results]))
        l1 = [r["layer_visits"][1] for r in results if len(r["layer_visits"]) > 1]
        rows.append({
            "strategy":strategy,
            "shift_sigma": sigma,
            "shift_absolute": round(sigma * cluster_std, 4),
            "ef_search": ef,
            "mean_recall": avg("recall"),
            "mean_ep_distance": avg("ep_dist"),
            "mean_bl_entry_distance": avg("bl_entry_dist"),
            "mean_ul_dist_comps": avg("ul_dist_comps"),
            "mean_layer1_visits": float(np.mean(l1)) if l1 else float("nan"),
            "mean_base_visited": avg("base_visited"),
            "mean_base_dist_comps": avg("base_dist_comps"),
            "mean_candidates_remaining": avg("candidates_remaining"),
            "n_queries": len(results),
        })

    df = pd.DataFrame(rows).round(4)
    path = os.path.join(out_dir, "summary.csv")
    df.to_csv(path, index=False)
    print(f" saved {path}")

    for strategy in df["strategy"].unique():
        sub = df[df["strategy"] == strategy]
        pivot = sub.pivot(index="shift_sigma", columns="ef_search", values="mean_recall").round(3)
        print(f"\n{strategy} recall@{k}:")
        print(pivot.to_string())

    base = df[df["strategy"] == "no_adaptation"].set_index(["shift_sigma", "ef_search"])["mean_recall"]
    adap = df[df["strategy"] == "adaptive_hnsw"].set_index(["shift_sigma", "ef_search"])["mean_recall"]
    if not base.empty and not adap.empty:
        delta = (adap - base).round(3).unstack("ef_search")
        print("\nrecall delta (adaptive_hnsw - no_adaptation), positive = improvement:")
        print(delta.to_string())


def save_per_query(all_results):
    rows = []
    for (strategy, sigma, ef), results in sorted(all_results.items()):
        for i, r in enumerate(results):
            row = {
                "strategy": strategy,
                "shift_sigma": sigma,
                "ef_search": ef,
                "query_id": i,
                "recall": r["recall"],
                "ep_dist": r["ep_dist"],
                "bl_entry_dist": r["bl_entry_dist"],
                "ul_dist_comps": r["ul_dist_comps"],
                "base_visited": r["base_visited"],
                "base_dist_comps": r["base_dist_comps"],
                "candidates_remaining": r["candidates_remaining"],
                "lb_trace_final": r["lb_trace"][-1] if r["lb_trace"] else float("nan"),
                "lb_trace_len": len(r["lb_trace"]),
            }
            for layer, visits in enumerate(r["layer_visits"]):
                row[f"layer{layer}_visits"] = visits
            rows.append(row)
    path = os.path.join(out_dir, "per_query.csv")
    pd.DataFrame(rows).round(6).to_csv(path, index=False)
    print(f"  saved {path}")


def save_adaptation_log(all_logs):
    if not all_logs:
        return
    rows = []
    for sigma, log in all_logs:
        for entry in log:
            rows.append({
                "shift_sigma":  sigma,
                "adaptation_n":  entry["adaptation_n"],
                "best_node": entry["best_node"],
                "max_level": entry["max_level"],
                "n_repaired": entry["n_repaired"],
                "window_mean": entry["window_mean_before"],
                "threshold": entry["threshold"],
            })
    path = os.path.join(out_dir, "adaptation_log.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f" saved {path} ({len(rows)} adaptation events)")



def main():
    if n_clusters > 1:
        print(f"dim={dim}  n_init={n_init:,}  n_clusters={n_clusters}  cluster_std={cluster_std}  cluster_spread={cluster_spread}")
    else:
        print(f"dim={dim}  n_init={n_init:,}  cluster_std={cluster_std}")
    print(f"n_queries={n_queries}  n_warmup={n_warmup}  k={k}")
    print(f"k_repair_nodes={k_repair_nodes}  ef_repair={ef_repair}  n_repair_rounds={n_repair_rounds}")
    print(f"window_size={window_size}  threshold_factor={threshold_factor}")
    print(f"shift_sigmas={shift_sigmas}")
    print(f"ef_sweep={ef_sweep}\n")

    # data
    print("generating index data...")
    init_vecs = make_blob(n_init)

    direction = rng.standard_normal(dim).astype(np.float32)
    direction /= np.linalg.norm(direction)
    print(f"  drift direction: unit vector in R^{dim}\n")

    print("generating query sets...")
    query_sets = {}
    for sigma in shift_sigmas:
        center = direction * sigma * cluster_std
        query_sets[sigma] = make_blob(n_queries, center=center)
        print(f"  sigma={sigma:>5}  |center|={np.linalg.norm(center):.2f}")

    # build index
    print(f"\nbuilding base index on {n_init:,} vectors...")
    base_idx = build_index(init_vecs)
    index_path = os.path.join(out_dir, "base_index.bin")
    base_idx.save_index(index_path)
    print(f"  saved to {index_path}")

    # GT
    print(f"\ncomputing ground truth ({gt_backend})...")
    ground_truths = {}
    for sigma in shift_sigmas:
        print(f"  sigma={sigma}")
        ground_truths[sigma] = compute_gt(init_vecs, query_sets[sigma])
    print("  done")
    del init_vecs  # free memory, index holds its own copy

    # calibrate
    print("\nmeasuring baseline (sigma=0, ef=200)...")
    tmp_adaptive = AdaptiveHNSW( base_idx, window_size=window_size, threshold_factor=threshold_factor, k_repair_nodes=k_repair_nodes, ef_repair=ef_repair, n_repair_rounds=n_repair_rounds)
    baseline_dist = tmp_adaptive.calibrate(query_sets[0], ef=200, k=k)
    print(f"  baseline bl_entry_dist: {baseline_dist:.4f}")
    print(f"  detection threshold:    {baseline_dist * threshold_factor:.4f}")
    del tmp_adaptive, base_idx

    all_results    = {}
    all_adapt_logs = []


    # strategy: no_adaptation
    print(f"\n{'='*60}")
    print("strategy: no_adaptation")
    print(f"{'='*60}")

    for sigma in shift_sigmas:
        idx = load_fresh_index(index_path)
        eval_queries = query_sets[sigma][n_warmup:]
        eval_gt      = ground_truths[sigma][n_warmup:]
        print(f"\n  sigma={sigma}")
        for ef in ef_sweep:
            results = run_query_batch(idx, eval_queries, eval_gt, ef)
            all_results[("no_adaptation", sigma, ef)] = results
            mean_r  = np.mean([r["recall"]       for r in results])
            mean_bl = np.mean([r["bl_entry_dist"] for r in results])
            print(f"    ef={ef:>4}  recall={mean_r:.4f}  bl_entry={mean_bl:.2f}")


    # strategy: adaptive_hnsw
    print(f"\n{'='*60}")
    print("strategy: adaptive_hnsw")
    print(f"{'='*60}")

    for sigma in shift_sigmas:
        idx = load_fresh_index(index_path)

        adaptive = AdaptiveHNSW(
            idx,
            window_size=window_size,
            threshold_factor=threshold_factor,
            k_repair_nodes=k_repair_nodes,
            ef_repair=ef_repair,
        )
        # inject pre-measured baseline so all sigmas use the same threshold
        adaptive._baseline  = baseline_dist
        adaptive._threshold = baseline_dist * threshold_factor

        warmup_queries = query_sets[sigma][:n_warmup]
        warmup_gt      = ground_truths[sigma][:n_warmup]
        eval_queries   = query_sets[sigma][n_warmup:]
        eval_gt        = ground_truths[sigma][n_warmup:]

        print(f"\n  sigma={sigma}  running {n_warmup} warmup queries...")
        run_warmup_adaptive(adaptive, warmup_queries, warmup_gt, ef=200)

        n_adapt = len(adaptive.adaptation_log)
        print(f"  adaptations triggered during warmup: {n_adapt}")

        for ef in ef_sweep:
            results = run_eval_adaptive(adaptive, eval_queries, eval_gt, ef)
            all_results[("adaptive_hnsw", sigma, ef)] = results
            mean_r  = np.mean([r["recall"]       for r in results])
            mean_bl = np.mean([r["bl_entry_dist"] for r in results])
            print(f"    ef={ef:>4}  recall={mean_r:.4f}  bl_entry={mean_bl:.2f}")

        if adaptive.adaptation_log:
            all_adapt_logs.append((sigma, adaptive.adaptation_log))

    
    # save
    print("\nsaving results...")
    save_params(direction)
    save_summary(all_results)
    save_per_query(all_results)
    save_adaptation_log(all_adapt_logs)
    print(f"\ndone - results in {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
