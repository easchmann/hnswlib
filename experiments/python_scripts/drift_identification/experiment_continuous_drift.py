# Build an HNSW index on a Gaussian blob, then query it with vectors drawn from
# increasingly shifted distributions. The shift is along a fixed random direction
# so the distance from the construction distribution increases monotonically.
#
# Ground truth is computed against the initial index data only (no insertion).
# Saves summary.csv, per_query.csv, lb_traces.csv, params.csv
# plot_continuous_drift.py for figures.
#
# Usage:
#   python experiment_continuous_drift.py --mode small
#   python experiment_continuous_drift.py --mode large

import os
import sys
import argparse
import numpy as np
import pandas as pd

import hnswlib

rng = np.random.default_rng(42)

configs = {
    "small": {
        "dim": 128,
        "n_init": 10_000,
        "cluster_std": 1.0,
        "shift_sigmas": [0, 1, 2, 3, 4, 6, 8, 10, 12, 16],
        "n_queries": 200,
        "M": 16,
        "ef_construction": 200,
        "ef_sweep": [5, 10, 20, 50, 100, 200],
        "k": 10,
        "gt_backend": "numpy",
    },
    "large": {
        "dim": 128,
        "n_init": 10_000_000,
        "cluster_std": 50.0,
        "shift_sigmas": [0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20],
        "n_queries": 1_000,
        "M": 16,
        "ef_construction": 200,
        "ef_sweep": [10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
        "k": 10,
        "gt_backend": "faiss",
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["small", "large"], default="small")
parser.add_argument("--out_dir", default=None)
args = parser.parse_args()

cfg = configs[args.mode]
out_dir = args.out_dir or f"results_continuous_drift_{args.mode}_std50_higherEF"
os.makedirs(out_dir, exist_ok=True)

dim           = cfg["dim"]
n_init        = cfg["n_init"]
cluster_std   = cfg["cluster_std"]
shift_sigmas  = cfg["shift_sigmas"]
n_queries     = cfg["n_queries"]
ef_sweep      = cfg["ef_sweep"]
k             = cfg["k"]
gt_backend    = cfg["gt_backend"]


# --- data generation ---------------------------------------------------------

def make_blob(n, center=None):
    c = center if center is not None else np.zeros(dim, dtype=np.float32)
    return rng.standard_normal((n, dim)).astype(np.float32) * cluster_std + c


# --- ground truth ------------------------------------------------------------

def compute_gt_numpy(data, queries):
    # batched to avoid OOM on large datasets
    results = []
    for i in range(0, len(queries), 50):
        batch = queries[i:i + 50]
        dists = np.sum((batch[:, None, :] - data[None, :, :]) ** 2, axis=-1)
        results.append(np.argsort(dists, axis=1)[:, :k])
    return np.vstack(results).astype(np.int32)


def compute_gt_faiss(data, queries):
    import faiss

    index = faiss.IndexFlatL2(dim)
    if faiss.get_num_gpus() > 0:
        # print(f"  faiss: {faiss.get_num_gpus()} GPU(s) available")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    # else:
    #     print("  faiss: no GPU found, using CPU")

    index.add(np.ascontiguousarray(data, dtype=np.float32))
    _, neighbours = index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
    return neighbours.astype(np.int32)


def compute_gt(data, queries):
    if gt_backend == "faiss":
        return compute_gt_faiss(data, queries)
    return compute_gt_numpy(data, queries)


# --- index -------------------------------------------------------------------

def build_index(data):
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.init_index(
        max_elements=len(data),
        ef_construction=cfg["ef_construction"],
        M=cfg["M"]
    )
    idx.add_items(data, np.arange(len(data), dtype=np.int32))
    return idx


# --- querying ----------------------------------------------------------------

def recall(predicted, ground_truth):
    hits = sum(len(set(p) & set(g)) for p, g in zip(predicted, ground_truth))
    return hits / (len(ground_truth) * ground_truth.shape[1])


def query_index(idx, queries, gt):
    # returns a list of dicts, one per query
    results = []
    for i, q in enumerate(queries):
        pred, _ = idx.knn_query(q.reshape(1, -1), k=k)
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

def save_params(direction, path):
    rows = [
        {"param": "mode",           "value": args.mode},
        {"param": "dim",            "value": dim},
        {"param": "n_init",         "value": n_init},
        {"param": "cluster_std",    "value": cluster_std},
        {"param": "shift_sigmas",   "value": str(shift_sigmas)},
        {"param": "n_queries",      "value": n_queries},
        {"param": "M",              "value": cfg["M"]},
        {"param": "ef_construction","value": cfg["ef_construction"]},
        {"param": "ef_sweep",       "value": str(ef_sweep)},
        {"param": "k_neighbours",   "value": k},
        {"param": "gt_backend",     "value": gt_backend},
        {"param": "rng_seed",       "value": 42},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  saved {path}")


def save_summary(all_results, path):
    # all_results: dict keyed by (sigma, ef) -> list of per-query dicts
    rows = []
    for (sigma, ef), results in sorted(all_results.items()):
        def avg(field):
            return float(np.nanmean([r[field] for r in results]))

        l1_visits = [r["layer_visits"][1] for r in results if len(r["layer_visits"]) > 1]

        rows.append({
            "shift_sigma":              sigma,
            "shift_absolute":           round(sigma * cluster_std, 4),
            "ef_search":                ef,
            "mean_recall":              avg("recall"),
            "mean_ep_distance":         avg("ep_dist"),
            "mean_bl_entry_distance":   avg("bl_entry_dist"),
            "mean_ul_dist_comps":       avg("ul_dist_comps"),
            "mean_layer1_visits":       float(np.mean(l1_visits)) if l1_visits else float("nan"),
            "mean_base_visited":        avg("base_visited"),
            "mean_base_dist_comps":     avg("base_dist_comps"),
            "mean_candidates_remaining":avg("candidates_remaining"),
            "n_queries":                len(results),
        })

    df = pd.DataFrame(rows).round(4)
    df.to_csv(path, index=False)
    print(f"  saved {path}")

    # print recall table for quick inspection
    pivot = df.pivot(index="shift_sigma", columns="ef_search", values="mean_recall").round(3)
    print("\nRecall@k (rows=shift σ, cols=ef):")
    print(pivot.to_string())


def save_per_query(all_results, path):
    rows = []
    for (sigma, ef), results in sorted(all_results.items()):
        for i, r in enumerate(results):
            row = {
                "shift_sigma":          sigma,
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
    # store mean lowerbound trace as one row per (sigma, ef, iteration)
    # this keeps the CSV flat and makes it easy to plot later
    rows = []
    for (sigma, ef), results in sorted(all_results.items()):
        traces = [r["lb_trace"] for r in results if r["lb_trace"]]
        if not traces:
            continue
        min_len = min(len(t) for t in traces)
        mean_trace = np.mean([t[:min_len] for t in traces], axis=0)
        for iteration, value in enumerate(mean_trace):
            rows.append({
                "shift_sigma":     sigma,
                "ef_search":       ef,
                "iteration":       iteration,
                "mean_lowerbound": round(float(value), 6),
            })

    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  saved {path}")


# --- main --------------------------------------------------------------------

def main():
    print(f"mode={args.mode}  n_init={n_init:,}  cluster_std={cluster_std}")
    print(f"shift_sigmas={shift_sigmas}")
    print(f"ef_sweep={ef_sweep}  n_queries={n_queries} per level\n")

    # build index on the initial blob
    print("building index...")
    init_vecs = make_blob(n_init)
    idx = build_index(init_vecs)
    print(f"  {n_init:,} vectors, M={cfg['M']}, ef_construction={cfg['ef_construction']}\n")

    # pick a random unit direction and generate query sets along it
    direction = rng.standard_normal(dim).astype(np.float32)
    direction /= np.linalg.norm(direction)

    print("generating query sets...")
    query_sets = {}
    for sigma in shift_sigmas:
        center = direction * sigma * cluster_std
        query_sets[sigma] = make_blob(n_queries, center=center)
        print(f"  sigma={sigma:>4}  |center| = {np.linalg.norm(center):.2f}")

    # ground truth against initial data only (no insertion)
    print(f"\ncomputing ground truth ({gt_backend})...")
    ground_truths = {}
    for sigma in shift_sigmas:
        ground_truths[sigma] = compute_gt(init_vecs, query_sets[sigma])
    print("  done\n")

    # query at every (ef, sigma) combination
    print(f"querying: {len(shift_sigmas)} shift levels x {len(ef_sweep)} ef values")
    all_results = {}

    for ef in ef_sweep:
        idx.set_ef(ef)
        print(f"\nef={ef}")
        for sigma in shift_sigmas:
            results = query_index(idx, query_sets[sigma], ground_truths[sigma])
            all_results[(sigma, ef)] = results
            mean_recall = np.mean([r["recall"] for r in results])
            mean_ep     = np.nanmean([r["ep_dist"] for r in results])
            mean_bl     = np.nanmean([r["bl_entry_dist"] for r in results])
            print(f"  sigma={sigma:>4}  recall={mean_recall:.4f}  ep={mean_ep:.1f}  bl_entry={mean_bl:.1f}")

    # save
    print("\nsaving results...")
    save_params(direction,   os.path.join(out_dir, "params.csv"))
    save_summary(all_results, os.path.join(out_dir, "summary.csv"))
    save_per_query(all_results, os.path.join(out_dir, "per_query.csv"))
    save_lb_traces(all_results, os.path.join(out_dir, "lb_traces.csv"))
    print(f"\ndone — results in {os.path.abspath(out_dir)}")
    print("run plot_continuous_drift.py to generate figures")


if __name__ == "__main__":
    main()