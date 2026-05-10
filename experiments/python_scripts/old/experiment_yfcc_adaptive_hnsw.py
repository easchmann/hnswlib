# AdaptiveHNSW experiment on YFCC directional drift.
#
# Compares two strategies across a sweep of sigma (shift magnitude) and ef values:
#   no_adaptation  — plain hnswlib.Index, no structural changes
#   adaptive_hnsw  — AdaptiveHNSW wrapper: auto-detects drift via sliding window
#                    on base_layer_entry_distance, then builds a progressive
#                    anchor chain from the last centroid to the current one
#
# For each sigma:
#   1. Load a fresh copy of the base index.
#   2. (adaptive only) Wrap with AdaptiveHNSW and run n_warmup queries with
#      adapt=True so drift detection and adaptation happen automatically.
#   3. Evaluate the remaining n_queries - n_warmup queries at each ef value.
#
# Output files (in --out_dir):
#   params.csv          experiment hyperparameters
#   summary.csv         per (strategy, sigma, ef) mean metrics + recall delta table
#   per_query.csv       per-query metrics for every (strategy, sigma, ef)
#   adaptation_log.csv  one row per triggered adaptation event
#
# Usage:
#   python experiments/experiment_yfcc_adaptive_hnsw.py \
#       --embeddings_path data/yfcc_sampled/embeddings/embeddings_float32.npy \
#       --metadata_path   data/yfcc_sampled/embeddings/metadata.csv \
#       --n_index         500000 \
#       --out_dir         results_yfcc_adaptive_hnsw

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hnswlib
from experiments.controllers.adaptive_hnsw import AdaptiveHNSW

rng = np.random.default_rng(42)


# argument parsing

parser = argparse.ArgumentParser()
parser.add_argument("--embeddings_path",       required=True)
parser.add_argument("--metadata_path",         required=True)
parser.add_argument("--n_index",               type=int,   default=None)
parser.add_argument("--n_queries",             type=int,   default=1_000)
parser.add_argument("--n_warmup",              type=int,   default=200)
parser.add_argument("--shift_sigmas",          type=float, nargs="+",
                    default=[0, 1, 2, 3, 4, 6, 8, 10, 12, 16])
parser.add_argument("--ef_sweep",              type=int,   nargs="+",
                    default=[10, 20, 50, 100, 200, 500])
parser.add_argument("--M",                     type=int,   default=16)
parser.add_argument("--ef_construction",       type=int,   default=200)
parser.add_argument("--k",                     type=int,   default=10)
# AdaptiveHNSW hyperparameters
parser.add_argument("--k_repair_nodes",        type=int,   default=50)
parser.add_argument("--ef_repair",             type=int,   default=500)
parser.add_argument("--n_repair_rounds",       type=int,   default=3)
parser.add_argument("--window_size",           type=int,   default=200)
parser.add_argument("--threshold_factor",      type=float, default=1.5)
parser.add_argument("--out_dir",               default="results_yfcc_adaptive_hnsw")
args = parser.parse_args()

dim = 384
os.makedirs(args.out_dir, exist_ok=True)
INDEX_SAVE_PATH = os.path.join(args.out_dir, "base_index.bin")


def load_embeddings():
    print("loading embeddings...")
    emb  = np.load(args.embeddings_path, mmap_mode='r')
    meta = pd.read_csv(args.metadata_path)
    assert len(emb) == len(meta), "embedding/metadata length mismatch"
    print(f"  {len(emb):,} embeddings")
    print(f"  year distribution:\n{meta.groupby('year').size().to_string()}")
    return emb, meta


def compute_drift_direction(emb, meta):
    mean_2007 = np.array(emb[(meta['year'] == 2007).values], dtype=np.float32).mean(axis=0)
    mean_2013 = np.array(emb[(meta['year'] == 2013).values], dtype=np.float32).mean(axis=0)
    direction = mean_2013 - mean_2007
    magnitude = np.linalg.norm(direction)
    direction /= magnitude
    print(f"\ndrift direction magnitude: {magnitude:.3f}")
    return direction.astype(np.float32), magnitude


def build_splits(emb, direction, displacement_magnitude):
    n_total = len(emb)
    if args.n_index and n_total > args.n_index:
        chosen = rng.choice(n_total, args.n_index, replace=False)
        chosen.sort()
        index_vecs = np.array(emb[chosen], dtype=np.float32)
    else:
        index_vecs = np.array(emb, dtype=np.float32)
    print(f"\nindex: {len(index_vecs):,} vectors")

    q_idx      = rng.choice(n_total, args.n_queries, replace=False)
    q_idx.sort()
    query_base = np.array(emb[q_idx], dtype=np.float32)

    query_sets = {}
    for sigma in args.shift_sigmas:
        shift = direction * sigma * displacement_magnitude
        query_sets[sigma] = query_base + shift
        print(f"  sigma={sigma:>5}  |shift|={np.linalg.norm(shift):.1f}")

    return index_vecs, query_sets



# ground truth

def compute_gt(index_vecs, queries):
    import faiss
    index_gt = faiss.IndexFlatL2(dim)
    index_gt.add(np.ascontiguousarray(index_vecs, dtype=np.float32))
    _, neighbours = index_gt.search(
        np.ascontiguousarray(queries, dtype=np.float32), args.k
    )
    return neighbours.astype(np.int32)



# index helpers


def build_index(data):
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.init_index(
        max_elements=len(data),
        ef_construction=args.ef_construction,
        M=args.M,
    )
    idx.add_items(data, np.arange(len(data), dtype=np.int32))
    return idx


def load_fresh_index(n_elements):
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.load_index(INDEX_SAVE_PATH, max_elements=n_elements)
    return idx



# query helpers


def collect_stats_after_query(pred, gt_row, k):
    s = hnswlib.get_last_query_stats()
    return {
        "recall": len(set(pred) & set(gt_row)) / k,
        "ep_dist": float(s["entry_point_distance"]),
        "bl_entry_dist": float(s["base_layer_entry_distance"]),
        "ul_dist_comps": int(s["upper_layer_distance_computations"]),
        "layer_visits": list(s["layer_visit_counts"]),
        "base_visited": int(s["base_layer_visited_count"]),
        "base_dist_comps":  int(s["base_layer_distance_computations"]),
        "candidates_remaining": int(s["candidates_remaining_at_termination"]),
        "lb_trace":  list(s["lowerbound_trace"]),
    }


def run_query_batch(raw_index, queries, gt, k, ef):
    raw_index.set_ef(ef)
    results = []
    for i, q in enumerate(queries):
        pred, _ = raw_index.knn_query(q.reshape(1, -1), k=k)
        results.append(collect_stats_after_query(pred[0], gt[i], k))
    return results


def run_warmup_adaptive(adaptive, queries, gt, k, ef=200):
    adaptive.index.set_ef(ef)
    results = []
    for i, q in enumerate(queries):
        labels, _ = adaptive.knn_query(q.reshape(1, -1), k=k, adapt=True)
        results.append(collect_stats_after_query(labels[0], gt[i], k))
    return results


def run_eval_adaptive(adaptive, queries, gt, k, ef):
    
    adaptive.index.set_ef(ef)
    results = []
    for i, q in enumerate(queries):
        labels, _ = adaptive.index.knn_query(q.reshape(1, -1), k=k)
        results.append(collect_stats_after_query(labels[0], gt[i], k))
    return results



# output helpers


def save_params():
    rows = [
        {"param": "dataset",            "value": "YFCC-DINO"},
        {"param": "dim",                "value": dim},
        {"param": "n_index",            "value": args.n_index or "all"},
        {"param": "n_queries",          "value": args.n_queries},
        {"param": "n_warmup",           "value": args.n_warmup},
        {"param": "shift_sigmas",       "value": str(args.shift_sigmas)},
        {"param": "ef_sweep",           "value": str(args.ef_sweep)},
        {"param": "M",                  "value": args.M},
        {"param": "ef_construction",    "value": args.ef_construction},
        {"param": "k",                  "value": args.k},
        {"param": "k_repair_nodes",     "value": args.k_repair_nodes},
        {"param": "ef_repair",          "value": args.ef_repair},
        {"param": "n_repair_rounds",    "value": args.n_repair_rounds},
        {"param": "window_size",        "value": args.window_size},
        {"param": "threshold_factor",   "value": args.threshold_factor},
        {"param": "drift_direction",    "value": "mean_2007_to_mean_2013"},
        {"param": "rng_seed",           "value": 42},
    ]
    path = os.path.join(args.out_dir, "params.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  saved {path}")


def save_summary(all_results):
    rows = []
    for (strategy, sigma, ef), results in sorted(all_results.items()):
        def avg(field):
            return float(np.nanmean([r[field] for r in results]))
        l1 = [r["layer_visits"][1] for r in results if len(r["layer_visits"]) > 1]
        rows.append({
            "strategy":  strategy,
            "shift_sigma": sigma,
            "ef_search":   ef,
            "mean_recall":  avg("recall"),
            "mean_ep_distance":  avg("ep_dist"),
            "mean_bl_entry_distance":  avg("bl_entry_dist"),
            "mean_ul_dist_comps": avg("ul_dist_comps"),
            "mean_layer1_visits": float(np.mean(l1)) if l1 else float("nan"),
            "mean_base_visited":  avg("base_visited"),
            "mean_base_dist_comps":  avg("base_dist_comps"),
            "mean_candidates_remaining":  avg("candidates_remaining"),
            "n_queries": len(results),
        })

    df = pd.DataFrame(rows).round(4)
    path = os.path.join(args.out_dir, "summary.csv")
    df.to_csv(path, index=False)
    print(f"  saved {path}")

    for strategy in df["strategy"].unique():
        sub = df[df["strategy"] == strategy]
        pivot = sub.pivot(index="shift_sigma", columns="ef_search", values="mean_recall").round(3)
        print(f"\n{strategy} recall@{args.k}:")
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
                "strategy":   strategy,
                "shift_sigma": sigma,
                "ef_search": ef,
                "query_id":  i,
                "recall":  r["recall"],
                "ep_dist": r["ep_dist"],
                "bl_entry_dist": r["bl_entry_dist"],
                "ul_dist_comps":  r["ul_dist_comps"],
                "base_visited": r["base_visited"],
                "base_dist_comps":  r["base_dist_comps"],
                "candidates_remaining": r["candidates_remaining"],
                "lb_trace_final":  r["lb_trace"][-1] if r["lb_trace"] else float("nan"),
            }
            for layer, visits in enumerate(r["layer_visits"]):
                row[f"layer{layer}_visits"] = visits
            rows.append(row)
    path = os.path.join(args.out_dir, "per_query.csv")
    pd.DataFrame(rows).round(6).to_csv(path, index=False)
    print(f"  saved {path}")


def save_adaptation_log(all_logs):
    if not all_logs:
        return
    rows = []
    for sigma, log in all_logs:
        for entry in log:
            rows.append({
                "shift_sigma": sigma,
                "adaptation_n":  entry["adaptation_n"],
                "best_node":   entry["best_node"],
                "max_level":  entry["max_level"],
                "n_repaired": entry["n_repaired"],
                "window_mean":   entry["window_mean_before"],
                "threshold":   entry["threshold"],
            })
    path = os.path.join(args.out_dir, "adaptation_log.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  saved {path} ({len(rows)} adaptation events)")




def main():
    print("AdaptiveHNSW experiment — YFCC directional drift")
    print(f"n_queries={args.n_queries}  n_warmup={args.n_warmup}  k={args.k}")
    print(f"k_repair_nodes={args.k_repair_nodes}  ef_repair={args.ef_repair}  n_repair_rounds={args.n_repair_rounds}")
    print(f"window_size={args.window_size}  threshold_factor={args.threshold_factor}")
    print(f"shift_sigmas={args.shift_sigmas}")
    print(f"ef_sweep={args.ef_sweep}\n")

    emb, meta = load_embeddings()
    direction, displacement_magnitude = compute_drift_direction(emb, meta)
    index_vecs, query_sets = build_splits(emb, direction, displacement_magnitude)
    del emb

    print(f"\nbuilding base index on {len(index_vecs):,} vectors...")
    base_idx = build_index(index_vecs)
    base_idx.save_index(INDEX_SAVE_PATH)
    print(f"  saved to {INDEX_SAVE_PATH}")

    print("\ncomputing ground truth...")
    ground_truths = {}
    for sigma, queries in sorted(query_sets.items()):
        print(f"  sigma={sigma}")
        ground_truths[sigma] = compute_gt(index_vecs, queries)
    print("  done")

    # calibrate threshold on sigma=0 (no shift) using the base index
    print("\nmeasuring baseline (sigma=0, ef=200)...")
    tmp_adaptive = AdaptiveHNSW(
        base_idx,
        window_size=args.window_size,
        threshold_factor=args.threshold_factor,
        k_repair_nodes=args.k_repair_nodes,
        ef_repair=args.ef_repair,
        n_repair_rounds=args.n_repair_rounds,
    )
    baseline_dist = tmp_adaptive.calibrate(query_sets[0.0], ef=200, k=args.k)
    print(f"  baseline bl_entry_dist: {baseline_dist:.3f}")
    print(f"  detection threshold:    {baseline_dist * args.threshold_factor:.3f}")
    del tmp_adaptive, base_idx

    n_elements  = len(index_vecs)
    all_results = {}
    all_adapt_logs = []


    # no_adaptation baseline
    
    print(f"\n{'='*60}")
    print("strategy: no_adaptation")
    print(f"{'='*60}")

    for sigma, queries in sorted(query_sets.items()):
        idx = load_fresh_index(n_elements)
        eval_queries = queries[args.n_warmup:]
        eval_gt      = ground_truths[sigma][args.n_warmup:]
        print(f"\n  sigma={sigma}")
        for ef in args.ef_sweep:
            results = run_query_batch(idx, eval_queries, eval_gt, k=args.k, ef=ef)
            all_results[("no_adaptation", sigma, ef)] = results
            mean_r  = np.mean([r["recall"]       for r in results])
            mean_bl = np.mean([r["bl_entry_dist"] for r in results])
            print(f"    ef={ef:>4}  recall={mean_r:.4f}  bl_entry={mean_bl:.1f}")


    # adaptive_hnsw

    print(f"\n{'='*60}")
    print("strategy: adaptive_hnsw")
    print(f"{'='*60}")

    for sigma, queries in sorted(query_sets.items()):
        idx = load_fresh_index(n_elements)

        adaptive = AdaptiveHNSW(
            idx,
            window_size=args.window_size,
            threshold_factor=args.threshold_factor,
            k_repair_nodes=args.k_repair_nodes,
            ef_repair=args.ef_repair,
        )
        # inject the pre-measured baseline so the threshold is consistent
        adaptive._baseline  = baseline_dist
        adaptive._threshold = baseline_dist * args.threshold_factor

        warmup_queries = queries[:args.n_warmup]
        warmup_gt      = ground_truths[sigma][:args.n_warmup]
        eval_queries   = queries[args.n_warmup:]
        eval_gt        = ground_truths[sigma][args.n_warmup:]

        print(f"\n  sigma={sigma}  running {args.n_warmup} warmup queries...")
        run_warmup_adaptive(adaptive, warmup_queries, warmup_gt, k=args.k, ef=200)

        n_adaptations = len(adaptive.adaptation_log)
        print(f"  adaptations triggered during warmup: {n_adaptations}")

        for ef in args.ef_sweep:
            results = run_eval_adaptive(adaptive, eval_queries, eval_gt, k=args.k, ef=ef)
            all_results[("adaptive_hnsw", sigma, ef)] = results
            mean_r  = np.mean([r["recall"]       for r in results])
            mean_bl = np.mean([r["bl_entry_dist"] for r in results])
            print(f"    ef={ef:>4}  recall={mean_r:.4f}  bl_entry={mean_bl:.1f}")

        if adaptive.adaptation_log:
            all_adapt_logs.append((sigma, adaptive.adaptation_log))


    # save
    
    print("\nsaving results...")
    save_params()
    save_summary(all_results)
    save_per_query(all_results)
    save_adaptation_log(all_adapt_logs)
    print(f"\ndone — results in {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
