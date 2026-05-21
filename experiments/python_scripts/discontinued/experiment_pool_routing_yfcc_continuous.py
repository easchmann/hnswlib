# Pool routing experiment on YFCC DINO embeddings with continuous drift.
#
# Unlike the per-sigma version, the index and pool are built once and
# reused across all shift-sigma levels in order. The pool is seeded from
# sigma=0 warmup queries (the initial distribution), then eval batches for
# sigma=0,1,2,... are run sequentially on the same live index and pool.
# This mirrors real continuous drift: the pool represents the query
# distribution at deployment time, not a freshly-tuned oracle per shift.
#
# Drift direction is the 2007->2013 mean displacement; queries are shifted
# along it by increasing sigma.
#
# Strategies compared:
#   no_adaptation  — standard HNSW search
#   pool_<N>       — pool of N nodes built from sigma=0 warmup queries,
#                    each eval query routed to nearest pool node
#
# Saves summary.csv, per_query.csv, pool_build_log.csv, params.csv
#
# Usage:
#   python experiment_pool_routing_yfcc_continuous.py \
#       --embeddings_path data/yfcc/embeddings.npy \
#       --metadata_path   data/yfcc/metadata.csv \
#       --n_index 500000

import os
import sys
import argparse
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, ".."))

import hnswlib
from experiments.controllers.pool_router import PoolRouter

rng = np.random.default_rng(42)

parser = argparse.ArgumentParser()
parser.add_argument("--embeddings_path", required=True)
parser.add_argument("--metadata_path",   required=True)
parser.add_argument("--n_index",         type=int,   default=None)
parser.add_argument("--n_queries",       type=int,   default=1_000)
parser.add_argument("--n_warmup",        type=int,   default=200)
parser.add_argument("--shift_sigmas",    type=float, nargs="+",
                    default=[0, 1, 2, 3, 4, 6, 8, 10, 12, 16])
parser.add_argument("--ef_sweep",        type=int,   nargs="+",
                    default=[10, 20, 50, 100, 200, 500])
parser.add_argument("--M",               type=int,   default=16)
parser.add_argument("--ef_construction", type=int,   default=200)
parser.add_argument("--k",               type=int,   default=10)
parser.add_argument("--pool_sizes",      type=int,   nargs="+", default=[100, 500, 1000])
parser.add_argument("--ef_build",        type=int,   default=2000)
parser.add_argument("--k_build",         type=int,   default=5)
parser.add_argument("--out_dir", default="results_pool_routing_yfcc_continuous")
args = parser.parse_args()

dim     = 384
k       = args.k
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)


def load_embeddings():
    print("loading embeddings...")
    emb  = np.load(args.embeddings_path, mmap_mode='r')
    meta = pd.read_csv(args.metadata_path)
    assert len(emb) == len(meta), "embedding/metadata length mismatch"
    print(f"  {len(emb):,} embeddings, dim={emb.shape[1]}")
    print(f"  year distribution:\n{meta.groupby('year').size().to_string()}")
    return emb, meta


def compute_drift_direction(emb, meta):
    mean_2007 = np.array(emb[(meta['year'] == 2007).values], dtype=np.float32).mean(axis=0)
    mean_2013 = np.array(emb[(meta['year'] == 2013).values], dtype=np.float32).mean(axis=0)
    direction = mean_2013 - mean_2007
    magnitude = np.linalg.norm(direction)
    direction /= magnitude
    print(f"\ndrift direction (2007 -> 2013):")
    print(f"  displacement magnitude: {magnitude:.4f}")
    return direction.astype(np.float32), magnitude


def build_splits(emb, meta, direction, magnitude):
    n_total = len(emb)
    if args.n_index and n_total > args.n_index:
        chosen = rng.choice(n_total, args.n_index, replace=False)
        chosen.sort()
        index_vecs = np.array(emb[chosen], dtype=np.float32)
    else:
        index_vecs = np.array(emb, dtype=np.float32)
    print(f"\nindex: {len(index_vecs):,} vectors")

    q_idx = rng.choice(n_total, args.n_queries + args.n_warmup, replace=False)
    q_idx.sort()
    query_base = np.array(emb[q_idx], dtype=np.float32)
    print(f"queries: {len(query_base):,} base vectors (warmup + eval combined)")

    query_sets = {}
    for sigma in args.shift_sigmas:
        shift = direction * sigma * magnitude
        query_sets[sigma] = query_base + shift
        print(f"  sigma={sigma:>5}  |shift|={np.linalg.norm(shift):.4f}")

    return index_vecs, query_sets


def compute_gt(index_vecs, queries):
    import faiss
    flat = faiss.IndexFlatL2(dim)
    flat.add(np.ascontiguousarray(index_vecs, dtype=np.float32))
    _, nbrs = flat.search(np.ascontiguousarray(queries, dtype=np.float32), args.k)
    return nbrs.astype(np.int32)


def build_index(data):
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.init_index(
        max_elements=len(data),
        ef_construction=args.ef_construction,
        M=args.M,
    )
    idx.add_items(data, np.arange(len(data), dtype=np.int32))
    return idx


def load_fresh(path, n_elements):
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.load_index(path, max_elements=n_elements)
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


def save_params(displacement_magnitude):
    rows = [
        {"param": "dataset",                "value": "YFCC-DINO"},
        {"param": "experiment_type",        "value": "continuous_drift"},
        {"param": "dim",                    "value": dim},
        {"param": "n_index",                "value": args.n_index},
        {"param": "n_queries",              "value": args.n_queries},
        {"param": "n_warmup",               "value": args.n_warmup},
        {"param": "shift_sigmas",           "value": str(args.shift_sigmas)},
        {"param": "displacement_magnitude", "value": round(float(displacement_magnitude), 4)},
        {"param": "ef_sweep",               "value": str(args.ef_sweep)},
        {"param": "M",                      "value": args.M},
        {"param": "ef_construction",        "value": args.ef_construction},
        {"param": "k",                      "value": args.k},
        {"param": "pool_sizes",             "value": str(args.pool_sizes)},
        {"param": "ef_build",               "value": args.ef_build},
        {"param": "k_build",                "value": args.k_build},
        {"param": "drift_direction",        "value": "mean_2007_to_mean_2013"},
        {"param": "rng_seed",               "value": 42},
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
    n_index_label = f"{args.n_index:,}" if args.n_index else "all"
    print(f"YFCC pool routing continuous-drift experiment")
    print(f"n_index={n_index_label}  n_queries={args.n_queries}  n_warmup={args.n_warmup}  k={k}")
    print(f"pool_sizes={args.pool_sizes}  ef_build={args.ef_build}  k_build={args.k_build}")
    print(f"shift_sigmas={args.shift_sigmas}")
    print(f"continuous drift mode: pool built once from sigma={args.shift_sigmas[0]} warmup, "
          f"reused across all sigmas\n")

    emb, meta = load_embeddings()
    direction, magnitude = compute_drift_direction(emb, meta)
    index_vecs, query_sets = build_splits(emb, meta, direction, magnitude)
    del emb

    n_index_actual = len(index_vecs)

    print(f"\nbuilding HNSW index on {n_index_actual:,} vectors...")
    base_idx = build_index(index_vecs)
    index_path = os.path.join(out_dir, "base_index.bin")
    base_idx.save_index(index_path)
    print(f"  saved to {index_path}")

    print("\ncomputing ground truth (faiss)...")
    ground_truths = {}
    for sigma in args.shift_sigmas:
        print(f"  sigma={sigma}")
        ground_truths[sigma] = compute_gt(index_vecs, query_sets[sigma])
    del index_vecs

    all_results = {}
    pool_log    = []

    # no_adaptation baseline: one fresh index per ef, queries fed in sigma order
    print(f"\n{'='*60}")
    print("strategy: no_adaptation  (continuous — one run per ef)")
    print(f"{'='*60}")
    for ef in args.ef_sweep:
        idx = load_fresh(index_path, n_index_actual)
        cumulative_query = 0
        print(f"\n  ef={ef}")
        for sigma in args.shift_sigmas:
            eval_queries = query_sets[sigma][args.n_warmup:]
            eval_gt      = ground_truths[sigma][args.n_warmup:]
            results = run_queries_plain(idx, eval_queries, eval_gt, ef)
            all_results[("no_adaptation", ef, sigma)] = results
            mean_r  = np.mean([r["recall"]       for r in results])
            mean_bl = np.mean([r["bl_entry_dist"] for r in results])
            cumulative_query += len(results)
            print(f"    sigma={sigma:>5}  recall={mean_r:.4f}  bl_entry={mean_bl:.2f}"
                  f"  cumulative_q={cumulative_query}")

    # pool routing: pool built once from sigma=0, reused across all sigmas
    for pool_size in args.pool_sizes:
        strategy = f"pool_{pool_size}"
        print(f"\n{'='*60}")
        print(f"strategy: {strategy}  (continuous — pool from sigma={args.shift_sigmas[0]})")
        print(f"{'='*60}")

        for ef in args.ef_sweep:
            idx = load_fresh(index_path, n_index_actual)
            router = PoolRouter(
                idx,
                pool_size=pool_size,
                ef_build=args.ef_build,
                k_build=args.k_build,
            )

            warmup_queries = query_sets[args.shift_sigmas[0]][:args.n_warmup]
            print(f"\n  ef={ef}  building pool from {args.n_warmup} warmup queries "
                  f"at sigma={args.shift_sigmas[0]}...")
            n_collected = router.build_pool(warmup_queries)
            print(f"  pool built: {n_collected} nodes")

            pool_log.append({
                "strategy":            strategy,
                "ef":                  ef,
                "pool_size_requested": pool_size,
                "pool_size_achieved":  n_collected,
                "warmup_sigma":        args.shift_sigmas[0],
                "n_warmup":            args.n_warmup,
                "ef_build":            args.ef_build,
            })

            cumulative_query = 0
            for sigma in args.shift_sigmas:
                eval_queries = query_sets[sigma][args.n_warmup:]
                eval_gt      = ground_truths[sigma][args.n_warmup:]
                results = run_queries_routed(router, eval_queries, eval_gt, ef)
                all_results[(strategy, ef, sigma)] = results
                mean_r  = np.mean([r["recall"]       for r in results])
                mean_bl = np.mean([r["bl_entry_dist"] for r in results])
                cumulative_query += len(results)
                print(f"    sigma={sigma:>5}  recall={mean_r:.4f}  bl_entry={mean_bl:.2f}"
                      f"  cumulative_q={cumulative_query}")

    print("\nsaving results...")
    save_params(magnitude)
    save_summary(all_results)
    save_per_query(all_results)
    save_pool_log(pool_log)
    print(f"\ndone - results in {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
