# Ablation study for PoolAndRewireController on YFCC continuous drift.
#
# Isolates the contribution of the three mechanisms:
#   rewire  — upper-layer rewire_for_query on recent queries
#   pool    — promoted entry-point pool with union-merge search
#   highway — directed highway edge from global EP toward drift centroid
#
# Variants (use_rewire x use_highway x use_pool):
#   no_adaptation  — vanilla HNSW
#   pool_only      — pool + union-merge, no rewire, no highway
#   rewire_only    — rewire, no pool, no highway
#   highway_only   — highway edge, no pool, no rewire
#   pool_rewire    — pool + rewire, no highway
#   pool_highway   — pool + highway, no rewire
#   full           — all three mechanisms
#
# Usage:
#   python experiment_poolAndRewire_yfcc_ablation.py \
#       --embeddings_path data/yfcc/embeddings.npy \
#       --metadata_path   data/yfcc/metadata.csv \
#       --n_index 5000000

import os
import sys
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
parser.add_argument("--embeddings_path",    required=True)
parser.add_argument("--metadata_path",      required=True)
parser.add_argument("--n_index",            type=int,   default=None)
parser.add_argument("--n_queries",          type=int,   default=1_000)
parser.add_argument("--n_warmup",           type=int,   default=200)
parser.add_argument("--shift_sigmas",       type=float, nargs="+",
                    default=[0, 1, 2, 3, 4, 6, 8, 10, 12, 16])
parser.add_argument("--ef_sweep",           type=int,   nargs="+",
                    default=[10, 20, 50, 100, 200, 500])
parser.add_argument("--M",                  type=int,   default=16)
parser.add_argument("--ef_construction",    type=int,   default=200)
parser.add_argument("--k",                  type=int,   default=10)
parser.add_argument("--alpha",              type=float, default=1.1)
parser.add_argument("--max_layer",          type=int,   default=3)
parser.add_argument("--queries_per_rewire", type=int,   default=200)
parser.add_argument("--cooldown",           type=int,   default=50)
parser.add_argument("--window_size",        type=int,   default=200)
parser.add_argument("--max_pool_size",      type=int,   default=50)
parser.add_argument("--out_dir", default="results_poolAndRewire_yfcc_ablation")
args = parser.parse_args()

dim     = 384
k       = args.k
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

# (use_pool, use_rewire, use_highway)
VARIANTS = [
    ("pool_only",    True,  False, False),
    ("rewire_only",  False, True,  False),
    ("highway_only", False, False, True),
    ("pool_rewire",  True,  True,  False),
    ("pool_highway", True,  False, True),
    ("full",         True,  True,  True),
]


def load_embeddings():
    print("loading embeddings...")
    emb  = np.load(args.embeddings_path, mmap_mode='r')
    meta = pd.read_csv(args.metadata_path)
    assert len(emb) == len(meta)
    print(f"  {len(emb):,} embeddings  dim={emb.shape[1]}")
    return emb, meta


def compute_drift_direction(emb, meta):
    mean_2007 = np.array(emb[(meta['year'] == 2007).values], dtype=np.float32).mean(axis=0)
    mean_2013 = np.array(emb[(meta['year'] == 2013).values], dtype=np.float32).mean(axis=0)
    direction = mean_2013 - mean_2007
    magnitude = np.linalg.norm(direction)
    direction /= magnitude
    print(f"drift direction 2007→2013  |displacement|={magnitude:.4f}")
    return direction.astype(np.float32), magnitude


def build_splits(emb, meta, direction, magnitude):
    n_total = len(emb)
    if args.n_index and n_total > args.n_index:
        chosen = rng.choice(n_total, args.n_index, replace=False)
        chosen.sort()
        index_vecs = np.array(emb[chosen], dtype=np.float32)
    else:
        index_vecs = np.array(emb, dtype=np.float32)
    print(f"index: {len(index_vecs):,} vectors")

    q_idx = rng.choice(n_total, args.n_queries + args.n_warmup, replace=False)
    q_idx.sort()
    query_base = np.array(emb[q_idx], dtype=np.float32)

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
    _, nbrs = flat.search(np.ascontiguousarray(queries, dtype=np.float32), k)
    return nbrs.astype(np.int32)


def build_index(data):
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.init_index(max_elements=len(data), ef_construction=args.ef_construction, M=args.M)
    idx.add_items(data, np.arange(len(data), dtype=np.int32))
    return idx


def load_fresh(path, n_elements):
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.load_index(path, max_elements=n_elements)
    return idx


def run_queries_plain(idx, queries, gt, ef):
    idx.set_ef(ef)
    results = []
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        pred, _ = idx.knn_query(q.reshape(1, -1), k=k)
        t_ms = (time.perf_counter() - t0) * 1000
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
            "t_query_ms":           t_ms,
            "t_pool_scan_ms":       0.0,
            "t_pool_knn_ms":        0.0,
            "t_orig_knn_ms":        t_ms,
            "t_adapt_ms":           0.0,
        })
    return results


def run_variant(name, use_pool, use_rewire, use_highway,
                index_path, n_index_actual, warmup_queries,
                query_sets, ground_truths, all_results):
    print(f"\n{'='*60}\nvariant: {name}  "
          f"(pool={use_pool} rewire={use_rewire} highway={use_highway})\n{'='*60}")

    for ef in args.ef_sweep:
        idx = load_fresh(index_path, n_index_actual)
        ctrl = PoolAndRewireController(
            idx,
            reference_queries=warmup_queries,
            window_size=args.window_size,
            alpha=args.alpha,
            max_layer=args.max_layer,
            queries_per_rewire=args.queries_per_rewire,
            cooldown=args.cooldown,
            max_pool_size=args.max_pool_size,
            use_pool=use_pool,
            use_rewire=use_rewire,
            use_highway=use_highway,
        )
        print(f"\n  ef={ef}")
        for sigma in args.shift_sigmas:
            eval_q  = query_sets[sigma][args.n_warmup:]
            eval_gt = ground_truths[sigma][args.n_warmup:]
            results = run_query_batch(
                idx, eval_q, eval_gt, k=k, ef=ef,
                controller=ctrl, use_pool=use_pool,
            )
            all_results[(name, ef, sigma)] = results
            mean_r = np.mean([r["recall"] for r in results])
            print(f"    sigma={sigma:>5}  recall={mean_r:.4f}"
                  f"  updates={ctrl.update_count}  pool={len(ctrl._pool)}")


def save_summary(all_results):
    rows = []
    for (strategy, ef, sigma), results in sorted(all_results.items()):
        def avg(f): return float(np.nanmean([r[f] for r in results]))
        l1 = [r["layer_visits"][1] for r in results if len(r["layer_visits"]) > 1]
        rows.append({
            "strategy":               strategy,
            "ef_search":              ef,
            "shift_sigma":            sigma,
            "mean_recall":            avg("recall"),
            "mean_ep_distance":       avg("ep_dist"),
            "mean_bl_entry_distance": avg("bl_entry_dist"),
            "mean_ul_dist_comps":     avg("ul_dist_comps"),
            "mean_layer1_visits":     float(np.mean(l1)) if l1 else float("nan"),
            "mean_base_visited":      avg("base_visited"),
            "mean_base_dist_comps":   avg("base_dist_comps"),
            "n_queries":              len(results),
            "mean_t_query_ms":        avg("t_query_ms"),
            "mean_t_pool_scan_ms":    avg("t_pool_scan_ms"),
            "mean_t_pool_knn_ms":     avg("t_pool_knn_ms"),
            "mean_t_orig_knn_ms":     avg("t_orig_knn_ms"),
            "mean_t_adapt_ms":        avg("t_adapt_ms"),
        })
    df = pd.DataFrame(rows).round(4)
    df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    base = df[df["strategy"] == "no_adaptation"].set_index(["shift_sigma", "ef_search"])["mean_recall"]
    for s in df["strategy"].unique():
        sub = df[df["strategy"] == s]
        pivot = sub.pivot(index="shift_sigma", columns="ef_search", values="mean_recall").round(3)
        print(f"\n{s} recall@{k}:")
        print(pivot.to_string())
        if s != "no_adaptation":
            delta = (df[df["strategy"] == s]
                     .set_index(["shift_sigma", "ef_search"])["mean_recall"] - base)
            print(f"  Δ vs no_adaptation: mean={delta.mean():+.4f}  "
                  f"min={delta.min():+.4f}  max={delta.max():+.4f}")


def save_params(magnitude):
    rows = [
        {"param": "dataset",                "value": "YFCC-DINO"},
        {"param": "experiment_type",        "value": "poolAndRewire_yfcc_ablation"},
        {"param": "dim",                    "value": dim},
        {"param": "n_index",                "value": args.n_index},
        {"param": "n_queries",              "value": args.n_queries},
        {"param": "n_warmup",               "value": args.n_warmup},
        {"param": "shift_sigmas",           "value": str(args.shift_sigmas)},
        {"param": "displacement_magnitude", "value": round(float(magnitude), 4)},
        {"param": "ef_sweep",               "value": str(args.ef_sweep)},
        {"param": "M",                      "value": args.M},
        {"param": "ef_construction",        "value": args.ef_construction},
        {"param": "k",                      "value": k},
        {"param": "alpha",                  "value": args.alpha},
        {"param": "max_layer",              "value": args.max_layer},
        {"param": "queries_per_rewire",     "value": args.queries_per_rewire},
        {"param": "cooldown",               "value": args.cooldown},
        {"param": "window_size",            "value": args.window_size},
        {"param": "max_pool_size",          "value": args.max_pool_size},
        {"param": "drift_direction",        "value": "mean_2007_to_mean_2013"},
        {"param": "rng_seed",               "value": 42},
    ]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "params.csv"), index=False)


def main():
    print(f"PoolAndRewire YFCC ablation")
    print(f"n_index={args.n_index}  n_queries={args.n_queries}  "
          f"n_warmup={args.n_warmup}  k={k}")
    print(f"shift_sigmas={args.shift_sigmas}")

    emb, meta = load_embeddings()
    direction, magnitude = compute_drift_direction(emb, meta)
    index_vecs, query_sets = build_splits(emb, meta, direction, magnitude)
    del emb

    n_index_actual = len(index_vecs)
    print(f"\nbuilding HNSW index on {n_index_actual:,} vectors...")
    base_idx = build_index(index_vecs)
    index_path = os.path.join(out_dir, "base_index.bin")
    base_idx.save_index(index_path)

    print("\ncomputing ground truth (faiss)...")
    ground_truths = {}
    for sigma in args.shift_sigmas:
        print(f"  sigma={sigma}")
        ground_truths[sigma] = compute_gt(index_vecs, query_sets[sigma])
    del index_vecs

    warmup_queries = query_sets[args.shift_sigmas[0]][:args.n_warmup]

    all_results = {}

    # no_adaptation baseline
    print(f"\n{'='*60}\nvariant: no_adaptation\n{'='*60}")
    for ef in args.ef_sweep:
        idx = load_fresh(index_path, n_index_actual)
        print(f"\n  ef={ef}")
        for sigma in args.shift_sigmas:
            eval_q  = query_sets[sigma][args.n_warmup:]
            eval_gt = ground_truths[sigma][args.n_warmup:]
            results = run_queries_plain(idx, eval_q, eval_gt, ef)
            all_results[("no_adaptation", ef, sigma)] = results
            mean_r = np.mean([r["recall"] for r in results])
            print(f"    sigma={sigma:>5}  recall={mean_r:.4f}")

    # ablation variants
    for name, use_pool, use_rewire, use_highway in VARIANTS:
        run_variant(name, use_pool, use_rewire, use_highway,
                    index_path, n_index_actual, warmup_queries,
                    query_sets, ground_truths, all_results)

    print("\nsaving results...")
    save_params(magnitude)
    save_summary(all_results)
    print(f"\ndone — {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
