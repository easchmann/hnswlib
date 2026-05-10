# Progressive neighbourhood expansion on YFCC directional drift.
#
# Each sigma level gets a fresh copy of the base index so adaptations are
# independent
#
# Saves summary.csv, per_query.csv, params.csv, update_log.csv
# summary.csv also prints a recall delta table (adapted - baseline).
#
# Usage:
#   python experiments/experiment_yfcc_progressive.py \
#       --embeddings_path data/yfcc_sampled/embeddings/embeddings_float32.npy \
#       --metadata_path   data/yfcc_sampled/embeddings/metadata.csv \
#       --n_index         500000 \
#       --out_dir         results_yfcc_progressive

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hnswlib
from experiments.controllers.drift_adaptation_anchor_2 import ProgressiveExpansionController, measure_baseline,run_query_batch


rng = np.random.default_rng(42)

parser = argparse.ArgumentParser()
parser.add_argument("--embeddings_path", required=True)
parser.add_argument("--metadata_path",required=True)
parser.add_argument("--n_index", type=int,   default=None)
parser.add_argument("--n_queries",  type=int,   default=1_000)
parser.add_argument("--n_warmup", type=int,   default=200)
parser.add_argument("--shift_sigmas", type=float, nargs="+",
                    default=[0, 1, 2, 3, 4, 6, 8, 10, 12, 16])
parser.add_argument("--ef_sweep",           type=int,   nargs="+",
                    default=[10, 20, 50, 100, 200, 500])
parser.add_argument("--M", type=int,   default=16)
parser.add_argument("--ef_construction",    type=int,   default=200)
parser.add_argument("--k",   type=int,   default=10)
parser.add_argument("--k_anchors", type=int,   default=4)
parser.add_argument("--k_nodes_per_anchor", type=int,   default=8)
parser.add_argument("--window_size",  type=int,   default=200)
parser.add_argument("--threshold_factor",   type=float, default=1.5)
parser.add_argument("--out_dir",  default="results_yfcc_progressive_2")
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


# GT

def compute_gt(index_vecs, queries):
    import faiss
    index = faiss.IndexFlatL2(dim)
    index.add(np.ascontiguousarray(index_vecs, dtype=np.float32))
    _, neighbours = index.search(
        np.ascontiguousarray(queries, dtype=np.float32), args.k
    )
    return neighbours.astype(np.int32)


# build index

def build_index(data):
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.init_index(
        max_elements=len(data),
        ef_construction=args.ef_construction,
        M=args.M
    )
    idx.add_items(data, np.arange(len(data), dtype=np.int32))
    return idx


def load_fresh_index(n_elements):
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.load_index(INDEX_SAVE_PATH, max_elements=n_elements)
    return idx


# save

def save_params(path):
    rows = [
        {"param": "dataset",              "value": "YFCC-DINO"},
        {"param": "dim",                  "value": dim},
        {"param": "n_index",              "value": args.n_index or "all"},
        {"param": "n_queries",            "value": args.n_queries},
        {"param": "n_warmup",             "value": args.n_warmup},
        {"param": "shift_sigmas",         "value": str(args.shift_sigmas)},
        {"param": "ef_sweep",             "value": str(args.ef_sweep)},
        {"param": "M",                    "value": args.M},
        {"param": "ef_construction",      "value": args.ef_construction},
        {"param": "k",                    "value": args.k},
        {"param": "k_anchors",            "value": args.k_anchors},
        {"param": "k_nodes_per_anchor",   "value": args.k_nodes_per_anchor},
        {"param": "window_size",          "value": args.window_size},
        {"param": "threshold_factor",     "value": args.threshold_factor},
        {"param": "drift_direction",      "value": "mean_2007_to_mean_2013"},
        {"param": "rng_seed",             "value": 42},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f" saved {path}")


def save_summary(all_results, path):
    rows = []
    for (strategy, sigma, ef), results in sorted(all_results.items()):
        def avg(field):
            return float(np.nanmean([r[field] for r in results]))
        l1 = [r["layer_visits"][1] for r in results if len(r["layer_visits"]) > 1]
        rows.append({
            "strategy": strategy,
            "shift_sigma": sigma,
            "ef_search":  ef,
            "mean_recall": avg("recall"),
            "mean_ep_distance": avg("ep_dist"),
            "mean_bl_entry_distance": avg("bl_entry_dist"),
            "mean_ul_dist_comps": avg("ul_dist_comps"),
            "mean_layer1_visits":  float(np.mean(l1)) if l1 else float("nan"),
            "mean_base_visited":  avg("base_visited"),
            "mean_base_dist_comps": avg("base_dist_comps"),
            "mean_candidates_remaining":  avg("candidates_remaining"),
            "n_queries": len(results),
        })

    df = pd.DataFrame(rows).round(4)
    df.to_csv(path, index=False)
    print(f" saved {path}")

    for strategy in df["strategy"].unique():
        sub = df[df["strategy"] == strategy]
        pivot = sub.pivot(index="shift_sigma", columns="ef_search", values="mean_recall").round(3)
        print(f"\n{strategy} recall@k:")
        print(pivot.to_string())

    # recall delta so the improvement is immediately visible
    base = df[df["strategy"] == "no_adaptation"].set_index(["shift_sigma", "ef_search"])["mean_recall"]
    adap = df[df["strategy"] == "progressive"].set_index(["shift_sigma", "ef_search"])["mean_recall"]
    if not base.empty and not adap.empty:
        delta = (adap - base).round(3).unstack("ef_search")
        print("\nrecall delta (progressive - no_adaptation), positive = improvement:")
        print(delta.to_string())


def save_per_query(all_results, path):
    rows = []
    for (strategy, sigma, ef), results in sorted(all_results.items()):
        for i, r in enumerate(results):
            row = {
                "strategy":  strategy,
                "shift_sigma": sigma,
                "ef_search": ef,
                "query_id":  i,
                "recall":   r["recall"],
                "ep_dist":  r["ep_dist"],
                "bl_entry_dist":   r["bl_entry_dist"],
                "ul_dist_comps":  r["ul_dist_comps"],
                "base_visited": r["base_visited"],
                "base_dist_comps":  r["base_dist_comps"],
                "candidates_remaining": r["candidates_remaining"],
                "lb_trace_final": r["lb_trace"][-1] if r["lb_trace"] else float("nan"),
            }
            for layer, visits in enumerate(r["layer_visits"]):
                row[f"layer{layer}_visits"] = visits
            rows.append(row)
    pd.DataFrame(rows).round(6).to_csv(path, index=False)
    print(f"  saved {path}")




def main():
    print("YFCC progressive expansion experiment")
    print(f"n_queries={args.n_queries}  n_warmup={args.n_warmup}  k={args.k}")
    print(f"k_anchors={args.k_anchors}  k_nodes_per_anchor={args.k_nodes_per_anchor}")
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

    print(f"\ncomputing ground truth...")
    ground_truths = {}
    for sigma, queries in sorted(query_sets.items()):
        print(f"  sigma={sigma}")
        ground_truths[sigma] = compute_gt(index_vecs, queries)
    print("  done")

    print(f"\nmeasuring baseline at sigma=0...")
    baseline_dist = measure_baseline(base_idx, query_sets[0.0], ef=200, k=args.k)
    print(f" baseline bl_entry_dist: {baseline_dist:.1f}")
    print(f"detection threshold: {baseline_dist * args.threshold_factor:.1f}")
    del base_idx

    n_elements  = len(index_vecs)
    all_results = {}
    update_log  = []

    for strategy_name, ControllerClass in [("no_adaptation", None), ("progressive", ProgressiveExpansionController)]:
        print(f"\n{'='*60}")
        print(f"strategy: {strategy_name}")
        print(f"{'='*60}")

        for sigma, queries in sorted(query_sets.items()):
            idx = load_fresh_index(n_elements)

            controller = None
            if ControllerClass is not None:
                controller = ControllerClass(idx, baseline_dist, window_size=args.window_size, threshold_factor=args.threshold_factor, k_anchors=args.k_anchors, k_nodes_per_anchor=args.k_nodes_per_anchor)

            adapted = False
            if controller is not None:
                run_query_batch(idx, queries[:args.n_warmup], ground_truths[sigma][:args.n_warmup], k=args.k, ef=200, controller=controller)

                if controller.drift_detected():
                    controller.update(queries[:args.n_warmup], sigma=sigma)
                    update_log.extend(controller.update_log)
                    adapted = True

                status = "adapted" if adapted else "no trigger"
                print(f"\n  sigma={sigma} [{status}]"
                      f"window_mean={controller.window_mean():.1f}")
            else:
                print(f"\n sigma={sigma}")

            eval_queries = queries[args.n_warmup:]
            eval_gt = ground_truths[sigma][args.n_warmup:]

            for ef in args.ef_sweep:
                results = run_query_batch(idx, eval_queries, eval_gt, k=args.k, ef=ef)
                all_results[(strategy_name, sigma, ef)] = results
                mean_r = np.mean([r["recall"] for r in results])
                mean_bl = np.mean([r["bl_entry_dist"] for r in results])
                print(f" ef={ef:>4} recall={mean_r:.4f}  bl_entry={mean_bl:.1f}")

    print("\nsaving results...")
    save_params(os.path.join(args.out_dir, "params.csv"))
    save_summary(all_results, os.path.join(args.out_dir, "summary.csv"))
    save_per_query(all_results, os.path.join(args.out_dir, "per_query.csv"))

    if update_log:
        log_path = os.path.join(args.out_dir, "update_log.csv")
        pd.DataFrame(update_log).to_csv(log_path, index=False)
        print(f"saved {log_path} ({len(update_log)} updates)")

    print(f"\ndone — results in {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()