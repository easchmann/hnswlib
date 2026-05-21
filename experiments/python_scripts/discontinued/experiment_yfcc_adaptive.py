# Adaptive HNSW experiment on YFCC directional drift.
# based on experiment_yfcc_directional_drift.py but adds adaptation:
# after a warm-up batch of queries at each sigma level, drift is detected with a sliding window of base layer entry distances and a
# structural updates is applied if the threshold is reached
#
# Three strategies are compared:
# option1 - DirectedEdgeController (add edges toward centroid)
# option2 - PromotionController (promote centroid node to layer 1)
# option3 - RewireController (recompute local neighbourhood)
#
# For each sigma level the script records recall before and after adaptation and the update cost in distance computations.
#
# Saves:
#   summary_adaptive.csv     - mean recall per (strategy, sigma, ef)
#   per_query_adaptive.csv   - per-query results
#   update_log.csv           - when each strategy triggered and what it did
#
# python experiments/experiment_yfcc_adaptive.py --embeddings_path data/yfcc_sampled/embeddings/embeddings_float32.npy --metadata_path data/yfcc_sampled/embeddings/metadata.csv --out_dir results_yfcc_adaptive

import os
import argparse
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hnswlib
from experiments.controllers.drift_adaptation import DirectedEdgeController, PromotionController, RewireController, measure_baseline, run_query_batch


rng = np.random.default_rng(42)

parser = argparse.ArgumentParser()
parser.add_argument("--embeddings_path", required=True)
parser.add_argument("--metadata_path",   required=True)
parser.add_argument("--n_index",         type=int,   default=None)
parser.add_argument("--n_queries",       type=int,   default=1_000)
parser.add_argument("--n_warmup",        type=int,   default=200,
                    help="queries fed to controller before evaluation at each sigma")
parser.add_argument("--shift_sigmas",    type=float, nargs="+",
                    default=[0, 1, 2, 3, 4, 6, 8, 10, 12, 16])
parser.add_argument("--ef_sweep",        type=int,   nargs="+",
                    default=[10, 20, 50, 100, 200, 500])
parser.add_argument("--M",               type=int,   default=16)
parser.add_argument("--ef_construction", type=int,   default=200)
parser.add_argument("--k",               type=int,   default=10)
parser.add_argument("--window_size",     type=int,   default=200)
parser.add_argument("--threshold_factor",type=float, default=1.5)
parser.add_argument("--out_dir",         default="results_yfcc_adaptive")
args = parser.parse_args()

dim = 384
os.makedirs(args.out_dir, exist_ok=True)


# --- loading -----------------------------------------------------------------

def load_embeddings():
    print(f"loading embeddings...")
    emb  = np.load(args.embeddings_path, mmap_mode='r')
    meta = pd.read_csv(args.metadata_path)
    assert len(emb) == len(meta), "embedding/metadata length mismatch"
    print(f"  {len(emb):,} embeddings")
    return emb, meta


def compute_drift_direction(emb, meta):
    mean_2007 = np.array(emb[(meta['year'] == 2007).values], dtype=np.float32).mean(axis=0)
    mean_2013 = np.array(emb[(meta['year'] == 2013).values], dtype=np.float32).mean(axis=0)
    direction = mean_2013 - mean_2007
    magnitude = np.linalg.norm(direction)
    direction /= magnitude
    print(f"  drift direction magnitude: {magnitude:.3f}")
    return direction.astype(np.float32), magnitude


def build_splits(emb, direction, displacement_magnitude):
    n_total = len(emb)
    if args.n_index and n_total > args.n_index:
        chosen = rng.choice(n_total, args.n_index, replace=False)
        chosen.sort()
        index_vecs = np.array(emb[chosen], dtype=np.float32)
    else:
        index_vecs = np.array(emb, dtype=np.float32)
    print(f"index: {len(index_vecs):,} vectors")

    q_idx      = rng.choice(n_total, args.n_queries, replace=False)
    q_idx.sort()
    query_base = np.array(emb[q_idx], dtype=np.float32)

    query_sets = {}
    for sigma in args.shift_sigmas:
        shift = direction * sigma * displacement_magnitude
        query_sets[sigma] = query_base + shift
    return index_vecs, query_sets


# --- ground truth ------------------------------------------------------------

def compute_gt(index_vecs, queries):
    import faiss
    index = faiss.IndexFlatL2(dim)
    index.add(np.ascontiguousarray(index_vecs, dtype=np.float32))
    _, neighbours = index.search(
        np.ascontiguousarray(queries, dtype=np.float32), args.k
    )
    return neighbours.astype(np.int32)


# --- index -------------------------------------------------------------------

def build_index(data):
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.init_index(
        max_elements=len(data),
        ef_construction=args.ef_construction,
        M=args.M
    )
    idx.add_items(data, np.arange(len(data), dtype=np.int32))
    return idx


# --- saving ------------------------------------------------------------------

def save_summary(all_results, path):
    rows = []
    for (strategy, sigma, ef), results in sorted(all_results.items()):
        def avg(field):
            return float(np.nanmean([r[field] for r in results]))
        l1 = [r["layer_visits"][1] for r in results if len(r["layer_visits"]) > 1]
        rows.append({
            "strategy":                     strategy,
            "shift_sigma":                  sigma,
            "ef_search":                    ef,
            "mean_recall":                  avg("recall"),
            "mean_ep_distance":             avg("ep_dist"),
            "mean_bl_entry_distance":       avg("bl_entry_dist"),
            "mean_ul_dist_comps":           avg("ul_dist_comps"),
            "mean_layer1_visits":           float(np.mean(l1)) if l1 else float("nan"),
            "mean_base_visited":            avg("base_visited"),
            "mean_base_dist_comps":         avg("base_dist_comps"),
            "mean_candidates_remaining":    avg("candidates_remaining"),
            "n_queries":                    len(results),
        })
    df = pd.DataFrame(rows).round(4)
    df.to_csv(path, index=False)
    print(f"  saved {path}")

    # quick recall table per strategy
    for strategy in df["strategy"].unique():
        sub = df[df["strategy"] == strategy]
        pivot = sub.pivot(index="shift_sigma", columns="ef_search",
                          values="mean_recall").round(3)
        print(f"\n{strategy} recall@k:")
        print(pivot.to_string())


def save_per_query(all_results, path):
    rows = []
    for (strategy, sigma, ef), results in sorted(all_results.items()):
        for i, r in enumerate(results):
            row = {
                "strategy":             strategy,
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
            }
            for layer, visits in enumerate(r["layer_visits"]):
                row[f"layer{layer}_visits"] = visits
            rows.append(row)
    pd.DataFrame(rows).round(6).to_csv(path, index=False)
    print(f"  saved {path}")


def save_update_log(controllers, path):
    rows = []
    for strategy, controller in controllers.items():
        for entry in controller.update_log:
            rows.append({"strategy": strategy, **entry})
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  saved {path}")


# --- main --------------------------------------------------------------------

def main():
    print(f"YFCC adaptive drift experiment")
    print(f"n_queries={args.n_queries}  n_warmup={args.n_warmup}  k={args.k}")
    print(f"shift_sigmas={args.shift_sigmas}")
    print(f"ef_sweep={args.ef_sweep}\n")

    emb, meta = load_embeddings()
    direction, displacement_magnitude = compute_drift_direction(emb, meta)
    index_vecs, query_sets = build_splits(emb, direction, displacement_magnitude)
    del emb

    print(f"\nbuilding index on {len(index_vecs):,} vectors...")
    idx = build_index(index_vecs)
    print(f"  done  (M={args.M}, ef_construction={args.ef_construction})")

    print(f"\ncomputing ground truth...")
    ground_truths = {}
    for sigma, queries in sorted(query_sets.items()):
        ground_truths[sigma] = compute_gt(index_vecs, queries)
    print("  done")

    # measure baseline bl_entry_dist at sigma=0 for drift detection threshold
    print(f"\nmeasuring baseline...")
    baseline = measure_baseline(idx, query_sets[0.0], ef=200, k=args.k)
    print(f"baseline bl_entry_dist: {baseline:.1f}")
    print(f"detection threshold: {baseline * args.threshold_factor:.1f}  "
          f"(factor={args.threshold_factor})")

    # initialise one controller per strategy: each gets its own fresh index copy so the strategies don't interfere
    strategies = {
        "no_adaptation": None,
        "option1_directed_edges": DirectedEdgeController,
        "option2_promotion": PromotionController,
        "option3_rewire": RewireController,
    }

    print(f"\nrunning experiment: {len(query_sets)} sigma levels "
          f"x {len(args.ef_sweep)} ef values x {len(strategies)} strategies")

    all_results  = {}

    for strategy_name, ControllerClass in strategies.items():
        print(f"\n{'='*60}")
        print(f"strategy: {strategy_name}")
        print(f"{'='*60}")

        # rebuild the index for each strategy so they start from identical state
        print("  rebuilding index...")
        idx = build_index(index_vecs)

        controller = None
        if ControllerClass is not None:
            controller = ControllerClass(idx, baseline, window_size=args.window_size, threshold_factor=args.threshold_factor)

        for ef in args.ef_sweep:
            print(f"\n  ef={ef}")
            for sigma, queries in sorted(query_sets.items()):
                # warm-up: feed queries into the controller
                # this simulates an online system where recent queries drive adaptation
                if controller is not None:
                    run_query_batch(idx, queries[:args.n_warmup], ground_truths[sigma][:args.n_warmup], k=args.k, ef=ef, controller=controller)

                    if controller.drift_detected():
                        controller.update(queries[:args.n_warmup], sigma=sigma)

                # evaluation on remaining queries
                eval_queries = queries[args.n_warmup:]
                eval_gt = ground_truths[sigma][args.n_warmup:]
                results = run_query_batch(idx, eval_queries, eval_gt, k=args.k, ef=ef)

                all_results[(strategy_name, sigma, ef)] = results

                mean_r = np.mean([r["recall"] for r in results])
                mean_bl = np.mean([r["bl_entry_dist"] for r in results])
                print(f"sigma={sigma:>5} recall={mean_r:.4f} bl_entry={mean_bl:.1f}")

        # save this strategy's controller log
        if controller is not None:
            log_path = os.path.join(args.out_dir, f"update_log_{strategy_name}.csv")
            pd.DataFrame(controller.update_log).to_csv(log_path, index=False)
            print(f"  update log: {len(controller.update_log)} updates → {log_path}")

    print("\nsaving results...")
    save_summary(all_results, os.path.join(args.out_dir, "summary_adaptive.csv"))
    save_per_query(all_results, os.path.join(args.out_dir, "per_query_adaptive.csv"))
    print(f"\ndone — results in {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()