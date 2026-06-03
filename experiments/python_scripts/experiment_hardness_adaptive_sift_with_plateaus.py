# Comparison experiment: HardnessAdaptiveController vs PoolAndRewire vs baseline
# on the SIFT1B hardness dataset (small/fast variant).
#
# Strategies compared:
#   no_adaptation      — standard HNSW search
#   poolAndRewire      — existing PoolAndRewireController (spatial centroid drift detection)
#   hardness_adaptive  — new controller: hard-query pool + difficulty-triggered rewiring
#                        + adaptive ef escalation
#
# the difference to the previous approach is that instead of having constant drift, we can introduce
# plateaus to closer mimic a real production system that is not constantly in adaptation mode. 
#
# Usage:
#   python experiment_hardness_adaptive_sift_with_plateaus.py \
#       --data_dir data/sift_hardness \
#       --n_bins 10 --n_queries_per_bin 5000 --ef_sweep 10 50 100 \
#       --drift_schedule "0:500,3:500,3:300,7:500,7:300,9:500"

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
from experiments.controllers.hardness_adaptive import HardnessAdaptiveController, run_query_batch_hardness

rng = np.random.default_rng(42)

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",              required=True)
parser.add_argument("--n_bins",                type=int,   default=10)
parser.add_argument("--n_queries_per_bin",     type=int,   default=5000)
parser.add_argument("--drift_schedule",        type=str)
parser.add_argument("--ef_sweep",              type=int,   nargs="+", default=[10, 50, 100])
parser.add_argument("--k",                     type=int,   default=10)
# poolAndRewire params
parser.add_argument("--alpha",                 type=float, default=1.1)
parser.add_argument("--max_layer",             type=int,   default=3)
parser.add_argument("--queries_per_rewire",    type=int,   default=200)
parser.add_argument("--cooldown",              type=int,   default=50)
parser.add_argument("--window_size",           type=int,   default=200)
parser.add_argument("--max_pool_size",         type=int,   default=50)
# hardness_adaptive params
parser.add_argument("--hard_percentile",       type=float, default=75)
parser.add_argument("--warmup_ef",             type=int,   default=50)
parser.add_argument("--hard_rewire_cooldown",  type=int,   default=10)
parser.add_argument("--hard_max_pool_size",    type=int,   default=50)
parser.add_argument("--pool_seed_ef",          type=int,   default=20)
parser.add_argument("--escalation_window",     type=int,   default=50)
parser.add_argument("--escalation_trigger",    type=float, default=0.3)
parser.add_argument("--escalation_factor",     type=int,   default=3)
parser.add_argument("--out_dir", default="results_hardness_adaptive_sift")
args = parser.parse_args()

k = args.k
os.makedirs(args.out_dir, exist_ok=True)

# helper to parse bin/plateau argument
def parse_schedule(s, bins):
    phases = []
    for entry in s.split(","):
        bin, n = entry.strip().split(":")
        bin, n = int(bin), int(n)
        assert 0<=bin < len(bins)
        phases.append({"bin": bin, "n_queries": n, "is_plateau": False})
    #mark plateaus (any bin index that appears more than once)
    seen = {}
    first_seen = set()
    for p in phases:
        seen[p["bin"]] = seen.get(p["bin"],0)+1
    for p in phases:
        if seen[p["bin"]] > 1:
            if p["bin"] in first_seen:
                p["is_plateau"] = True
            else:
                first_seen.add(p["bin"])
    return phases


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


def save_params(n_index, n_queries_total, dim, bin_edges):
    rows = [
        {"param": "dataset",                "value": "SIFT1B-hardness"},
        {"param": "experiment_type",        "value": "hardness_adaptive_comparison"},
        {"param": "dim",                    "value": dim},
        {"param": "n_index",                "value": n_index},
        {"param": "n_queries_total",        "value": n_queries_total},
        {"param": "n_bins",                 "value": args.n_bins},
        {"param": "hardness_bin_edges",     "value": str(list(bin_edges.round(4)))},
        {"param": "ef_sweep",               "value": str(args.ef_sweep)},
        {"param": "k",                      "value": k},
        {"param": "n_queries_per_bin",      "value": args.n_queries_per_bin},
        # poolAndRewire
        {"param": "alpha",                  "value": args.alpha},
        {"param": "max_layer",              "value": args.max_layer},
        {"param": "queries_per_rewire",     "value": args.queries_per_rewire},
        {"param": "cooldown",               "value": args.cooldown},
        {"param": "window_size",            "value": args.window_size},
        {"param": "max_pool_size",          "value": args.max_pool_size},
        # hardness_adaptive
        {"param": "hard_percentile",        "value": args.hard_percentile},
        {"param": "warmup_ef",              "value": args.warmup_ef},
        {"param": "hard_rewire_cooldown",   "value": args.hard_rewire_cooldown},
        {"param": "hard_max_pool_size",     "value": args.hard_max_pool_size},
        {"param": "pool_seed_ef",           "value": args.pool_seed_ef},
        {"param": "escalation_window",      "value": args.escalation_window},
        {"param": "escalation_trigger",     "value": args.escalation_trigger},
        {"param": "escalation_factor",      "value": args.escalation_factor},
        {"param": "drift_schedule",         "value": args.drift_schedule},
        {"param": "rng_seed",               "value": 42},
    ]
    path = os.path.join(args.out_dir, "params.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  saved {path}")


def save_summary(all_results, bin_edges, phase_meta):
    rows = []
    for (strategy, ef, phase_idx), results in sorted(all_results.items()):
        def avg(f): return float(np.nanmean([r[f] for r in results]))
        def avg_opt(f): return float(np.nanmean([r[f] for r in results])) if f in results[0] else float("nan")
        l1 = [r["layer_visits"][1] for r in results if len(r["layer_visits"]) > 1]
        actual_bin = phase_meta[phase_idx]["bin"]
        rows.append({
            "strategy":               strategy,
            "ef_search":              ef,
            "phase_idx":              phase_idx,
            "hardness_bin":           actual_bin,
            "is_plateau":             phase_meta[phase_idx]["is_plateau"],
            "hardness_bin_lo":        float(bin_edges[actual_bin]),
            "hardness_bin_hi":        float(bin_edges[actual_bin + 1]),
            "mean_recall":            avg("recall"),
            "mean_ep_distance":       avg("ep_dist"),
            "mean_bl_entry_distance": avg("bl_entry_dist"),
            "mean_ul_dist_comps":     avg("ul_dist_comps"),
            "mean_layer1_visits":     float(np.mean(l1)) if l1 else float("nan"),
            "mean_base_visited":      avg("base_visited"),
            "mean_base_dist_comps":   avg("base_dist_comps"),
            "mean_candidates_remaining": avg("candidates_remaining"),
            "n_queries":              len(results),
            "mean_t_query_ms":        avg_opt("t_query_ms"),
            "mean_t_global_knn_ms":   avg_opt("t_global_knn_ms"),
            "mean_t_orig_knn_ms":     avg_opt("t_orig_knn_ms"),
            "mean_t_pool_scan_ms":    avg_opt("t_pool_scan_ms"),
            "mean_t_pool_knn_ms":     avg_opt("t_pool_knn_ms"),
            "mean_t_adapt_ms":        avg_opt("t_adapt_ms"),
        })

    df = pd.DataFrame(rows).round(4)
    path = os.path.join(args.out_dir, "summary.csv")
    df.to_csv(path, index=False)
    print(f"  saved {path}")

    for strategy in df["strategy"].unique():
        sub = df[df["strategy"] == strategy]
        pivot = sub.pivot(index="phase_idx", columns="ef_search",
                          values="mean_recall").round(3)
        print(f"\n{strategy} recall@{k}:")
        print(pivot.to_string())

    base = df[df["strategy"] == "no_adaptation"].set_index(
        ["phase_idx", "ef_search"])["mean_recall"]
    for strategy in df["strategy"].unique():
        if strategy == "no_adaptation":
            continue
        adap  = df[df["strategy"] == strategy].set_index(
            ["phase_idx", "ef_search"])["mean_recall"]
        delta = (adap - base).round(3).unstack("ef_search")
        print(f"\nrecall delta ({strategy} - no_adaptation):")
        print(delta.to_string())

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


def save_per_query(all_results, phase_meta):
    rows = []
    for (strategy, ef, phase_idx), results in sorted(all_results.items()):
        actual_bin = phase_meta[phase_idx]["bin"]
        for i, r in enumerate(results):
            row = {
                "strategy":             strategy,
                "ef_search":            ef,
                "phase_idx":            phase_idx,
                "hardness_bin":         actual_bin,
                "is_plateau":           phase_meta[phase_idx]["is_plateau"],
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
                "t_global_knn_ms":      r.get("t_global_knn_ms", float("nan")),
                "t_orig_knn_ms":        r.get("t_orig_knn_ms",  float("nan")),
                "t_pool_scan_ms":       r.get("t_pool_scan_ms", float("nan")),
                "t_pool_knn_ms":        r.get("t_pool_knn_ms", float("nan")),
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
    print("SIFT hardness adaptive-controller comparison experiment")
    print(f"n_bins={args.n_bins}  ef_sweep={args.ef_sweep}  k={k}")
    print(f"n_queries_per_bin={args.n_queries_per_bin}\n")

    queries, gt, scores = load_dataset()
    dim = queries.shape[1]

    bins = np.array_split(np.arange(len(queries)), args.n_bins)
    bin_edges = np.array(
        [scores[b[0]] for b in bins] + [scores[bins[-1][-1]]], dtype=np.float32
    )
    print(f"bin sizes (before cap): {[len(b) for b in bins]}")
    print(f"hardness bin edges: {bin_edges.round(3)}")
    print(f"capping each bin to {args.n_queries_per_bin} queries\n")

    bins      = [b[:args.n_queries_per_bin] for b in bins]
    #warmup_queries = queries[bins[0]]
    #eval_bins      = bins[1:]
    schedule = parse_schedule(args.drift_schedule, bins)
    warmup_queries = queries[bins[schedule[0]["bin"]]] #first phase is always warmup
    eval_phases = schedule[1:]
    phase_meta = {i: p for i, p in enumerate(eval_phases)}

    def sample_phase(bin_idx, n_queries):
        available = bins[bin_idx]
        if n_queries <= len(available):
            chosen = rng.choice(available, size=n_queries, replace=False)
        else:
            chosen = rng.choice(available, size=n_queries, replace=True)
        return chosen

    # precompute once so all strategies see the same query stream
    phase_samples = [sample_phase(p["bin"], p["n_queries"]) for p in eval_phases]


    params_path = os.path.join(args.data_dir, "params.json")
    if os.path.exists(params_path):
        with open(params_path) as f:
            n_index = json.load(f)["n_index"]
    else:
        n_index = int(input("n_index not found in params.json, enter manually: "))

    all_results = {}
    adapt_log   = []

    # no_adaptation baseline
    print(f"{'='*60}")
    print("strategy: no_adaptation")
    print(f"{'='*60}")
    for ef in args.ef_sweep:
        idx = load_index(n_index, dim)
        print(f"\n  ef={ef}")
        for phase_idx, phase in enumerate(eval_phases):
            bin_idx = phase_samples[phase_idx]
            results = run_query_batch_hardness(idx, queries[bin_idx], gt[bin_idx], k, ef)
            all_results[("no_adaptation", ef, phase_idx)] = results
            mean_r = np.mean([r["recall"] for r in results])
            mean_d = np.mean([r["base_dist_comps"] for r in results])
            print(f"    phase={phase_idx}  bin={phase['bin']}  plateau={phase['is_plateau']}  "
                  f"hardness=[{bin_edges[phase['bin']]:.3f},{bin_edges[phase['bin']+1]:.3f}]  "
                  f"recall={mean_r:.4f}  mean_dist_comps={mean_d:.1f}")


    # poolAndRewire
    print(f"\n{'='*60}")
    print("strategy: poolAndRewire")
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
        print(f"\n  ef={ef}  (controller initialized on warmup bin)")
        for phase_idx, phase in enumerate(eval_phases):
            bin_idx = phase_samples[phase_idx]
            results = run_query_batch(idx, queries[bin_idx], gt[bin_idx],
                                      k=k, ef=ef, controller=ctrl, use_pool=True)
            all_results[("poolAndRewire", ef, phase_idx)] = results
            mean_r = np.mean([r["recall"] for r in results])
            mean_d = np.mean([r["base_dist_comps"] for r in results])
            print(f"    phase={phase_idx}  bin={phase['bin']}  plateau={phase['is_plateau']}  "
                  f"hardness=[{bin_edges[phase['bin']]:.3f},{bin_edges[phase['bin']+1]:.3f}]  "
                  f"recall={mean_r:.4f}  mean_dist_comps={mean_d:.1f}  "
                  f"updates={ctrl.update_count}")
        n_eval_queries = sum(p["n_queries"] for p in eval_phases)
        amortized_ms = ctrl.total_adapt_time_ms / max(n_eval_queries, 1)
        print(f"  adapt events={ctrl.update_count}  total_adapt={ctrl.total_adapt_time_ms:.0f}ms  "
              f"amortized={amortized_ms:.4f}ms/query")
        for entry in ctrl.update_log:
            entry["ef"] = ef
            entry["strategy"] = "poolAndRewire"
            adapt_log.append(entry)


    # hardness_adaptive
    print(f"\n{'='*60}")
    print("strategy: hardness_adaptive")
    print(f"{'='*60}")
    for ef in args.ef_sweep:
        idx  = load_index(n_index, dim)
        ctrl = HardnessAdaptiveController(
            idx,
            reference_queries=warmup_queries,
            k=k,
            hard_percentile=args.hard_percentile,
            warmup_ef=args.warmup_ef,
            alpha=args.alpha,
            max_layer=args.max_layer,
            hard_rewire_cooldown=args.hard_rewire_cooldown,
            max_pool_size=args.hard_max_pool_size,
            pool_seed_ef=args.pool_seed_ef,
            escalation_window=args.escalation_window,
            escalation_trigger=args.escalation_trigger,
            escalation_factor=args.escalation_factor,
        )
        print(f"\n  ef={ef}")
        for phase_idx, phase in enumerate(eval_phases):
            bin_idx = phase_samples[phase_idx]
            results = run_query_batch_hardness(idx, queries[bin_idx], gt[bin_idx],
                                               k, ef, controller=ctrl)
            all_results[("hardness_adaptive", ef, phase_idx)] = results
            mean_r = np.mean([r["recall"] for r in results])
            mean_d = np.mean([r["base_dist_comps"] for r in results])
            print(f"    phase={phase_idx}  bin={phase['bin']}  plateau={phase['is_plateau']}  "
                  f"hardness=[{bin_edges[phase['bin']]:.3f},{bin_edges[phase['bin']+1]:.3f}]  "
                  f"recall={mean_r:.4f}  mean_dist_comps={mean_d:.1f}  "
                  f"rewires={ctrl.update_count}  escalations={ctrl.escalation_count}  "
                  f"pool={len(ctrl._pool)}")
        n_eval_queries = sum(p["n_queries"] for p in eval_phases)
        amortized_ms = ctrl.total_adapt_time_ms / max(n_eval_queries, 1)
        print(f"  adapt events={ctrl.update_count}  escalations={ctrl.escalation_count}  "
              f"total_adapt={ctrl.total_adapt_time_ms:.0f}ms  amortized={amortized_ms:.4f}ms/query")
        for entry in ctrl.update_log:
            entry["ef"] = ef
            entry["strategy"] = "hardness_adaptive"
            adapt_log.append(entry)

    print("\nsaving results...")
    save_params(n_index, len(queries), dim, bin_edges)
    save_summary(all_results, bin_edges, phase_meta)
    save_per_query(all_results, phase_meta)
    save_adapt_log(adapt_log)
    print(f"\ndone — {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
