# Ablation: pool improvements for HardnessAdaptiveController on SIFT hardness drift.
#
# Tests three targeted improvements over the original hardness_adaptive strategy:
#   redundancy_pool   — redundancy-based eviction (keeps pool diverse) vs. distance eviction
#   multi_ep          — union-merge from top-3 pool entries instead of top-1
#   continuous_ef     — continuous ef scaling with hard_fraction vs. binary on/off
#   ha_all            — all three improvements combined
#
# Strategies compared:
#   no_adaptation     — standard HNSW, fixed ef
#   ha_original       — pool + escalation only (rewiring disabled), distance eviction, top-1, binary ef
#   ha_redundancy     — same as ha_original but redundancy eviction
#   ha_multi_ep       — same as ha_original but top-3 pool entries
#   ha_continuous_ef  — same as ha_original but continuous ef escalation
#   ha_all            — redundancy + multi_ep + continuous_ef
#
# Usage:
#   python experiment_hardness_adaptive_pool_improvements.py \
#       --data_dir ../../data/sift_hardness \
#       --n_bins 10 --n_queries_per_bin 5000 --ef_sweep 10 50 100 \
#       --drift_schedule "0:500,3:500,7:500,9:500"

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, ".."))

import hnswlib
from experiments.controllers.hardness_adaptive import HardnessAdaptiveController, run_query_batch_hardness

rng = np.random.default_rng(42)

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",             required=True)
parser.add_argument("--n_bins",               type=int,   default=10)
parser.add_argument("--n_queries_per_bin",    type=int,   default=5000)
parser.add_argument("--drift_schedule",       type=str,   required=True)
parser.add_argument("--ef_sweep",             type=int,   nargs="+", default=[10, 50, 100])
parser.add_argument("--k",                    type=int,   default=10)
# shared controller params
parser.add_argument("--hard_percentile",      type=float, default=75)
parser.add_argument("--warmup_ef",            type=int,   default=50)
parser.add_argument("--hard_max_pool_size",   type=int,   default=100)
parser.add_argument("--pool_seed_ef",         type=int,   default=20)
parser.add_argument("--alpha",                type=float, default=1.1)
parser.add_argument("--max_layer",            type=int,   default=3)
parser.add_argument("--escalation_window",    type=int,   default=50)
parser.add_argument("--escalation_trigger",   type=float, default=0.3)
parser.add_argument("--escalation_factor",    type=int,   default=3)
parser.add_argument("--pool_top_k",           type=int,   default=3)
parser.add_argument("--out_dir", default="results_ha_pool_improvements")
args = parser.parse_args()

k = args.k
os.makedirs(args.out_dir, exist_ok=True)


def parse_schedule(s, bins):
    phases = []
    for entry in s.split(","):
        b, n = entry.strip().split(":")
        b, n = int(b), int(n)
        assert 0 <= b < len(bins)
        phases.append({"bin": b, "n_queries": n, "is_plateau": False})
    seen = {}
    first_seen = set()
    for p in phases:
        seen[p["bin"]] = seen.get(p["bin"], 0) + 1
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
    return queries, gt, scores


def load_index(n_elements, dim):
    path = os.path.join(args.data_dir, "hnsw_index.bin")
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.load_index(path, max_elements=n_elements)
    return idx


def make_controller(idx, warmup_queries, eviction_policy, pool_top_k, ef_escalation_mode):
    return HardnessAdaptiveController(
        idx,
        reference_queries=warmup_queries,
        k=k,
        hard_percentile=args.hard_percentile,
        warmup_ef=args.warmup_ef,
        alpha=args.alpha,
        max_layer=args.max_layer,
        hard_rewire_cooldown=999999,  # rewiring disabled for this ablation
        max_pool_size=args.hard_max_pool_size,
        pool_seed_ef=args.pool_seed_ef,
        pool_top_k=pool_top_k,
        eviction_policy=eviction_policy,
        escalation_window=args.escalation_window,
        escalation_trigger=args.escalation_trigger,
        escalation_factor=args.escalation_factor,
        ef_escalation_mode=ef_escalation_mode,
        use_pool=True,
        use_rewire=False,
        use_ef_escalation=True,
    )


STRATEGIES = [
    # (name, eviction_policy, pool_top_k, ef_escalation_mode)
    ("ha_original",    "distance",   1,               "binary"),
    ("ha_redundancy",  "redundancy", 1,               "binary"),
    ("ha_multi_ep",    "distance",   None,            "binary"),   # None -> use args.pool_top_k
    ("ha_continuous",  "distance",   1,               "continuous"),
    ("ha_all",         "redundancy", None,            "continuous"),
]


def save_summary(all_results, bin_edges, phase_meta):
    rows = []
    for (strategy, ef, phase_idx), results in sorted(all_results.items()):
        def avg(f): return float(np.nanmean([r[f] for r in results]))
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
            "mean_base_dist_comps":   avg("base_dist_comps"),
            "mean_t_query_ms":        avg("t_query_ms"),
            "mean_t_pool_scan_ms":    avg("t_pool_scan_ms"),
            "mean_t_pool_knn_ms":     avg("t_pool_knn_ms"),
            "mean_t_adapt_ms":        avg("t_adapt_ms"),
            "n_queries":              len(results),
        })
    df = pd.DataFrame(rows).round(4)
    path = os.path.join(args.out_dir, "summary.csv")
    df.to_csv(path, index=False)
    print(f"  saved {path}")

    for ef in sorted(df["ef_search"].unique()):
        sub = df[df["ef_search"] == ef]
        pivot = sub.pivot(index="phase_idx", columns="strategy", values="mean_recall").round(3)
        print(f"\nrecall@{k}  ef={ef}:")
        print(pivot.to_string())

    base = df[df["strategy"] == "no_adaptation"].set_index(["phase_idx", "ef_search"])["mean_recall"]
    for strategy in sorted(df["strategy"].unique()):
        if strategy == "no_adaptation":
            continue
        adap = df[df["strategy"] == strategy].set_index(["phase_idx", "ef_search"])["mean_recall"]
        delta = (adap - base).dropna().round(3).unstack("ef_search")
        print(f"\nrecall delta ({strategy} - no_adaptation):")
        print(delta.to_string())


def save_per_query(all_results, phase_meta):
    rows = []
    for (strategy, ef, phase_idx), results in sorted(all_results.items()):
        actual_bin = phase_meta[phase_idx]["bin"]
        for i, r in enumerate(results):
            rows.append({
                "strategy":        strategy,
                "ef_search":       ef,
                "phase_idx":       phase_idx,
                "hardness_bin":    actual_bin,
                "is_plateau":      phase_meta[phase_idx]["is_plateau"],
                "query_id":        i,
                "recall":          r["recall"],
                "base_dist_comps": r["base_dist_comps"],
                "t_query_ms":      r.get("t_query_ms", float("nan")),
                "t_pool_scan_ms":  r.get("t_pool_scan_ms", float("nan")),
                "t_pool_knn_ms":   r.get("t_pool_knn_ms", float("nan")),
                "t_adapt_ms":      r.get("t_adapt_ms", float("nan")),
            })
    path = os.path.join(args.out_dir, "per_query.csv")
    pd.DataFrame(rows).round(6).to_csv(path, index=False)
    print(f"  saved {path}")


def main():
    print("HardnessAdaptive pool-improvement ablation")
    print(f"ef_sweep={args.ef_sweep}  k={k}  pool_top_k_arg={args.pool_top_k}\n")

    queries, gt, scores = load_dataset()
    dim = queries.shape[1]

    bins = np.array_split(np.arange(len(queries)), args.n_bins)
    bin_edges = np.array(
        [scores[b[0]] for b in bins] + [scores[bins[-1][-1]]], dtype=np.float32
    )
    print(f"hardness bin edges: {bin_edges.round(3)}")

    bins = [b[:args.n_queries_per_bin] for b in bins]
    schedule = parse_schedule(args.drift_schedule, bins)
    warmup_queries = queries[bins[schedule[0]["bin"]]]
    eval_phases = schedule[1:]
    phase_meta = {i: p for i, p in enumerate(eval_phases)}
    phase_samples = [
        rng.choice(bins[p["bin"]], size=p["n_queries"], replace=p["n_queries"] > len(bins[p["bin"]]))
        for p in eval_phases
    ]

    params_path = os.path.join(args.data_dir, "params.json")
    if os.path.exists(params_path):
        with open(params_path) as f:
            n_index = json.load(f)["n_index"]
    else:
        n_index = int(input("n_index not found in params.json, enter manually: "))

    all_results = {}

    # baseline
    print(f"\n{'='*60}\nstrategy: no_adaptation\n{'='*60}")
    for ef in args.ef_sweep:
        idx = load_index(n_index, dim)
        print(f"  ef={ef}")
        for phase_idx, phase in enumerate(eval_phases):
            results = run_query_batch_hardness(idx, queries[phase_samples[phase_idx]],
                                               gt[phase_samples[phase_idx]], k, ef)
            all_results[("no_adaptation", ef, phase_idx)] = results
            print(f"    phase={phase_idx}  bin={phase['bin']}  "
                  f"recall={np.mean([r['recall'] for r in results]):.4f}")

    # controller variants
    for name, eviction, top_k_arg, ef_mode in STRATEGIES:
        top_k = args.pool_top_k if top_k_arg is None else top_k_arg
        print(f"\n{'='*60}\nstrategy: {name}  (eviction={eviction}  pool_top_k={top_k}  ef_mode={ef_mode})\n{'='*60}")
        for ef in args.ef_sweep:
            idx  = load_index(n_index, dim)
            ctrl = make_controller(idx, warmup_queries, eviction, top_k, ef_mode)
            print(f"  ef={ef}")
            for phase_idx, phase in enumerate(eval_phases):
                results = run_query_batch_hardness(
                    idx, queries[phase_samples[phase_idx]],
                    gt[phase_samples[phase_idx]], k, ef, controller=ctrl
                )
                all_results[(name, ef, phase_idx)] = results
                print(f"    phase={phase_idx}  bin={phase['bin']}  "
                      f"recall={np.mean([r['recall'] for r in results]):.4f}  "
                      f"pool={len(ctrl._pool)}  escalations={ctrl.escalation_count}")

    print("\nsaving results...")
    save_summary(all_results, bin_edges, phase_meta)
    save_per_query(all_results, phase_meta)
    print(f"\ndone — {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
