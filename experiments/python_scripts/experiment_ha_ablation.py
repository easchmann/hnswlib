# Ablation study for HardnessAdaptiveController on SIFT hardness-based drift.
#
# Runs the same plateau drift schedule as experiment_hardness_adaptive_sift_with_plateaus.py
# but instead of comparing HA vs PAR vs baseline, it isolates the contribution of each
# of HA's three mechanisms:
#
#   pool        — hard-query pool entry points
#   rewire      — difficulty-triggered local neighbourhood rewiring
#   escalation  — adaptive ef boost on hard phases
#
# Variants tested (all 2^3 mechanism combinations + two critical baselines):
#
#   no_adaptation      — vanilla HNSW, no changes
#   static_ef_escalated — vanilla HNSW at ef * escalation_factor, no adaptation logic
#                         this is the key comparison: does escalation add anything beyond
#                         just running at a higher ef from the start?
#   pool_only
#   rewire_only
#   escalation_only
#   pool_rewire
#   pool_escalation
#   rewire_escalation
#   full               — all three mechanisms (same as hardness_adaptive in the main experiment)
#
# Usage:
#   python experiment_ha_ablation.py \
#       --data_dir data/sift_hardness \
#       --drift_schedule "0:3000,1:3000,2:3000,3:3000,3:5000,4:3000,5:3000,6:3000,6:5000,7:3000,8:3000,9:3000,9:5000" \
#       --ef_sweep 10 50 100 \
#       --out_dir results_ha_ablation

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
parser.add_argument("--drift_schedule",       required=True)
parser.add_argument("--n_bins",               type=int,   default=10)
parser.add_argument("--n_queries_per_bin",    type=int,   default=5000)
parser.add_argument("--ef_sweep",             type=int,   nargs="+", default=[10, 50, 100])
parser.add_argument("--k",                    type=int,   default=10)
parser.add_argument("--alpha",                type=float, default=1.1)
parser.add_argument("--max_layer",            type=int,   default=3)
parser.add_argument("--hard_percentile",      type=float, default=75)
parser.add_argument("--warmup_ef",            type=int,   default=50)
parser.add_argument("--sliding_window",       type=int,   default=200)
parser.add_argument("--hard_rewire_cooldown", type=int,   default=10)
parser.add_argument("--rewire_k_nodes",       type=int,   default=5)
parser.add_argument("--max_pool_size",        type=int,   default=50)
parser.add_argument("--pool_seed_ef",         type=int,   default=20)
parser.add_argument("--escalation_window",    type=int,   default=50)
parser.add_argument("--escalation_trigger",   type=float, default=0.3)
parser.add_argument("--escalation_factor",    type=int,   default=3)
parser.add_argument("--out_dir",              default="results_ha_ablation")
args = parser.parse_args()

k = args.k
os.makedirs(args.out_dir, exist_ok=True)

# (pool, rewire, escalation)
ABLATION_VARIANTS = [
    ("pool_only",          True,  False, False),
    ("rewire_only",        False, True,  False),
    ("escalation_only",    False, False, True),
    ("pool_rewire",        True,  True,  False),
    ("pool_escalation",    True,  False, True),
    ("rewire_escalation",  False, True,  True),
    ("full",               True,  True,  True),
]


def parse_schedule(s, bins):
    phases = []
    for entry in s.split(","):
        b, n = entry.strip().split(":")
        phases.append({"bin": int(b), "n_queries": int(n), "is_plateau": False})
    first_seen = set()
    for p in phases:
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
    print(f"  {len(queries):,} queries  dim={queries.shape[1]}")
    return queries, gt, scores


def load_index(n_elements, dim):
    path = os.path.join(args.data_dir, "hnsw_index.bin")
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.load_index(path, max_elements=n_elements)
    return idx


def save_params(n_index, n_queries_total, dim, bin_edges):
    rows = [
        {"param": "dataset",              "value": "SIFT1B-hardness"},
        {"param": "experiment_type",      "value": "ha_ablation"},
        {"param": "dim",                  "value": dim},
        {"param": "n_index",              "value": n_index},
        {"param": "n_queries_total",      "value": n_queries_total},
        {"param": "n_bins",               "value": args.n_bins},
        {"param": "hardness_bin_edges",   "value": str(list(bin_edges.round(4)))},
        {"param": "ef_sweep",             "value": str(args.ef_sweep)},
        {"param": "k",                    "value": k},
        {"param": "n_queries_per_bin",    "value": args.n_queries_per_bin},
        {"param": "alpha",                "value": args.alpha},
        {"param": "max_layer",            "value": args.max_layer},
        {"param": "hard_percentile",      "value": args.hard_percentile},
        {"param": "warmup_ef",            "value": args.warmup_ef},
        {"param": "sliding_window",       "value": args.sliding_window},
        {"param": "hard_rewire_cooldown", "value": args.hard_rewire_cooldown},
        {"param": "rewire_k_nodes",       "value": args.rewire_k_nodes},
        {"param": "max_pool_size",        "value": args.max_pool_size},
        {"param": "pool_seed_ef",         "value": args.pool_seed_ef},
        {"param": "escalation_window",    "value": args.escalation_window},
        {"param": "escalation_trigger",   "value": args.escalation_trigger},
        {"param": "escalation_factor",    "value": args.escalation_factor},
        {"param": "drift_schedule",       "value": args.drift_schedule},
        {"param": "rng_seed",             "value": 42},
    ]
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "params.csv"), index=False)


def save_summary(all_results, bin_edges, phase_meta):
    rows = []
    for (strategy, ef, phase_idx), results in sorted(all_results.items()):
        def avg(f): return float(np.nanmean([r[f] for r in results]))
        actual_bin = phase_meta[phase_idx]["bin"]
        # static_ef_escalated runs at ef * escalation_factor but is keyed by base ef
        # so comparisons against no_adaptation and escalation_only are at the same ef_search row
        actual_ef = ef * args.escalation_factor if strategy == "static_ef_escalated" else ef
        rows.append({
            "strategy":               strategy,
            "ef_search":              ef,
            "actual_search_ef":       actual_ef,
            "phase_idx":              phase_idx,
            "hardness_bin":           actual_bin,
            "is_plateau":             phase_meta[phase_idx]["is_plateau"],
            "hardness_bin_lo":        float(bin_edges[actual_bin]),
            "hardness_bin_hi":        float(bin_edges[actual_bin + 1]),
            "mean_recall":            avg("recall"),
            "mean_base_dist_comps":   avg("base_dist_comps"),
            "n_queries":              len(results),
            "mean_t_query_ms":        avg("t_query_ms"),
            "mean_t_adapt_ms":        avg("t_adapt_ms"),
        })
    df = pd.DataFrame(rows).round(4)
    df.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)

    # print recall table per strategy at each ef
    baseline = df[df["strategy"] == "no_adaptation"].set_index(["phase_idx", "ef_search"])["mean_recall"]
    for strategy in df["strategy"].unique():
        sub = df[df["strategy"] == strategy]
        pivot = sub.pivot(index="phase_idx", columns="ef_search", values="mean_recall").round(3)
        print(f"\n{strategy} recall@{k}:")
        print(pivot.to_string())
        if strategy not in ("no_adaptation", "static_ef_escalated"):
            delta = (df[df["strategy"] == strategy].set_index(["phase_idx", "ef_search"])["mean_recall"] - baseline)
            print(f"  delta vs no_adaptation: {delta.mean():+.4f} mean  {delta.min():+.4f} min  {delta.max():+.4f} max")


def save_adapt_log(adapt_log):
    if not adapt_log:
        return
    pd.DataFrame(adapt_log).to_csv(os.path.join(args.out_dir, "adapt_log.csv"), index=False)
    print(f"  saved adapt_log.csv ({len(adapt_log)} entries)")


def make_controller(idx, warmup_queries, use_pool, use_rewire, use_escalation):
    return HardnessAdaptiveController(
        idx,
        reference_queries=warmup_queries,
        k=k,
        hard_percentile=args.hard_percentile,
        warmup_ef=args.warmup_ef,
        sliding_window=args.sliding_window,
        alpha=args.alpha,
        max_layer=args.max_layer,
        hard_rewire_cooldown=args.hard_rewire_cooldown,
        rewire_k_nodes=args.rewire_k_nodes,
        max_pool_size=args.max_pool_size,
        pool_seed_ef=args.pool_seed_ef,
        escalation_window=args.escalation_window,
        escalation_trigger=args.escalation_trigger,
        escalation_factor=args.escalation_factor,
        use_pool=use_pool,
        use_rewire=use_rewire,
        use_ef_escalation=use_escalation,
    )


def run_variant(name, n_index, warmup_queries, phase_samples, eval_phases, queries, gt, all_results, adapt_log):
    print(f"\n{'='*60}")
    print(f"variant: {name}")
    print(f"{'='*60}")

    for ef in args.ef_sweep:
        idx_fresh = load_index(n_index, queries.shape[1])

        if name == "no_adaptation":
            ctrl = None
            search_ef = ef
        elif name == "static_ef_escalated":
            ctrl = None
            search_ef = ef * args.escalation_factor
        else:
            use_pool, use_rewire, use_esc = next(
                (p, r, e) for (n, p, r, e) in ABLATION_VARIANTS if n == name
            )
            ctrl = make_controller(idx_fresh, warmup_queries, use_pool, use_rewire, use_esc)
            search_ef = ef

        print(f"\n  ef={ef}" + (f"  (search_ef={search_ef})" if search_ef != ef else ""))

        for phase_idx, phase in enumerate(eval_phases):
            bin_idx = phase_samples[phase_idx]
            results = run_query_batch_hardness(idx_fresh, queries[bin_idx], gt[bin_idx], k, search_ef, controller=ctrl)
            all_results[(name, ef, phase_idx)] = results
            mean_r = np.mean([r["recall"] for r in results])
            print(f"    phase={phase_idx}  bin={phase['bin']}  plateau={phase['is_plateau']}  "
                  f"recall={mean_r:.4f}")

        if ctrl is not None:
            for entry in ctrl.update_log:
                entry["ef"] = ef
                entry["strategy"] = name
                adapt_log.append(entry)


def main():
    print("HardnessAdaptive ablation study")
    print(f"ef_sweep={args.ef_sweep}  k={k}  rewire_k_nodes={args.rewire_k_nodes}")

    queries, gt, scores = load_dataset()
    dim = queries.shape[1]

    bins = np.array_split(np.arange(len(queries)), args.n_bins)
    bin_edges = np.array(
        [scores[b[0]] for b in bins] + [scores[bins[-1][-1]]], dtype=np.float32
    )
    bins = [b[:args.n_queries_per_bin] for b in bins]

    schedule = parse_schedule(args.drift_schedule, bins)
    warmup_queries = queries[bins[schedule[0]["bin"]]]
    eval_phases = schedule[1:]
    phase_meta = {i: p for i, p in enumerate(eval_phases)}

    phase_samples = [
        rng.choice(bins[p["bin"]], size=p["n_queries"],
                   replace=p["n_queries"] > len(bins[p["bin"]]))
        for p in eval_phases
    ]

    params_path = os.path.join(args.data_dir, "params.json")
    with open(params_path) as f:
        n_index = json.load(f)["n_index"]

    all_results = {}
    adapt_log = []

    all_variants = ["no_adaptation", "static_ef_escalated"] + [name for name, *_ in ABLATION_VARIANTS]
    for name in all_variants:
        run_variant(name, n_index, warmup_queries, phase_samples, eval_phases,
                    queries, gt, all_results, adapt_log)

    print("\nsaving results...")
    save_params(n_index, len(queries), dim, bin_edges)
    save_summary(all_results, bin_edges, phase_meta)
    save_adapt_log(adapt_log)
    print(f"\ndone — {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
