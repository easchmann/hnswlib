"""Target-coherence analysis for Exp52 (clustered hard pool)
"""

import itertools
import json
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).parent
PLATEAU_EPOCHS = [20, 21, 22, 23, 24]


def load_plateau_data(dataset_dir):
    """Load and return (queries, groundtruth) per plateau epoch."""
    queries, gts = [], []
    for ep in PLATEAU_EPOCHS:
        queries.append(np.load(dataset_dir / f"epoch_{ep:03d}.npy"))
        gts.append(np.load(dataset_dir / f"groundtruth_{ep:03d}.npy"))
    return queries, gts


def dedupe(queries, gt):
    """Keep one (query, gt) row per distinct query vector."""
    _, first_idx = np.unique(queries, axis=0, return_index=True)
    first_idx = np.sort(first_idx)
    return queries[first_idx], gt[first_idx]


def union_coverage_ratio(gt):
    n_queries, k = gt.shape
    distinct = len(np.unique(gt))
    return distinct, n_queries * k, distinct / (n_queries * k)


def mean_pairwise_jaccard(gt, max_pairs=200_000, seed=0):
    """Mean Jaccard overlap of NN sets, over all pairs (or a random sample if
    the number of distinct queries makes exhaustive pairing too large)."""
    n = gt.shape[0]
    sets = [set(row.tolist()) for row in gt]
    all_pairs = list(itertools.combinations(range(n), 2))
    if len(all_pairs) > max_pairs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(all_pairs), size=max_pairs, replace=False)
        pairs = [all_pairs[i] for i in idx]
    else:
        pairs = all_pairs
    total = 0.0
    for i, j in pairs:
        inter = len(sets[i] & sets[j])
        union = len(sets[i] | sets[j])
        total += inter / union if union else 0.0
    return total / len(pairs) if pairs else 0.0


def analyze_schedule(schedule_name):
    dataset_dir = EXP_DIR / f"dataset_{schedule_name}"
    queries_ep, gt_ep = load_plateau_data(dataset_dir)

    per_epoch = []
    for ep, q, gt in zip(PLATEAU_EPOCHS, queries_ep, gt_ep):
        q_dedup, gt_dedup = dedupe(q, gt)
        distinct, total_slots, ratio = union_coverage_ratio(gt_dedup)
        jaccard = mean_pairwise_jaccard(gt_dedup)
        per_epoch.append({
            "epoch": ep,
            "n_queries_raw": int(q.shape[0]),
            "n_queries_dedup": int(q_dedup.shape[0]),
            "k": int(gt_dedup.shape[1]),
            "distinct_nn_count": distinct,
            "total_nn_slots": total_slots,
            "union_coverage_ratio": ratio,
            "mean_pairwise_jaccard": jaccard,
        })
        print(f"  [{schedule_name}] epoch {ep}: {q_dedup.shape[0]}/{q.shape[0]} unique queries; "
              f"distinct_nn={distinct}/{total_slots} (ratio={ratio:.4f})  "
              f"mean_pairwise_jaccard={jaccard:.4f}")

    # Aggregate across all 5 plateau epochs, deduplicated globally
    q_all = np.concatenate(queries_ep, axis=0)
    gt_all = np.concatenate(gt_ep, axis=0)
    q_all_dedup, gt_all_dedup = dedupe(q_all, gt_all)
    distinct_all, total_slots_all, ratio_all = union_coverage_ratio(gt_all_dedup)
    jaccard_all = mean_pairwise_jaccard(gt_all_dedup)
    print(f"  [{schedule_name}] AGGREGATE: {q_all_dedup.shape[0]}/{q_all.shape[0]} unique queries "
          f"across 5 epochs; distinct_nn={distinct_all}/{total_slots_all} (ratio={ratio_all:.4f})  "
          f"mean_pairwise_jaccard={jaccard_all:.4f}")

    return {
        "schedule": schedule_name,
        "per_epoch": per_epoch,
        "aggregate_n_queries_raw": int(q_all.shape[0]),
        "aggregate_n_queries_dedup": int(q_all_dedup.shape[0]),
        "aggregate_distinct_nn_count": distinct_all,
        "aggregate_total_nn_slots": total_slots_all,
        "aggregate_union_coverage_ratio": ratio_all,
        "aggregate_mean_pairwise_jaccard": jaccard_all,
    }


def main():
    results = {}
    for schedule_name in ["gradual", "sudden"]:
        print(f"\n=== Exp52 clustered pool, {schedule_name} schedule, plateau epochs {PLATEAU_EPOCHS} ===")
        results[schedule_name] = analyze_schedule(schedule_name)

    out_path = EXP_DIR / "target_coherence_clustered.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
