"""Build clustered structural hardness drift dataset for exp52.

Applies the exp37 three-way filter then restricts the hard pool to the single
KMeans cluster with the highest mean joint hardness score, creating a drift
scenario where hard queries are both EH-high and spatially concentrated.
"""

import os
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.drift.cluster_drift import save_cluster_drift_dataset
from src.drift.hardness_drift import build_hardness_drift_dataset, build_hardness_drift_schedule


def select_hard_pool_filtered(scores, recall_high_threshold):
    """Three-way filter returning all passing indices without size cap."""
    eh_p75 = np.percentile(scores[:, 0], 75)
    used_threshold = recall_high_threshold
    for threshold in [recall_high_threshold, 0.7, 0.6]:
        mask = (
            (scores[:, 0] > eh_p75)
            & (scores[:, 1] < 0.5)
            & (scores[:, 2] > threshold)
        )
        if mask.sum() >= 5000:
            used_threshold = threshold
            if threshold < recall_high_threshold:
                print(f"  relaxed recall_high threshold to {threshold:.1f} "
                      f"(original {recall_high_threshold:.1f} yielded <5000 queries)")
            break
    else:
        raise RuntimeError("Hard pool filter yielded <5000 queries even at threshold=0.6")
    print(f"  recall_high threshold used: {used_threshold:.1f}")
    return np.where(mask)[0]


def find_best_cluster(hard_indices, queries_all, scores,
                      n_cluster_candidates, min_cluster_size, cluster_seed):
    """Cluster filtered hard queries; return indices and stats for the highest-joint-score cluster."""
    hard_queries = np.array(queries_all[hard_indices], dtype=np.float32)
    joint_all = (
        scores[hard_indices, 0]
        * (1.0 - scores[hard_indices, 1])
        * scores[hard_indices, 2]
    )

    print(f"  Running MiniBatchKMeans(n_clusters={n_cluster_candidates}, n_init=3) "
          f"on {len(hard_indices):,} queries...")
    km = MiniBatchKMeans(n_clusters=n_cluster_candidates, random_state=cluster_seed, n_init=3)
    cluster_labels = km.fit_predict(hard_queries)
    centroids = km.cluster_centers_

    cluster_stats = []
    for c in range(n_cluster_candidates):
        mask = cluster_labels == c
        n_members = int(mask.sum())
        if n_members < min_cluster_size:
            continue
        spread = float(np.linalg.norm(hard_queries[mask] - centroids[c], axis=1).mean())
        mem_global = hard_indices[mask]
        mean_eh = float(scores[mem_global, 0].mean())
        mean_recall32 = float(scores[mem_global, 1].mean())
        mean_recall256 = float(scores[mem_global, 2].mean())
        mean_joint = float(joint_all[mask].mean())
        cluster_stats.append({
            "cluster_id": c,
            "n_members": n_members,
            "spread": spread,
            "mean_eh": mean_eh,
            "mean_recall32": mean_recall32,
            "mean_recall256": mean_recall256,
            "mean_joint": mean_joint,
        })

    if not cluster_stats:
        raise RuntimeError(
            f"No cluster met min_cluster_size={min_cluster_size}; "
            "lower min_cluster_size in config.yaml"
        )

    cluster_stats.sort(key=lambda x: x["mean_joint"], reverse=True)

    print(f"\n  Top-5 candidate clusters by joint score (n >= {min_cluster_size}):")
    print(f"  {'cluster_id':>10}  {'n':>6}  {'spread':>8}  "
          f"{'mean_eh':>8}  {'recall32':>9}  {'recall256':>10}  {'joint':>8}")
    for s in cluster_stats[:5]:
        print(f"  {s['cluster_id']:>10}  {s['n_members']:>6}  {s['spread']:>8.3f}  "
              f"{s['mean_eh']:>8.3f}  {s['mean_recall32']:>9.3f}  "
              f"{s['mean_recall256']:>10.3f}  {s['mean_joint']:>8.4f}")

    best = cluster_stats[0]
    print(f"\n  Selected cluster {best['cluster_id']}: "
          f"n={best['n_members']}, spread={best['spread']:.3f}, "
          f"mean_eh={best['mean_eh']:.3f}, mean_recall32={best['mean_recall32']:.3f}, "
          f"mean_recall256={best['mean_recall256']:.3f}, "
          f"mean_joint={best['mean_joint']:.4f}")

    best_mask = cluster_labels == best["cluster_id"]
    return hard_indices[best_mask], joint_all[best_mask], best


def select_easy_pool(scores, easy_pool_size):
    """Low EH + high recall@32 filter for easy queries."""
    eh_p25 = np.percentile(scores[:, 0], 25)
    mask = (scores[:, 1] > 0.95) & (scores[:, 0] < eh_p25)
    easy_indices = np.where(mask)[0]
    if len(easy_indices) > easy_pool_size:
        top = np.argsort(scores[easy_indices, 1])[::-1][:easy_pool_size]
        easy_indices = easy_indices[top]
    return easy_indices


def main():
    config_path = sys.argv[1]
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    out_gradual = str(ROOT / cfg["dataset_gradual"])
    out_sudden = str(ROOT / cfg["dataset_sudden"])

    if (os.path.exists(os.path.join(out_gradual, "config.json"))
            and os.path.exists(os.path.join(out_sudden, "config.json"))):
        print("datasets already exist, skipping")
        return

    data_dir = cfg["data_dir"]
    scores_path = cfg["scores_path"]
    n_epochs = cfg["n_epochs"]
    epoch_size = cfg["epoch_size"]
    easy_pool_size = cfg["easy_pool_size"]
    hard_pool_size = cfg["hard_pool_size"]
    recall_high_threshold = cfg["recall_high_threshold"]
    n_cluster_candidates = cfg["n_cluster_candidates"]
    min_cluster_size = cfg["min_cluster_size"]
    cluster_seed = cfg["cluster_seed"]
    seed = cfg["seed"]

    print(f"Loading structural scores from {scores_path}...")
    scores = np.load(scores_path)
    print(f"  scores: {scores.shape}  "
          f"eh=[{scores[:, 0].min():.3f},{scores[:, 0].max():.3f}]  "
          f"recall32=[{scores[:, 1].min():.3f},{scores[:, 1].max():.3f}]  "
          f"recall256=[{scores[:, 2].min():.3f},{scores[:, 2].max():.3f}]")

    # Step A — three-way structural filter (same as exp37)
    print(f"\nStep A — three-way structural filter (same as exp37)...")
    hard_indices_all = select_hard_pool_filtered(scores, recall_high_threshold)
    print(f"  filtered hard candidates: N={len(hard_indices_all):,}")

    # Step B — cluster filtered hard pool, select the highest-joint-score cluster
    print(f"\nStep B — clustering filtered hard pool into {n_cluster_candidates} clusters...")
    queries_all = np.load(os.path.join(data_dir, "queries_by_hardness.npy"), mmap_mode='r')
    cluster_indices, cluster_joint_scores, cluster_info = find_best_cluster(
        hard_indices_all, queries_all, scores,
        n_cluster_candidates, min_cluster_size, cluster_seed,
    )

    # Step C — select hard and easy pools
    print(f"\nStep C — selecting hard and easy pools...")
    if len(cluster_indices) > hard_pool_size:
        top = np.argsort(cluster_joint_scores)[::-1][:hard_pool_size]
        hard_idx = cluster_indices[top]
    else:
        hard_idx = cluster_indices
    print(f"  hard pool: N={len(hard_idx):,} (cluster {cluster_info['cluster_id']})")

    easy_idx = select_easy_pool(scores, easy_pool_size)
    print(f"  easy pool: N={len(easy_idx):,}")

    gt_all = np.load(os.path.join(data_dir, "ground_truth.npy"), mmap_mode='r')
    hard_queries = np.array(queries_all[hard_idx], dtype=np.float32)
    hard_gt = np.array(gt_all[hard_idx])
    easy_queries = np.array(queries_all[easy_idx], dtype=np.float32)
    easy_gt = np.array(gt_all[easy_idx])

    n_easy_actual = len(easy_queries)
    n_hard_actual = len(hard_queries)

    print(f"\nValidation summary:")
    print(f"  Hard pool:  N={n_hard_actual:,}  "
          f"mean_eh={scores[hard_idx, 0].mean():.3f}  "
          f"mean_recall32={scores[hard_idx, 1].mean():.3f}  "
          f"mean_recall256={scores[hard_idx, 2].mean():.3f}")
    print(f"  Easy pool:  N={n_easy_actual:,}  "
          f"mean_eh={scores[easy_idx, 0].mean():.3f}  "
          f"mean_recall32={scores[easy_idx, 1].mean():.3f}")

    # easy first, hard last — matches build_hardness_drift_dataset API
    queries_filtered = np.vstack([easy_queries, hard_queries])
    gt_filtered = np.vstack([easy_gt, hard_gt])

    mean_easy_recall32 = float(scores[easy_idx, 1].mean())
    mean_hard_recall32 = float(scores[hard_idx, 1].mean())

    # Step D — build and save datasets
    for schedule_name in ["gradual", "sudden"]:
        schedule_t = build_hardness_drift_schedule(n_epochs, schedule_name)

        print(f"\nStep D — building {schedule_name} dataset ({n_epochs} epochs)...")
        dataset = build_hardness_drift_dataset(
            queries_filtered, gt_filtered, schedule_t,
            epoch_size, n_easy_actual, n_hard_actual, seed,
        )
        print(f"  {len(dataset['epochs'])} epochs, each shape {dataset['epochs'][0].shape}")

        print(f"  {'epoch':>6}  {'t':>6}  {'mean_recall32':>14}  {'hard_frac':>10}")
        diagnostics = []
        for epoch_idx, t in enumerate(schedule_t):
            n_hard_ep = round(epoch_size * t)
            n_easy_ep = epoch_size - n_hard_ep
            hard_frac = n_hard_ep / epoch_size
            approx_mean = (
                n_easy_ep * mean_easy_recall32 + n_hard_ep * mean_hard_recall32
            ) / epoch_size
            print(f"  {epoch_idx:>6}  {t:>6.4f}  {approx_mean:>14.4f}  {hard_frac:>10.4f}")
            diagnostics.append({
                "epoch_idx": epoch_idx,
                "t_value": float(t),
                "approx_mean_recall32": float(approx_mean),
                "hard_fraction": float(hard_frac),
            })

        out_path = out_gradual if schedule_name == "gradual" else out_sudden

        save_config = {
            "schedule_name": schedule_name,
            "n_epochs": n_epochs,
            "epoch_size": epoch_size,
            "easy_pool_size": n_easy_actual,
            "hard_pool_size": n_hard_actual,
            "seed": seed,
            "schedule": [float(t) for t in schedule_t],
            "selection_criterion": "clustered_structural",
            "cluster_id": cluster_info["cluster_id"],
            "n_cluster_candidates": n_cluster_candidates,
            "min_cluster_size": min_cluster_size,
            "cluster_seed": cluster_seed,
            "cluster_n_members": cluster_info["n_members"],
            "cluster_spread": cluster_info["spread"],
            "cluster_mean_eh": cluster_info["mean_eh"],
            "cluster_mean_recall32": cluster_info["mean_recall32"],
            "cluster_mean_recall256": cluster_info["mean_recall256"],
            "cluster_mean_joint": cluster_info["mean_joint"],
            "eh_high_percentile": cfg["eh_high_percentile"],
            "recall_low_threshold": cfg["recall_low_threshold"],
            "recall_high_threshold": cfg["recall_high_threshold"],
        }

        print(f"\nSaving {schedule_name} dataset to {out_path}...")
        save_cluster_drift_dataset(
            out_path, dataset["base"], dataset["epochs"],
            dataset["groundtruth"], save_config, diagnostics,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
