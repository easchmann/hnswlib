"""Build structural hardness drift dataset for exp37.

Filters 5M queries by three-way structural criterion (high EH + low recall@32 +
high recall@256), then constructs gradual and sudden epoch datasets.
"""

import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.drift.cluster_drift import save_cluster_drift_dataset
from src.drift.hardness_drift import build_hardness_drift_dataset, build_hardness_drift_schedule


def select_hard_pool(scores, recall_high_threshold, hard_pool_size):
    """Three-way filter for navigation-failure queries; relaxes third threshold if needed."""
    eh_p75 = np.percentile(scores[:, 0], 75)
    for threshold in [recall_high_threshold, 0.7, 0.6]:
        mask = (
            (scores[:, 0] > eh_p75)
            & (scores[:, 1] < 0.5)
            & (scores[:, 2] > threshold)
        )
        if mask.sum() >= 5000:
            if threshold < recall_high_threshold:
                print(f"  relaxed recall_high threshold to {threshold:.1f} "
                      f"(original {recall_high_threshold:.1f} yielded <5000 queries)")
            else:
                print(f"  recall_high threshold: {threshold:.1f}")
            break
    else:
        raise RuntimeError("Hard pool filter yielded <5000 queries even at threshold=0.6")

    hard_indices = np.where(mask)[0]
    if len(hard_indices) > hard_pool_size:
        # Prefer queries with highest joint hardness score: eh * (1 - recall32) * recall256
        joint = (
            scores[hard_indices, 0]
            * (1.0 - scores[hard_indices, 1])
            * scores[hard_indices, 2]
        )
        top = np.argsort(joint)[::-1][:hard_pool_size]
        hard_indices = hard_indices[top]

    return hard_indices


def select_easy_pool(scores, easy_pool_size):
    """Low EH + high recall@32 filter for easy queries."""
    eh_p25 = np.percentile(scores[:, 0], 25)
    mask = (scores[:, 1] > 0.95) & (scores[:, 0] < eh_p25)
    easy_indices = np.where(mask)[0]
    if len(easy_indices) > easy_pool_size:
        # Prefer highest recall@32
        top = np.argsort(scores[easy_indices, 1])[::-1][:easy_pool_size]
        easy_indices = easy_indices[top]
    return easy_indices


def main():
    config_path = sys.argv[1]
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg["data_dir"]
    scores_path = cfg["scores_path"]
    n_epochs = cfg["n_epochs"]
    epoch_size = cfg["epoch_size"]
    easy_pool_size = cfg["easy_pool_size"]
    hard_pool_size = cfg["hard_pool_size"]
    recall_high_threshold = cfg["recall_high_threshold"]
    seed = cfg["seed"]

    print(f"Loading queries and ground truth from {data_dir}...")
    queries_all = np.load(os.path.join(data_dir, "queries_by_hardness.npy"), mmap_mode='r')
    gt_all = np.load(os.path.join(data_dir, "ground_truth.npy"), mmap_mode='r')
    print(f"  queries: {queries_all.shape}  gt: {gt_all.shape}")

    print(f"Loading structural scores from {scores_path}...")
    scores = np.load(scores_path)
    print(f"  scores: {scores.shape}  "
          f"eh=[{scores[:, 0].min():.3f},{scores[:, 0].max():.3f}]  "
          f"recall32=[{scores[:, 1].min():.3f},{scores[:, 1].max():.3f}]  "
          f"recall256=[{scores[:, 2].min():.3f},{scores[:, 2].max():.3f}]")

    print(f"\nSelecting hard pool (target size: {hard_pool_size:,})...")
    hard_idx = select_hard_pool(scores, recall_high_threshold, hard_pool_size)
    print(f"  hard pool: N={len(hard_idx):,}")

    print(f"Selecting easy pool (target size: {easy_pool_size:,})...")
    easy_idx = select_easy_pool(scores, easy_pool_size)
    print(f"  easy pool: N={len(easy_idx):,}")

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

    # Build synthetic sorted array: easy first, hard last — matches build_hardness_drift_dataset API
    queries_filtered = np.vstack([easy_queries, hard_queries])
    gt_filtered = np.vstack([easy_gt, hard_gt])

    mean_easy_recall32 = float(scores[easy_idx, 1].mean())
    mean_hard_recall32 = float(scores[hard_idx, 1].mean())

    for schedule_name in ["gradual", "sudden"]:
        schedule_t = build_hardness_drift_schedule(n_epochs, schedule_name)

        print(f"\nBuilding {schedule_name} dataset ({n_epochs} epochs)...")
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
            approx_mean = (n_easy_ep * mean_easy_recall32 + n_hard_ep * mean_hard_recall32) / epoch_size
            print(f"  {epoch_idx:>6}  {t:>6.4f}  {approx_mean:>14.4f}  {hard_frac:>10.4f}")
            diagnostics.append({
                "epoch_idx": epoch_idx,
                "t_value": float(t),
                "approx_mean_recall32": float(approx_mean),
                "hard_fraction": float(hard_frac),
            })

        out_key = "dataset_gradual" if schedule_name == "gradual" else "dataset_sudden"
        out_path = str(ROOT / cfg[out_key])

        save_config = {
            "schedule_name": schedule_name,
            "n_epochs": n_epochs,
            "epoch_size": epoch_size,
            "easy_pool_size": n_easy_actual,
            "hard_pool_size": n_hard_actual,
            "seed": seed,
            "schedule": [float(t) for t in schedule_t],
            "selection_criterion": "three_way_structural",
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
