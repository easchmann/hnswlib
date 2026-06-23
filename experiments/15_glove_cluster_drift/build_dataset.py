"""Build cluster-reweighted drift dataset for GloVe-100 experiment."""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.drift.cluster_drift import (
    build_cluster_drift_sequence,
    build_query_clusters,
    characterise_cluster_drift,
    compute_groundtruth,
    load_glove_hdf5,
    save_cluster_drift_dataset,
    select_hot_clusters,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--schedule", default="schedule_gradual",
                        choices=["schedule_gradual", "schedule_sudden"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    hdf5_path = cfg["data"]["hdf5_path"]
    drift_cfg = cfg["drift"]
    schedule = drift_cfg[args.schedule]

    print(f"Loading GloVe-100 from {hdf5_path}...")
    base, query_pool = load_glove_hdf5(hdf5_path)
    print(f"  base:       {base.shape}  norms [{np.linalg.norm(base[:3], axis=1)}]")
    print(f"  query_pool: {query_pool.shape}  norms [{np.linalg.norm(query_pool[:3], axis=1)}]")

    print(f"\nClustering query pool into {drift_cfg['n_clusters']} clusters...")
    cluster_labels, centroids = build_query_clusters(
        query_pool, drift_cfg["n_clusters"], drift_cfg["base_seed"]
    )

    hot_ids = select_hot_clusters(centroids, drift_cfg["n_hot_clusters"])
    cold_centroid = centroids[[i for i in range(len(centroids)) if i not in hot_ids]].mean(axis=0)
    hot_centroid = centroids[hot_ids].mean(axis=0)
    sep = float(np.linalg.norm(hot_centroid - cold_centroid))
    print(f"  Hot clusters: {hot_ids}")
    print(f"  Hot–cold centroid separation: {sep:.4f}")

    print(f"\nBuilding drift sequence ({args.schedule}, {len(schedule)} epochs)...")
    epochs = build_cluster_drift_sequence(
        query_pool, cluster_labels, hot_ids,
        schedule, drift_cfg["epoch_size"], drift_cfg["base_seed"]
    )
    print(f"  Built {len(epochs)} epochs, each shape {epochs[0].shape}")

    print("\nComputing per-epoch ground truth with FAISS (k=100)...")
    groundtruth = []
    for epoch_idx, eq in enumerate(epochs):
        print(f"  epoch {epoch_idx}/{len(epochs)}...")
        gt = compute_groundtruth(base, eq, k=100, batch_size=1000)
        groundtruth.append(gt)

    print("\nCharacterising drift...")
    diagnostics = characterise_cluster_drift(base, epochs, schedule=schedule)

    if args.schedule == "schedule_gradual":
        out_path = str(ROOT / cfg["output"]["dataset_gradual_path"])
    else:
        out_path = str(ROOT / cfg["output"]["dataset_sudden_path"])

    save_config = {
        "schedule_name": args.schedule,
        "n_epochs": len(epochs),
        "epoch_size": drift_cfg["epoch_size"],
        "n_clusters": drift_cfg["n_clusters"],
        "n_hot_clusters": drift_cfg["n_hot_clusters"],
        "hot_cluster_ids": hot_ids,
        "base_seed": drift_cfg["base_seed"],
        "schedule": schedule,
    }

    print(f"\nSaving dataset to {out_path}...")
    save_cluster_drift_dataset(out_path, base, epochs, groundtruth, save_config, diagnostics)
    print("Done.")


if __name__ == "__main__":
    main()
