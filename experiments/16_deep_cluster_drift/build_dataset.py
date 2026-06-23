"""Build cluster-reweighted drift dataset for DEEP-96 experiment."""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hnswlib

from src.drift.cluster_drift import (
    build_cluster_drift_sequence,
    build_query_clusters,
    characterise_cluster_drift,
    compute_groundtruth,
    save_cluster_drift_dataset,
)
from src.drift.deep_cluster_drift import (
    build_cluster_drift_sequence_bimodal,
    load_deep_hdf5,
    select_cold_clusters_by_recall,
    select_hot_clusters_by_recall,
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
    n_base = cfg["data"].get("n_base_vectors", None)
    drift_cfg = cfg["drift"]
    schedule = drift_cfg[args.schedule]
    eval_cfg = cfg["eval"]
    bimodal = drift_cfg.get("bimodal", False)

    print(f"Loading DEEP-96 from {hdf5_path}...")
    base, query_pool = load_deep_hdf5(hdf5_path, n_vectors=n_base)
    print(f"  base:       {base.shape}  norms [{np.linalg.norm(base[:3], axis=1)}]")
    print(f"  query_pool: {query_pool.shape}  norms [{np.linalg.norm(query_pool[:3], axis=1)}]")
    print(f"  mode:       {'bimodal (cold@t=0 → hot@t=1)' if bimodal else 'unimodal (uniform@t=0 → hot@t=1)'}")

    print(f"\nClustering query pool into {drift_cfg['n_clusters']} clusters...")
    cluster_labels, centroids = build_query_clusters(
        query_pool, drift_cfg["n_clusters"], drift_cfg["base_seed"]
    )

    hot_cluster_ids_cfg = drift_cfg.get("hot_cluster_ids", None)
    cold_cluster_ids_cfg = drift_cfg.get("cold_cluster_ids", None)

    if hot_cluster_ids_cfg is not None:
        hot_ids = hot_cluster_ids_cfg
        per_cluster_recall = {}
        print(f"  Using pre-configured hot clusters: {hot_ids} (skipping recall diagnostic)")
        if bimodal and cold_cluster_ids_cfg is not None:
            cold_ids = cold_cluster_ids_cfg
            print(f"  Using pre-configured cold clusters: {cold_ids}")
        elif bimodal:
            raise ValueError("bimodal=true requires cold_cluster_ids when hot_cluster_ids is set")
    else:
        index_path = str(ROOT / cfg["output"]["index_path"])
        print(f"\nLoading pre-built index from {index_path} for recall diagnostic...")
        index = hnswlib.Index(space="l2", dim=base.shape[1])
        index.load_index(index_path, max_elements=base.shape[0])

        hot_ids, per_cluster_recall = select_hot_clusters_by_recall(
            index, base, query_pool, cluster_labels,
            drift_cfg["n_hot_clusters"], eval_cfg["recall_k"], eval_cfg["primary_ef"]
        )

        if bimodal:
            n_cold = drift_cfg.get("n_cold_clusters", drift_cfg["n_hot_clusters"])
            cold_ids = select_cold_clusters_by_recall(per_cluster_recall, n_cold, hot_ids)
            cold_recall_min = per_cluster_recall[cold_ids[-1]]
            hot_recall_max = per_cluster_recall[hot_ids[-1]]
            bimodal_gap = cold_recall_min - hot_recall_max
            if bimodal_gap < 0.05:
                print(f"  WARNING: bimodal gap is only {bimodal_gap:.4f} pp — drift signal may be weak")
        else:
            hot_recall_max = per_cluster_recall[hot_ids[-1]]
            sorted_clusters = sorted(per_cluster_recall, key=per_cluster_recall.__getitem__)
            cold_recall_min = per_cluster_recall[sorted_clusters[drift_cfg["n_hot_clusters"]]]
            gap = cold_recall_min - hot_recall_max
            if gap < 0.03:
                print(f"  WARNING: recall gap is only {gap:.4f} pp — drift signal may be weak")

    print(f"\nBuilding drift sequence ({args.schedule}, {len(schedule)} epochs)...")
    if bimodal:
        epochs = build_cluster_drift_sequence_bimodal(
            query_pool, cluster_labels, hot_ids, cold_ids,
            schedule, drift_cfg["epoch_size"], drift_cfg["base_seed"]
        )
    else:
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
        "cold_cluster_ids": cold_ids if bimodal else None,
        "bimodal": bimodal,
        "per_cluster_recall": {str(k): v for k, v in per_cluster_recall.items()},
        "base_seed": drift_cfg["base_seed"],
        "schedule": schedule,
    }

    print(f"\nSaving dataset to {out_path}...")
    save_cluster_drift_dataset(out_path, base, epochs, groundtruth, save_config, diagnostics)
    print("Done.")


if __name__ == "__main__":
    main()
