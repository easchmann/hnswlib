"""Build sparse-coverage cluster drift dataset for DEEP-96 experiment (exp32)."""

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
    save_cluster_drift_dataset,
    select_sparse_clusters,
)
from src.drift.deep_cluster_drift import load_deep_hdf5


def _print_density_diagnostic(centroids, mean_dists, hot_ids):
    """Print sparse vs dense cluster density comparison."""
    n = len(mean_dists)
    hot_set = set(hot_ids)
    dense_ids = np.argsort(mean_dists)[:len(hot_ids)].tolist()

    print(f"\n  Cluster density diagnostic (mean dist to {{}}-NN):")
    print(f"  {'Cluster':>8}  {'MeanDist':>10}  {'Type':>8}")
    for i in range(n):
        tag = "HOT(sparse)" if i in hot_set else ("dense" if i in dense_ids else "")
        if tag:
            print(f"  {i:>8}  {mean_dists[i]:>10.4f}  {tag:>8}")

    hot_dists = mean_dists[hot_ids]
    dense_dists = mean_dists[dense_ids]
    ratio = hot_dists.mean() / (dense_dists.mean() + 1e-12)
    print(f"\n  Hot (sparse) mean dist:   {hot_dists.mean():.4f}  (min {hot_dists.min():.4f})")
    print(f"  Dense cluster mean dist:  {dense_dists.mean():.4f}  (max {dense_dists.max():.4f})")
    print(f"  Sparse/dense ratio:       {ratio:.2f}x")
    if ratio < 1.5:
        print("  WARNING: ratio < 1.5 — cluster selection is not discriminative enough; "
              "consider increasing n_clusters or k_density")
    print()


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

    print(f"Loading DEEP-96 from {hdf5_path}...")
    base, query_pool = load_deep_hdf5(hdf5_path, n_vectors=n_base)
    print(f"  base:       {base.shape}  norms [{np.linalg.norm(base[:3], axis=1)}]")
    print(f"  query_pool: {query_pool.shape}  norms [{np.linalg.norm(query_pool[:3], axis=1)}]")

    print(f"\nClustering query pool into {drift_cfg['n_clusters']} clusters...")
    cluster_labels, centroids = build_query_clusters(
        query_pool, drift_cfg["n_clusters"], drift_cfg["base_seed"]
    )
    print(f"  Cluster sizes: min={np.bincount(cluster_labels).min()}  "
          f"max={np.bincount(cluster_labels).max()}")

    hot_cluster_ids_cfg = drift_cfg.get("hot_cluster_ids", None)

    if hot_cluster_ids_cfg is not None:
        hot_ids = hot_cluster_ids_cfg
        print(f"  Using pre-configured hot clusters: {hot_ids} (skipping density diagnostic)")
    else:
        k_density = drift_cfg.get("k_density", 200)
        n_hot = drift_cfg["n_hot_clusters"]
        print(f"\nSelecting {n_hot} sparsest clusters (k_density={k_density})...")

        import faiss
        nn_index = faiss.IndexFlatL2(base.shape[1])
        nn_index.add(base.astype(np.float32))
        dists_sq, _ = nn_index.search(centroids.astype(np.float32), k_density)
        mean_dists = dists_sq.mean(axis=1)

        hot_ids = select_sparse_clusters(base, centroids, n_hot=n_hot, k_density=k_density)
        _print_density_diagnostic(centroids, mean_dists, hot_ids)
        print(f"  Selected sparse hot clusters: {hot_ids}")

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
        "k_density": drift_cfg.get("k_density", 200),
        "cluster_selection": drift_cfg.get("cluster_selection", "sparse_coverage"),
        "hot_cluster_ids": hot_ids,
        "base_seed": drift_cfg["base_seed"],
        "schedule": schedule,
    }

    print(f"\nSaving dataset to {out_path}...")
    save_cluster_drift_dataset(out_path, base, epochs, groundtruth, save_config, diagnostics)
    print("Done.")


if __name__ == "__main__":
    main()
