"""Build dataset for experiment 23: DEEP-1B cluster-concentrated query drift."""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import faiss
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hnswlib

from src.drift.dataset import (
    characterise_drift,
    save_drift_dataset,
)
from src.drift.deep_drift import load_deep_hdf5
from src.drift.msmarco_drift import build_answerable_drift_sequence
from src.drift.sift_cluster_drift import (
    cluster_query_pool,
    get_entry_point_vec,
    routing_failure_diagnostic,
    select_far_cluster,
)


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _compute_groundtruth_epoch(gt_index, queries, k=100):
    """Run exact search against a pre-built FAISS index."""
    _, ids = gt_index.search(queries.astype(np.float32), k)
    return ids.astype(np.int32)


def _routing_failure_diag_with_index(hnsw_index, gt_index, uniform_queries,
                                      far_queries, ef_values, k):
    """Like routing_failure_diagnostic but reuses a pre-built FAISS index."""
    def gt_func(base, queries, k):
        _, ids = gt_index.search(queries.astype(np.float32), k)
        return ids.astype(np.int32)

    return routing_failure_diagnostic(
        hnsw_index, None, uniform_queries, far_queries, ef_values, k, gt_func
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument(
        "--schedule",
        choices=["schedule_gradual", "schedule_sudden"],
        default="schedule_gradual",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    schedule_tag = args.schedule
    schedule = cfg["drift"][schedule_tag]

    if schedule_tag == "schedule_gradual":
        output_path = str(ROOT / cfg["output"]["dataset_gradual_path"])
    else:
        output_path = str(ROOT / cfg["output"]["dataset_sudden_path"])

    if os.path.exists(os.path.join(output_path, "base.npy")):
        print(f"Dataset already exists at {output_path} — skipping build.")
        return

    print(f"Schedule: {schedule_tag}  ({len(schedule)} epochs)")
    print(f"Output:   {output_path}")

    n_base = cfg["data"].get("n_base", None)
    n_queries = cfg["data"].get("n_queries_pool", None)

    # --- Load vectors ---
    print(f"\nLoading data (n_base={n_base}, n_queries={n_queries})...")
    t0 = time.perf_counter()
    base, query_vecs = load_deep_hdf5(
        cfg["data"]["hdf5_path"], n_base=n_base, n_queries=n_queries
    )
    print(f"  base: {base.shape}  query pool: {query_vecs.shape}  ({time.perf_counter()-t0:.1f}s)")

    # --- Build FAISS index once for all ground truth queries ---
    print("\nBuilding FAISS index for ground truth (built once, reused per epoch)...")
    t0 = time.perf_counter()
    gt_index = faiss.IndexFlatL2(base.shape[1])
    gt_index.add(base.astype(np.float32))
    print(f"  FAISS index ready ({time.perf_counter()-t0:.1f}s)")

    # --- Load HNSW index ---
    index_path = str(ROOT / cfg["index"]["path"])
    print(f"\nLoading HNSW index from {index_path}...")
    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.load_index(index_path)

    # --- Cluster query pool ---
    print(f"\nClustering query pool into {cfg['cluster']['n_clusters']} clusters...")
    labels, centroids, cluster_diag = cluster_query_pool(
        query_vecs,
        n_clusters=cfg["cluster"]["n_clusters"],
        seed=cfg["cluster"]["seed"],
    )
    for i, sz in enumerate(cluster_diag["cluster_sizes"]):
        print(f"  cluster {i}: {sz} queries")

    # --- Entry point reference ---
    print("\nDetecting HNSW entry point...")
    ep_vec, ep_method = get_entry_point_vec(index, base)

    # --- Select far cluster ---
    print("\nSelecting far cluster...")
    chosen_idx, selection_diag = select_far_cluster(centroids, ep_vec)
    selection_diag["reference_method"] = ep_method
    for i, d in enumerate(selection_diag["all_distances_to_reference"]):
        marker = " <-- chosen" if i == chosen_idx else ""
        print(f"  cluster {i}: dist={d:.2f}{marker}")
    print(f"Chosen cluster: {chosen_idx}  (dist={selection_diag['chosen_distance_to_reference']:.2f})")

    # --- Routing failure diagnostic ---
    print("\nRunning routing failure diagnostic...")
    os.makedirs(output_path, exist_ok=True)

    rng = np.random.default_rng(42)
    uniform_sample = query_vecs[rng.integers(0, len(query_vecs), 500)]
    far_mask = labels == chosen_idx
    far_vecs = query_vecs[far_mask]
    replace = len(far_vecs) < 500
    far_sample = far_vecs[rng.choice(len(far_vecs), 500, replace=replace)]

    diag_rows = _routing_failure_diag_with_index(
        index, gt_index, uniform_sample, far_sample,
        ef_values=cfg["eval"]["diagnostic_ef_values"],
        k=cfg["eval"]["recall_k"],
    )

    with open(os.path.join(output_path, "routing_failure_diagnostic.json"), "w") as f:
        json.dump(diag_rows, f, indent=2)

    gap_ef32 = next((r["gap"] for r in diag_rows if r["ef"] == 32), None)
    if gap_ef32 is not None and gap_ef32 < 0.02:
        warnings.warn(
            f"Routing failure gap at ef=32 is {gap_ef32:.4f} < 0.02 — "
            "routing failure may be weak."
        )

    # --- Split query pool ---
    in_dist_vecs = query_vecs[~far_mask]
    far_cluster_vecs = far_vecs
    print(f"\nIn-distribution pool: {len(in_dist_vecs)} vectors")
    print(f"Far-cluster pool:     {len(far_cluster_vecs)} vectors")
    if len(far_cluster_vecs) < 200:
        raise ValueError(
            f"Far cluster has only {len(far_cluster_vecs)} vectors — need at least 200."
        )

    # --- Build drift sequence ---
    print("\nBuilding drift sequence...")
    epochs = build_answerable_drift_sequence(
        in_dist_vecs, far_cluster_vecs, schedule,
        cfg["drift"]["epoch_size"], cfg["drift"]["base_seed"]
    )
    print(f"  {len(epochs)} epochs, each {epochs[0].shape[0]} queries")

    # --- Ground truth (reuse FAISS index) ---
    print("\nComputing per-epoch ground truth...")
    groundtruth = []
    for i, ep in enumerate(epochs):
        gt = _compute_groundtruth_epoch(gt_index, ep, k=100)
        groundtruth.append(gt)
        if i % 5 == 0:
            print(f"  epoch {i}/{len(epochs)}")
    print(f"  epoch {len(epochs)}/{len(epochs)}")

    # --- characterise_drift ---
    print("\nCharacterising drift...")
    try:
        drift_diag = characterise_drift(base, epochs, schedule=schedule)
    except Exception as e:
        warnings.warn(f"characterise_drift raised: {e}")
        drift_diag = []

    # --- Save dataset ---
    print("\nSaving dataset...")
    save_config = {
        "schedule_tag": schedule_tag,
        "schedule": schedule,
        "n_epochs": len(epochs),
        "epoch_size": cfg["drift"]["epoch_size"],
        "n_base": int(base.shape[0]),
        "dim": int(base.shape[1]),
        "cluster": cfg["cluster"],
        "chosen_cluster_idx": chosen_idx,
    }
    save_drift_dataset(output_path, base, epochs, groundtruth, save_config, drift_diag)

    cluster_diagnostics_out = {
        "cluster_diagnostics": cluster_diag,
        "selection_diagnostics": selection_diag,
        "routing_failure_diagnostic": diag_rows,
    }
    diag_path = os.path.join(output_path, "cluster_diagnostics.json")
    with open(diag_path, "w") as f:
        json.dump(cluster_diagnostics_out, f, indent=2)
    print(f"Cluster diagnostics saved to {diag_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
