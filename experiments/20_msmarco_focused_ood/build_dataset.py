"""Experiment 20: build focused OOD drift dataset (single semantic cluster)."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.drift.dataset import (
    characterise_drift,
    compute_groundtruth,
    save_drift_dataset,
)
from src.drift.msmarco_drift import (
    build_answerable_drift_sequence,
    load_msmarco_passages,
    load_msmarco_queries,
    load_qrels,
    split_queries_by_coverage,
)
from src.drift.msmarco_focused_drift import select_focused_ood_cluster


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


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
    data_cfg = cfg["data"]
    drift_cfg = cfg["drift"]
    focused_cfg = cfg["focused_ood"]

    schedule = drift_cfg[args.schedule]
    epoch_size = drift_cfg["epoch_size"]
    base_seed = drift_cfg["base_seed"]

    schedule_tag = "gradual" if args.schedule == "schedule_gradual" else "sudden"
    out_key = f"dataset_{schedule_tag}_path"
    output_path = str(ROOT / cfg["output"][out_key])

    embeddings_dir = data_cfg["embeddings_dir"]

    # base embeddings (must already exist) 
    base_cache = os.path.join(embeddings_dir, "base_embeddings.npy")
    if not os.path.exists(base_cache):
        raise FileNotFoundError(
            f"Base embeddings not found at {base_cache}. Run experiment 18 first."
        )
    print(f"Loading cached base embeddings from {base_cache}")
    base = np.load(base_cache)
    print(f"Base shape: {base.shape}")

    # base passage IDs (for coverage split) 
    print("Loading passage IDs from collection.tsv for coverage split...")
    passage_ids, _ = load_msmarco_passages(
        data_cfg["collection_tsv"], data_cfg["n_base_vectors"]
    )

    # query embeddings
    query_cache = os.path.join(embeddings_dir, "query_embeddings.npy")
    if not os.path.exists(query_cache):
        raise FileNotFoundError(
            f"Query embeddings not found at {query_cache}. Run experiment 18 first."
        )
    print(f"Loading cached query embeddings from {query_cache}")
    query_vecs = np.load(query_cache)
    print(f"Query pool shape: {query_vecs.shape}")

    # query IDs
    print("Loading query IDs from queries.dev.small.tsv...")
    query_ids, _ = load_msmarco_queries(data_cfg["queries_tsv"])
    print(f"Loaded {len(query_ids):,} query IDs.")

    #  load qrels
    print("Loading qrels...")
    qrels = load_qrels(data_cfg["qrels_tsv"])
    base_passage_id_set = set(passage_ids)
    print(f"Base passage ID set: {len(base_passage_id_set):,} unique IDs")

    # split queries by coverage
    in_dist_vecs, ood_vecs, split_diagnostics = split_queries_by_coverage(
        query_ids, query_vecs, qrels, base_passage_id_set
    )

    print(f"\nQuery split:")
    print(f"  Total queries:                {split_diagnostics['n_total_queries']:,}")
    print(f"  In-distribution (answerable): {split_diagnostics['n_in_dist']:,}")
    print(f"  OOD (unanswerable):           {split_diagnostics['n_ood']:,}")
    print(f"  No qrel (treated as OOD):     {split_diagnostics['n_no_qrel']:,}")

    if len(in_dist_vecs) < epoch_size:
        raise ValueError(
            f"In-distribution pool too small: {len(in_dist_vecs)} < epoch_size={epoch_size}"
        )

    #Sselect focused OOD cluster 
    print(f"\nClustering OOD queries into {focused_cfg['n_clusters']} clusters...")
    focused_ood, chosen_idx, cluster_diag = select_focused_ood_cluster(
        ood_vecs,
        n_clusters=focused_cfg["n_clusters"],
        seed=focused_cfg["seed"],
    )

    print(f"\nCluster sizes:")
    for k, sz in enumerate(cluster_diag["cluster_sizes"]):
        marker = " <-- chosen" if k == chosen_idx else ""
        mean_d = cluster_diag["all_cluster_mean_dists"][k]
        print(f"  cluster {k}: size={sz:4d}  mean_dist_to_centroid={mean_d:.4f}{marker}")

    print(f"\nChosen cluster idx: {chosen_idx}")
    print(f"Chosen cluster size: {cluster_diag['chosen_cluster_size']}")
    print(
        f"Chosen cluster mean dist to centroid: "
        f"{cluster_diag['chosen_cluster_mean_dist_to_centroid']:.4f}"
    )

    # Validate cluster size 
    if len(focused_ood) < focused_cfg["min_cluster_size"]:
        raise ValueError(
            f"Chosen OOD cluster too small: {len(focused_ood)} < "
            f"min_cluster_size={focused_cfg['min_cluster_size']}"
        )

    # Build drift sequence 
    print(f"\nBuilding {len(schedule)}-epoch drift sequence ({schedule_tag})...")
    print(f"OOD pool: focused cluster of {len(focused_ood)} queries")
    epochs = build_answerable_drift_sequence(
        in_dist_vecs, focused_ood, schedule, epoch_size, base_seed
    )
    print(f"Built {len(epochs)} epochs, each of shape {epochs[0].shape}")

    # Compute ground truth 
    print("\nComputing per-epoch ground truth (FAISS exact search, k=100)...")
    groundtruth_per_epoch = []
    for i, eq in enumerate(epochs):
        gt = compute_groundtruth(base, eq, k=100)
        groundtruth_per_epoch.append(gt)
        if i % 5 == 0:
            print(f"  ground truth epoch {i}/{len(epochs)}")

    # Drift characterisation
    print("\nCharacterising drift...")
    drift_diagnostics = characterise_drift(base, epochs, schedule=schedule)

    mmd2_vals = [d["mmd_squared"] for d in drift_diagnostics]
    ood_vals = [d["ood_distance"] for d in drift_diagnostics]
    for name, vals in [("MMD²", mmd2_vals), ("OOD distance", ood_vals)]:
        violations = sum(1 for i in range(1, len(vals)) if vals[i] < vals[i - 1] - 1e-8)
        if violations > 0:
            print(
                f"  WARNING: {name} has {violations} non-monotone step(s) "
                f"(expected for stochastic sampling)"
            )

    dataset_config = {
        "n_epochs": len(epochs),
        "epoch_size": epoch_size,
        "schedule": schedule,
        "schedule_name": schedule_tag,
        "base_seed": base_seed,
        "n_base_vectors": data_cfg["n_base_vectors"],
        "model_name": data_cfg["model_name"],
        "dim": base.shape[1],
        "focused_ood_cluster_idx": chosen_idx,
        "focused_ood_cluster_size": len(focused_ood),
    }

    save_drift_dataset(
        output_path, base, epochs, groundtruth_per_epoch, dataset_config, drift_diagnostics
    )

    combined_diag = {
        "cluster_diagnostics": cluster_diag,
        "split_diagnostics": split_diagnostics,
    }
    cluster_diag_path = os.path.join(output_path, "cluster_diagnostics.json")
    with open(cluster_diag_path, "w") as f:
        json.dump(combined_diag, f, indent=2)
    print(f"Cluster diagnostics saved to {cluster_diag_path}")


if __name__ == "__main__":
    main()
