"""Experiment 21: build top-N centroid-proximity focused OOD datasets."""

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
from src.drift.msmarco_focused_drift import (
    select_focused_ood_cluster,
    select_top_n_ood_queries,
)


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
    parser.add_argument(
        "--top_n",
        type=int,
        required=True,
        help="Number of OOD queries to select by centroid proximity",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    data_cfg = cfg["data"]
    drift_cfg = cfg["drift"]
    focused_cfg = cfg["focused_ood"]

    if args.top_n not in focused_cfg["top_n_values"]:
        raise ValueError(
            f"--top_n {args.top_n} not in config top_n_values {focused_cfg['top_n_values']}"
        )

    schedule = drift_cfg[args.schedule]
    epoch_size = drift_cfg["epoch_size"]
    base_seed = drift_cfg["base_seed"]

    schedule_tag = "gradual" if args.schedule == "schedule_gradual" else "sudden"
    output_path = str(
        ROOT / cfg["output"]["results_dir"] / f"dataset_top{args.top_n}_{schedule_tag}"
    )

    embeddings_dir = data_cfg["embeddings_dir"]

    # ---- base embeddings (must already exist) ------------------------------------
    base_cache = os.path.join(embeddings_dir, "base_embeddings.npy")
    if not os.path.exists(base_cache):
        raise FileNotFoundError(f"Base embeddings not found at {base_cache}. Run exp18 first.")
    print(f"Loading cached base embeddings from {base_cache}")
    base = np.load(base_cache)
    print(f"Base shape: {base.shape}")

    # ---- base passage IDs --------------------------------------------------------
    print("Loading passage IDs from collection.tsv...")
    passage_ids, _ = load_msmarco_passages(
        data_cfg["collection_tsv"], data_cfg["n_base_vectors"]
    )

    # ---- query embeddings (must already exist) -----------------------------------
    query_cache = os.path.join(embeddings_dir, "query_embeddings.npy")
    if not os.path.exists(query_cache):
        raise FileNotFoundError(f"Query embeddings not found at {query_cache}. Run exp18 first.")
    print(f"Loading cached query embeddings from {query_cache}")
    query_vecs = np.load(query_cache)

    print("Loading query IDs...")
    query_ids, _ = load_msmarco_queries(data_cfg["queries_tsv"])

    print("Loading qrels...")
    qrels = load_qrels(data_cfg["qrels_tsv"])
    base_passage_id_set = set(passage_ids)

    # ---- split -------------------------------------------------------------------
    in_dist_vecs, ood_vecs, split_diagnostics = split_queries_by_coverage(
        query_ids, query_vecs, qrels, base_passage_id_set
    )
    print(f"\nQuery split:")
    print(f"  In-distribution: {split_diagnostics['n_in_dist']:,}")
    print(f"  OOD:             {split_diagnostics['n_ood']:,}")

    if len(in_dist_vecs) < epoch_size:
        raise ValueError(f"In-dist pool too small: {len(in_dist_vecs)} < {epoch_size}")

    # ---- find tightest cluster centroid ------------------------------------------
    print(f"\nClustering OOD queries into {focused_cfg['n_clusters']} clusters...")
    _, chosen_idx, cluster_diag = select_focused_ood_cluster(
        ood_vecs,
        n_clusters=focused_cfg["n_clusters"],
        seed=focused_cfg["seed"],
    )
    centroid = cluster_diag["centroid"]
    print(f"Tightest cluster: idx={chosen_idx}, "
          f"size={cluster_diag['chosen_cluster_size']}, "
          f"mean_dist={cluster_diag['chosen_cluster_mean_dist_to_centroid']:.4f}")

    # ---- select top-N by centroid proximity --------------------------------------
    print(f"\nSelecting top {args.top_n} OOD queries by proximity to centroid...")
    focused_ood, topn_diag = select_top_n_ood_queries(ood_vecs, centroid, args.top_n)
    print(f"  Selected: {topn_diag['n_returned']} queries")
    print(f"  Mean dist to centroid: {topn_diag['mean_dist_to_centroid']:.4f}")
    print(f"  Max dist to centroid:  {topn_diag['max_dist_to_centroid']:.4f}")

    # ---- build drift sequence ----------------------------------------------------
    print(f"\nBuilding {len(schedule)}-epoch drift sequence ({schedule_tag})...")
    epochs = build_answerable_drift_sequence(
        in_dist_vecs, focused_ood, schedule, epoch_size, base_seed
    )
    print(f"Built {len(epochs)} epochs, each of shape {epochs[0].shape}")

    # ---- ground truth ------------------------------------------------------------
    print("\nComputing per-epoch ground truth (FAISS exact, k=100)...")
    groundtruth_per_epoch = []
    for i, eq in enumerate(epochs):
        gt = compute_groundtruth(base, eq, k=100)
        groundtruth_per_epoch.append(gt)
        if i % 5 == 0:
            print(f"  ground truth epoch {i}/{len(epochs)}")

    # ---- drift characterisation --------------------------------------------------
    print("\nCharacterising drift...")
    drift_diagnostics = characterise_drift(base, epochs, schedule=schedule)

    mmd2_vals = [d["mmd_squared"] for d in drift_diagnostics]
    ood_vals = [d["ood_distance"] for d in drift_diagnostics]
    for name, vals in [("MMD²", mmd2_vals), ("OOD distance", ood_vals)]:
        violations = sum(1 for i in range(1, len(vals)) if vals[i] < vals[i - 1] - 1e-8)
        if violations > 0:
            print(f"  WARNING: {name} has {violations} non-monotone step(s)")

    # ---- save --------------------------------------------------------------------
    dataset_config = {
        "n_epochs": len(epochs),
        "epoch_size": epoch_size,
        "schedule": schedule,
        "schedule_name": schedule_tag,
        "base_seed": base_seed,
        "n_base_vectors": data_cfg["n_base_vectors"],
        "model_name": data_cfg["model_name"],
        "dim": base.shape[1],
        "top_n": args.top_n,
        "chosen_cluster_idx": chosen_idx,
    }

    save_drift_dataset(
        output_path, base, epochs, groundtruth_per_epoch, dataset_config, drift_diagnostics
    )

    combined_diag = {
        "cluster_diagnostics": cluster_diag,
        "topn_diagnostics": topn_diag,
        "split_diagnostics": split_diagnostics,
        "top_n": args.top_n,
    }
    diag_path = os.path.join(output_path, "selection_diagnostics.json")
    with open(diag_path, "w") as f:
        json.dump(combined_diag, f, indent=2)
    print(f"Selection diagnostics saved to {diag_path}")


if __name__ == "__main__":
    main()
