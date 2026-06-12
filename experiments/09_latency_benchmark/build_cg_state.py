"""Build conjugate graph state snapshots for latency benchmarking.

Runs 25 epochs of adaptation on the gradual drift dataset, saving the
conjugate graph at each checkpoint epoch and benchmark queries/results.
Run once; subsequent benchmark runs load from disk.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hnswlib

from src.drift.adapter import AdaptationManager
from src.drift.conjugate_graph import ConjugateGraph
from src.drift.dataset import load_drift_dataset
from src.drift.detector import (
    DriftDetector,
    assign_cell,
    build_spatial_index,
    compute_eh_batch,
)


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _out_dir(cfg):
    p = ROOT / cfg["output"]["results_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_and_save(config_path):
    cfg = _load_config(config_path)
    adapt_cfg = cfg["adaptation"]
    bench_cfg = cfg["benchmark"]
    out_dir = _out_dir(cfg)
    cg_dir = out_dir / "cg_states"
    cg_dir.mkdir(exist_ok=True)

    checkpoint_epochs = set(cfg["checkpoint_epochs"])
    bench_epoch = bench_cfg["bench_epoch"]
    k = bench_cfg["recall_k"]
    primary_ef = bench_cfg["primary_ef"]
    n_calib = adapt_cfg["n_calibration_epochs"]

    dataset = load_drift_dataset(str(ROOT / cfg["output"]["dataset_gradual_path"]))
    base = dataset["base"]
    index_path = str(ROOT / cfg["output"]["index_path"])

    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.load_index(index_path)

    print("Building spatial index...")
    centroids, cell_labels = build_spatial_index(
        base, n_cells=adapt_cfg["n_cells"], seed=42
    )

    print(f"Calibrating on first {n_calib} epochs...")
    calib_eh, calib_cell_ids = [], []
    index.set_ef(primary_ef)
    for i in range(n_calib):
        queries = dataset["epochs"][i]
        labels, _ = index.knn_query(queries, k=k)
        eh_vals = compute_eh_batch([base[labels[j]] for j in range(len(queries))])
        cids = assign_cell(queries, centroids)
        calib_eh.extend(eh_vals)
        calib_cell_ids.extend(cids.tolist())

    detector = DriftDetector(
        window_size=adapt_cfg["window_size"],
        n_cells=adapt_cfg["n_cells"],
        n_rff=adapt_cfg["n_rff"],
    )
    detector.hot_cell_lambda = adapt_cfg["hot_cell_lambda"]
    detector.calibrate(np.array(calib_eh), np.array(calib_cell_ids, dtype=np.int32))

    mgr_config = {
        "M_candidates": adapt_cfg["M_candidates"],
        "repair_ef_search": adapt_cfg["ef_repair"],
        "max_repair_nodes": adapt_cfg["max_repair_nodes"],
        "eh_threshold_percentile": adapt_cfg["eh_threshold_percentile"],
        "rng_relaxation": adapt_cfg["rng_relaxation"],
        "n_epochs": len(dataset["epochs"]),
        "use_diversity": adapt_cfg.get("diversity", False),
        "two_hop": adapt_cfg.get("two_hop", True),
        "repair_cooldown": adapt_cfg["repair_cooldown"],
    }

    cg = ConjugateGraph(
        M_conj=adapt_cfg["M_conj"],
        max_total_edges=adapt_cfg["max_total_edges"],
    )
    mgr = AdaptationManager(
        index=index,
        base=base,
        conjugate_graph=cg,
        detector=detector,
        cell_labels=cell_labels,
        centroids=centroids,
        config=mgr_config,
    )

    print("\nRunning 25 adaptation epochs and saving CG checkpoints...")
    for epoch_idx, (queries, gt) in enumerate(
            zip(dataset["epochs"], dataset["groundtruth"])):

        index.set_ef(primary_ef)
        ids, dists = index.knn_query(queries, k=k)
        ep = mgr.process_epoch(queries, ids, dists, k)

        print(
            f"  epoch {epoch_idx:2d}  edges={cg._total_edges:>6d}  "
            f"repair={ep['repair_stats'] is not None}"
        )

        if epoch_idx in checkpoint_epochs:
            cg_path = cg_dir / f"cg_ep{epoch_idx:02d}.json"
            cg.save(str(cg_path))
            print(f"  => saved CG checkpoint: {cg_path.name}  ({cg._total_edges} edges)")

        if epoch_idx == bench_epoch:
            # Save benchmark queries, ground truth, and pre-computed HNSW results
            np.save(str(out_dir / "bench_queries.npy"), queries.astype(np.float32))
            np.save(str(out_dir / "bench_gt.npy"), gt.astype(np.int32))
            # Pre-compute batch knn results for all ef values (used by repair benchmark)
            index.set_ef(primary_ef)
            batch_ids, batch_dists = index.knn_query(queries, k=k)
            np.save(str(out_dir / "bench_hnsw_ids.npy"), batch_ids.astype(np.int64))
            np.save(str(out_dir / "bench_hnsw_dists.npy"), batch_dists.astype(np.float32))
            print(f"  => saved benchmark queries/gt/hnsw_results for epoch {bench_epoch}")

    # Also save a zero-edge CG for the edge=0 baseline
    cg_zero_path = cg_dir / "cg_ep00_zero.json"
    cg_zero = ConjugateGraph(M_conj=adapt_cfg["M_conj"], max_total_edges=adapt_cfg["max_total_edges"])
    cg_zero.save(str(cg_zero_path))
    print(f"\nSaved zero-edge CG: {cg_zero_path.name}")

    print(f"\nAll checkpoints saved to {cg_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.yaml"),
    )
    args = parser.parse_args()
    build_and_save(args.config)


if __name__ == "__main__":
    main()
