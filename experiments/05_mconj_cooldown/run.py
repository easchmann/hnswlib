"""Experiment 05: M_conj × repair_cooldown focused sweep."""

import argparse
import csv
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


def _compute_recall(result_ids, groundtruth, k):
    hits = []
    for i in range(len(result_ids)):
        true_set = set(groundtruth[i, :k].tolist())
        hits.append(len(set(result_ids[i].tolist()) & true_set) / k)
    return float(np.mean(hits))


def _save_results(rows, out_path):
    if not rows:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = sorted(set(k for r in rows for k in r))
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    print(f"  Saved {len(rows)} rows to {out_path}")


def run_condition(dataset, index_path, base, config, condition):
    """Run one condition; returns list of result rows."""
    adapt_cfg = config["adaptation"]
    k = config["eval"]["recall_k"]
    primary_ef = config["eval"]["primary_ef"]
    ef_values = config["eval"]["ef_search_values"]
    n_calib = adapt_cfg.get("n_calibration_epochs", 5)

    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.load_index(index_path)

    print("  Building spatial index...")
    centroids, cell_labels = build_spatial_index(
        base, n_cells=adapt_cfg["n_cells"], seed=42
    )

    print(f"  Calibrating detector on first {n_calib} epochs...")
    calib_eh = []
    calib_cell_ids = []
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
        "use_diversity": False,
        "two_hop": True,
        "repair_cooldown": condition["repair_cooldown"],
    }

    cg = ConjugateGraph(
        M_conj=condition["M_conj"],
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

    results = []
    for epoch_idx, (queries, gt) in enumerate(
            zip(dataset["epochs"], dataset["groundtruth"])):

        ef_results = {}
        for ef in ef_values:
            t0 = time.perf_counter()
            ids, dists = mgr.search_enhanced(queries, k=k, ef_search=ef)
            elapsed_ms = (time.perf_counter() - t0) * 1000 / len(queries)
            ef_results[ef] = {
                "ids": ids,
                "dists": dists,
                "recall": _compute_recall(ids, gt, k),
                "elapsed_ms": elapsed_ms,
            }

        primary = ef_results[primary_ef]
        ep = mgr.process_epoch(queries, primary["ids"], primary["dists"], k)

        print(
            f"  epoch {epoch_idx:2d}  recall@{primary_ef}={primary['recall']:.4f}  "
            f"drift={ep['drift_detected']}  edges={cg._total_edges}"
        )

        for ef in ef_values:
            r = ef_results[ef]
            results.append({
                "epoch_idx": epoch_idx,
                "condition": condition["name"],
                "recall_at_k": r["recall"],
                "mean_latency_ms": r["elapsed_ms"],
                "queries_per_second": 1000.0 / r["elapsed_ms"] if r["elapsed_ms"] > 0 else 0.0,
                "ef_search": ef,
                "drift_detected": ep["drift_detected"],
                "mmd_squared": ep["mmd_squared"],
                "n_conjugate_edges": cg._total_edges,
                "repair_happened": ep["repair_stats"] is not None,
                "mean_eh": ep["mean_eh_this_epoch"],
            })

    return results


def run_all(config_path):
    cfg = _load_config(config_path)
    results_dir = ROOT / cfg["output"]["results_dir"]
    index_path = str(ROOT / cfg["output"]["index_path"])

    dataset = load_drift_dataset(str(ROOT / cfg["output"]["dataset_gradual_path"]))
    base = dataset["base"]

    for condition in cfg["conditions"]:
        name = condition["name"]
        print(f"\n{'='*60}")
        print(
            f"CONDITION: {name}  "
            f"(M_conj={condition['M_conj']}, "
            f"repair_cooldown={condition['repair_cooldown']})"
        )
        print("=" * 60)

        rows = run_condition(dataset, index_path, base, cfg, condition)
        out_csv = str(results_dir / f"{name}.csv")
        _save_results(rows, out_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.yaml"),
    )
    args = parser.parse_args()
    run_all(args.config)


if __name__ == "__main__":
    main()
