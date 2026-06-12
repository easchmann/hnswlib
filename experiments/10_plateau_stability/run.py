"""Experiment 10: plateau stability — does repair quiesce on a stable drift plateau?"""

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

SCHEDULE_TO_DATASET_KEY = {
    "schedule_sudden_plateau": "dataset_sudden_plateau_path",
    "schedule_gradual_plateau": "dataset_gradual_plateau_path",
}

SCHEDULE_NAMES = {
    "schedule_sudden_plateau": "sudden_plateau",
    "schedule_gradual_plateau": "gradual_plateau",
}


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
    print(f"  Saved {len(rows)} rows -> {out_path}")


def run_schedule(dataset, index_path, base, cfg, schedule_name):
    """Run adaptive method on one schedule; returns list of result rows."""
    adapt_cfg = cfg["adaptation"]
    k = cfg["eval"]["recall_k"]
    primary_ef = cfg["eval"]["primary_ef"]
    ef_values = cfg["eval"]["ef_search_values"]
    n_calib = adapt_cfg["n_calibration_epochs"]

    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.load_index(index_path)

    print("  Building spatial index...")
    centroids, cell_labels = build_spatial_index(
        base, n_cells=adapt_cfg["n_cells"], seed=42
    )

    print(f"  Calibrating on first {n_calib} epochs...")
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

    rows = []
    prev_edges = 0

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

        edges_now = cg._total_edges
        edges_added = edges_now - prev_edges
        prev_edges = edges_now

        print(
            f"  epoch {epoch_idx:2d}  recall={primary['recall']:.4f}  "
            f"drift={ep['drift_detected']}  repair={ep['repair_stats'] is not None}  "
            f"edges={edges_now}  +{edges_added}  eh={ep['mean_eh_this_epoch']:.4f}"
        )

        for ef in ef_values:
            r = ef_results[ef]
            rows.append({
                "epoch_idx": epoch_idx,
                "schedule": schedule_name,
                "recall_at_k": r["recall"],
                "mean_latency_ms": r["elapsed_ms"],
                "ef_search": ef,
                "drift_detected": ep["drift_detected"],
                "mmd_squared": ep["mmd_squared"],
                "n_conjugate_edges": edges_now,
                "edges_added_this_epoch": edges_added if ef == primary_ef else 0,
                "repair_happened": ep["repair_stats"] is not None,
                "mean_eh": ep["mean_eh_this_epoch"],
            })

    return rows


def run_all(config_path):
    cfg = _load_config(config_path)
    results_dir = ROOT / cfg["output"]["results_dir"]
    index_path = str(ROOT / cfg["output"]["index_path"])

    for schedule_key, dataset_key in SCHEDULE_TO_DATASET_KEY.items():
        schedule_name = SCHEDULE_NAMES[schedule_key]
        dataset_path = str(ROOT / cfg["output"][dataset_key])

        print(f"\n{'='*60}")
        print(f"SCHEDULE: {schedule_name}")
        print("=" * 60)

        dataset = load_drift_dataset(dataset_path)
        base = dataset["base"]

        rows = run_schedule(dataset, index_path, base, cfg, schedule_name)
        out_csv = str(results_dir / f"{schedule_name}.csv")
        _save_results(rows, out_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    args = parser.parse_args()
    run_all(args.config)


if __name__ == "__main__":
    main()
