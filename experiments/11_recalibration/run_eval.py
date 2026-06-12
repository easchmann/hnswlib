"""Experiment 11: rolling recalibration under permanent drift plateau."""

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
    print(f"  Saved {len(rows)} rows -> {out_path}")


def run_condition(dataset, index_path, base, cfg, condition_name, recalibration_enabled):
    """Run one condition (enabled / disabled); returns list of result rows."""
    adapt_cfg = cfg["adaptation"]
    eval_cfg = cfg["eval"]
    k = eval_cfg["recall_k"]
    ef = eval_cfg["ef_search"]
    n_calib = adapt_cfg["n_calibration_epochs"]
    n_epochs_cap = eval_cfg.get("n_epochs") or len(dataset["epochs"])

    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.load_index(index_path)

    print("  Building spatial index...")
    centroids, cell_labels = build_spatial_index(
        base, n_cells=adapt_cfg["n_cells"], seed=42
    )

    print(f"  Calibrating on first {n_calib} epochs...")
    calib_eh, calib_cell_ids = [], []
    index.set_ef(ef)
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
        "n_epochs": n_epochs_cap,
        "use_diversity": adapt_cfg.get("diversity", False),
        "two_hop": adapt_cfg.get("two_hop", True),
        "repair_cooldown": adapt_cfg["repair_cooldown"],
        "recalibration_enabled": recalibration_enabled,
        "recal_slope_window": adapt_cfg.get("recal_slope_window", 8),
        "recal_slope_threshold": adapt_cfg.get("recal_slope_threshold", 0.10),
        "recal_min_epochs": adapt_cfg.get("recal_min_epochs", 10),
        "recal_forget_age": adapt_cfg.get("recal_forget_age", 10),
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
    epochs = list(zip(dataset["epochs"], dataset["groundtruth"]))[:n_epochs_cap]

    for epoch_idx, (queries, gt) in enumerate(epochs):
        t0 = time.perf_counter()
        ids, dists = mgr.search_enhanced(queries, k=k, ef_search=ef)
        elapsed_ms = (time.perf_counter() - t0) * 1000 / len(queries)

        recall = _compute_recall(ids, gt, k)
        ep = mgr.process_epoch(queries, ids, dists, k)

        repair = ep["repair_stats"]
        edges_added = repair["edges_added"] if repair is not None else 0

        print(
            f"  [{condition_name}] epoch {epoch_idx:2d}  recall={recall:.4f}  "
            f"drift={ep['drift_detected']}  mmd2={ep['mmd_squared']:.5f}  "
            f"edges_total={cg._total_edges}  +{edges_added}  "
            f"recal={ep['recalibrated']}"
        )

        rows.append({
            "epoch_idx": epoch_idx,
            "condition": condition_name,
            "recall_at_k": recall,
            "mean_latency_ms": elapsed_ms,
            "drift_detected": ep["drift_detected"],
            "mmd_squared": ep["mmd_squared"],
            "total_edges": cg._total_edges,
            "edges_added": edges_added,
            "repair_happened": repair is not None,
            "recalibrated": ep["recalibrated"],
            "mean_eh": ep["mean_eh_this_epoch"],
        })

    return rows


def run_all(config_path, recalibration="both"):
    cfg = _load_config(config_path)
    results_dir = ROOT / cfg["output"]["results_dir"]
    index_path = str(ROOT / cfg["data"]["index_path"])
    dataset_path = str(ROOT / cfg["data"]["dataset_path"])

    dataset = load_drift_dataset(dataset_path)
    base = dataset["base"]

    all_conditions = [(True, "recal_enabled"), (False, "recal_disabled")]
    if recalibration == "enabled":
        conditions = [all_conditions[0]]
    elif recalibration == "disabled":
        conditions = [all_conditions[1]]
    else:
        conditions = all_conditions

    for enabled, name in conditions:
        print(f"\n{'='*60}")
        print(f"CONDITION: {name}")
        print("=" * 60)
        rows = run_condition(dataset, index_path, base, cfg, name, enabled)
        out_csv = str(results_dir / f"{name}.csv")
        _save_results(rows, out_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument(
        "--recalibration",
        choices=["enabled", "disabled", "both"],
        default="both",
    )
    args = parser.parse_args()
    run_all(args.config, recalibration=args.recalibration)


if __name__ == "__main__":
    main()
