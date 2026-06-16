"""Experiment 12: compare drift signals (EH, QRD, NND, CSR, joint EH&QRD (in second round)) alongside recall@10.
"""

import argparse
import csv
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import faiss
import hnswlib

from src.drift.adapter import AdaptationManager
from src.drift.conjugate_graph import ConjugateGraph
from src.drift.dataset import load_drift_dataset
from src.drift.detector import (
    DriftDetector,
    DriftDetector2D,
    _SIGNAL_BATCH_FN,
    assign_cell,
    build_spatial_index,
    compute_joint_batch,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _compute_recall_faiss(base, queries, result_ids, k):
    """True recall@k via exact faiss search — never via HNSW."""
    dim = base.shape[1]
    index_flat = faiss.IndexFlatL2(dim)
    index_flat.add(base.astype(np.float32))
    _, gt = index_flat.search(queries.astype(np.float32), k)
    hits = []
    for i in range(len(result_ids)):
        true_set = set(gt[i].tolist())
        hits.append(len(set(result_ids[i].tolist()) & true_set) / k)
    return float(np.mean(hits)), gt


def _build_faiss_gt(base, queries, k):
    """Exact k-NN ground truth for a batch of queries."""
    index_flat = faiss.IndexFlatL2(base.shape[1])
    index_flat.add(base.astype(np.float32))
    _, gt = index_flat.search(queries.astype(np.float32), k)
    return gt


def _recall_from_gt(result_ids, gt, k):
    hits = []
    for i in range(len(result_ids)):
        true_set = set(gt[i, :k].tolist())
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
    log.info(f"Saved {len(rows)} rows -> {out_path}")


def _try_get_cpp_stats(index):
    """Try to fetch C++ last-query stats; return None if unavailable."""
    try:
        return index.get_last_query_stats()
    except Exception:
        return None


def run(config_path):
    cfg = _load_config(config_path)
    results_dir = ROOT / cfg["output"]["results_dir"]
    dataset_path = str(ROOT / cfg["output"]["dataset_gradual_path"])
    index_path = str(ROOT / cfg["output"]["index_path"])

    adapt_cfg = cfg["adaptation"]
    eval_cfg = cfg["eval"]
    k = eval_cfg["recall_k"]
    ef = eval_cfg["ef_search"]
    n_calib = adapt_cfg["n_calibration_epochs"]
    signal_names = cfg.get("signals", ["eh", "qrd", "nnd", "csr"])

    log.info("Loading dataset...")
    dataset = load_drift_dataset(dataset_path)
    base = dataset["base"]
    log.info(f"  base: {base.shape}, epochs: {len(dataset['epochs'])}")

    log.info("Loading index...")
    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.load_index(index_path)

    # Check C++ stats availability once
    index.set_ef(ef)
    dummy_q = base[:1]
    index.knn_query(dummy_q, k=k)
    cpp_stats_available = _try_get_cpp_stats(index) is not None
    if not cpp_stats_available:
        warnings.warn("get_last_query_stats() unavailable — skipping C++ signals")
    log.info(f"C++ query stats available: {cpp_stats_available}")

    log.info("Building spatial index...")
    centroids, cell_labels = build_spatial_index(
        base, n_cells=adapt_cfg["n_cells"], seed=42
    )

    log.info(f"Calibrating on first {n_calib} epochs...")
    # "joint" is a 2D signal — handled separately from scalar signals
    scalar_signals = [s for s in signal_names if s != "joint"]
    calib_vals = {s: [] for s in scalar_signals}
    calib_joint_feats = []
    calib_cell_ids = []
    index.set_ef(ef)
    for i in range(n_calib):
        queries = dataset["epochs"][i]
        labels, dists = index.knn_query(queries, k=k)
        result_ids = np.array(labels, dtype=np.int64)
        result_vecs = [base[result_ids[j]] for j in range(len(queries))]
        cell_ids = assign_cell(queries, centroids)
        calib_cell_ids.extend(cell_ids.tolist())
        for sig in scalar_signals:
            fn = _SIGNAL_BATCH_FN[sig]
            vals = fn(result_vecs, queries)
            calib_vals[sig].extend(vals.tolist())
        if "joint" in signal_names:
            calib_joint_feats.append(compute_joint_batch(result_vecs, queries))

    calib_cell_arr = np.array(calib_cell_ids, dtype=np.int32)

    # Calibrate one DriftDetector per scalar signal
    signal_detectors = {}
    for sig in scalar_signals:
        ref = np.array(calib_vals[sig], dtype=np.float64)
        det = DriftDetector(
            window_size=adapt_cfg["window_size"],
            n_cells=adapt_cfg["n_cells"],
            n_rff=adapt_cfg["n_rff"],
            signal=sig,
        )
        det.hot_cell_lambda = adapt_cfg["hot_cell_lambda"]
        log.info(f"  Calibrating {sig} detector...")
        det.calibrate(ref, calib_cell_arr)
        signal_detectors[sig] = det

    # Calibrate joint 2D detector
    joint_detector = None
    if "joint" in signal_names:
        ref_joint = np.vstack(calib_joint_feats)
        joint_detector = DriftDetector2D(
            window_size=adapt_cfg["window_size"],
            n_cells=adapt_cfg["n_cells"],
            n_rff=adapt_cfg["n_rff"],
        )
        joint_detector.hot_cell_lambda = adapt_cfg["hot_cell_lambda"]
        log.info("  Calibrating joint (EH+QRD) detector...")
        joint_detector.calibrate(ref_joint, calib_cell_arr)

    # Build AdaptationManager using the EH detector for repair
    eh_detector = signal_detectors["eh"]
    mgr_config = {
        "M_candidates": adapt_cfg["M_candidates"],
        "repair_ef_search": adapt_cfg["ef_repair"],
        "max_repair_nodes": adapt_cfg["max_repair_nodes"],
        "eh_threshold_percentile": adapt_cfg["eh_threshold_percentile"],
        "rng_relaxation": adapt_cfg["rng_relaxation"],
        "n_epochs": len(dataset["epochs"]),
        "use_diversity": adapt_cfg.get("diversity", False),
        "two_hop": adapt_cfg.get("two_hop", True),
        "repair_cooldown": adapt_cfg.get("repair_cooldown", 0),
        "recalibration_enabled": False,
    }
    cg = ConjugateGraph(
        M_conj=adapt_cfg["M_conj"],
        max_total_edges=adapt_cfg["max_total_edges"],
    )
    mgr = AdaptationManager(
        index=index,
        base=base,
        conjugate_graph=cg,
        detector=eh_detector,
        cell_labels=cell_labels,
        centroids=centroids,
        config=mgr_config,
    )

    rows = []
    prev_edges = 0

    log.info("Running adaptive loop...")
    for epoch_idx, queries in enumerate(dataset["epochs"]):
        gt = dataset["groundtruth"][epoch_idx]

        # Enhanced search (EH-based repair runs inside mgr.process_epoch)
        ids, dists = mgr.search_enhanced(queries, k=k, ef_search=ef)

        # True recall via pre-built ground truth from dataset
        recall = _recall_from_gt(ids, gt, k)

        # Compute result vectors for signal comparison
        result_vecs = [base[ids[j]] for j in range(len(queries))]
        cell_ids = assign_cell(queries, centroids)

        # Advance the AdaptationManager (uses eh_detector internally)
        ep = mgr.process_epoch(queries, ids, dists, k)

        edges_now = cg._total_edges
        edges_added = edges_now - prev_edges
        prev_edges = edges_now

        # Compute all signals and their MMD² (signal detectors are separate from mgr's detector)
        row = {
            "epoch": epoch_idx,
            "recall": recall,
            "n_conjugate_edges": edges_now,
            "edges_added": edges_added,
            "repair_happened": ep["repair_stats"] is not None,
        }

        for sig in scalar_signals:
            fn = _SIGNAL_BATCH_FN[sig]
            vals = fn(result_vecs, queries)
            row[f"{sig}_mean"] = float(np.mean(vals))

            if sig == "eh":
                # EH detector was already updated and checked inside mgr.process_epoch;
                # read results directly from ep to avoid a double check_drift() call.
                row["eh_mmd_sq"] = ep["mmd_squared"]
                row["eh_drift_detected"] = ep["drift_detected"]
            else:
                signal_detectors[sig].update_batch(vals, cell_ids)
                drift_res = signal_detectors[sig].check_drift()
                row[f"{sig}_mmd_sq"] = drift_res["mmd_squared"] if drift_res is not None else None
                row[f"{sig}_drift_detected"] = drift_res["drift_detected"] if drift_res is not None else False

        # Joint 2D detector update
        if joint_detector is not None:
            joint_feats = compute_joint_batch(result_vecs, queries)
            joint_detector.update_batch(joint_feats, cell_ids)
            joint_res = joint_detector.check_drift()
            row["joint_mmd_sq"] = joint_res["mmd_squared"] if joint_res is not None else None
            row["joint_drift_detected"] = joint_res["drift_detected"] if joint_res is not None else False

        log.info(
            f"epoch {epoch_idx:2d}  recall={recall:.4f}  "
            f"eh_mmd={row.get('eh_mmd_sq', 0):.4f}  "
            f"qrd_mmd={row.get('qrd_mmd_sq', 0):.4f}  "
            f"joint_mmd={row.get('joint_mmd_sq', 0):.4f}  "
            f"edges={edges_now}  repair={ep['repair_stats'] is not None}"
        )
        rows.append(row)

    out_csv = str(results_dir / "signals.csv")
    _save_results(rows, out_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
