"""Experiment 35: static vs adaptive_current vs adaptive_improved on SIFT rotation drift."""

import csv
import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hnswlib

from src.drift.adapter import AdaptationManager
from src.drift.improved_adapter import ImprovedAdaptationManager
from src.drift.conjugate_graph import ConjugateGraph
from src.drift.dataset import load_drift_dataset
from src.drift.detector import (
    DriftDetector,
    assign_cell,
    build_spatial_index,
    compute_eh_batch,
)

EXP_DIR = Path(__file__).parent


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _compute_recall(ids, gt, k):
    hits = []
    for i in range(len(ids)):
        true_set = set(gt[i, :k].tolist())
        hits.append(len(set(ids[i].tolist()) & true_set) / k)
    return float(np.mean(hits))


def _save_rows(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = sorted(set(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({fk: row.get(fk) for fk in fieldnames})
    print(f"Saved {len(rows)} rows to {path}")


def run_static(dataset, index_path, cfg):
    """Plain HNSW search with no adaptation."""
    k = cfg["k"]
    ef_values = cfg["ef_values"]

    index = hnswlib.Index(space="l2", dim=dataset["base"].shape[1])
    index.load_index(str(ROOT / index_path))

    rows = []
    for epoch_idx, (queries, gt) in enumerate(zip(dataset["epochs"], dataset["groundtruth"])):
        for ef in ef_values:
            index.set_ef(ef)
            ids, _ = index.knn_query(queries, k=k, num_threads=1)
            recall = _compute_recall(ids, gt, k)
            rows.append({
                "condition": "static",
                "epoch": epoch_idx,
                "ef": ef,
                "recall": recall,
                "edge_count": 0,
            })
        primary_recall = rows[-len(ef_values)]["recall"]
        print(f"  [static] epoch {epoch_idx:2d}  recall@{cfg['primary_ef']}={primary_recall:.4f}")

    return rows


def run_adaptive_current(dataset, index_path, cfg):
    """Existing AdaptationManager (exp08 baseline) with recalibration disabled."""
    k = cfg["k"]
    ef_values = cfg["ef_values"]
    primary_ef = cfg["primary_ef"]
    n_calib = cfg["n_calibration_epochs"]
    base = dataset["base"]

    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.load_index(str(ROOT / index_path))

    print("  [adaptive_current] Building spatial index...")
    centroids, cell_labels = build_spatial_index(base, n_cells=cfg["n_cells"], seed=42)

    print(f"  [adaptive_current] Calibrating on first {n_calib} epochs...")
    calib_eh, calib_cell_ids = [], []
    index.set_ef(primary_ef)
    for i in range(n_calib):
        queries = dataset["epochs"][i]
        labels, _ = index.knn_query(queries, k=k, num_threads=1)
        eh_vals = compute_eh_batch([base[labels[j]] for j in range(len(queries))])
        cids = assign_cell(queries, centroids)
        calib_eh.extend(eh_vals)
        calib_cell_ids.extend(cids.tolist())

    detector = DriftDetector(
        window_size=cfg["window_size"],
        n_cells=cfg["n_cells"],
        n_rff=cfg["n_rff"],
    )
    detector.hot_cell_lambda = cfg["hot_cell_lambda"]
    detector.calibrate(np.array(calib_eh), np.array(calib_cell_ids, dtype=np.int32))

    mgr_cfg = {
        "M_candidates":          cfg["M_candidates"],
        "repair_ef_search":      cfg["repair_ef_search"],
        "max_repair_nodes":      cfg["max_repair_nodes"],
        "eh_threshold_percentile": cfg["eh_threshold_percentile"],
        "rng_relaxation":        cfg["rng_relaxation"],
        "n_epochs":              len(dataset["epochs"]),
        "use_diversity":         False,
        "two_hop":               cfg["two_hop"],
        "repair_cooldown":       cfg["repair_cooldown"],
        "recalibration_enabled": cfg.get("recalibration_enabled", False),
        "primary_ef":            cfg["primary_ef"],
    }

    cg = ConjugateGraph(M_conj=cfg["M_conj"], max_total_edges=cfg["max_total_edges"])
    mgr = AdaptationManager(
        index=index,
        base=base,
        conjugate_graph=cg,
        detector=detector,
        cell_labels=cell_labels,
        centroids=centroids,
        config=mgr_cfg,
    )

    rows = []
    for epoch_idx, (queries, gt) in enumerate(zip(dataset["epochs"], dataset["groundtruth"])):
        ef_results = {}
        for ef in ef_values:
            ids, dists = mgr.search_enhanced(queries, k=k, ef_search=ef)
            recall = _compute_recall(ids, gt, k)
            ef_results[ef] = {"ids": ids, "dists": dists, "recall": recall}

        primary = ef_results[primary_ef]
        ep = mgr.process_epoch(queries, primary["ids"], primary["dists"], k)

        print(
            f"  [adaptive_current] epoch {epoch_idx:2d}  recall@{primary_ef}={primary['recall']:.4f}"
            f"  drift={ep['drift_detected']}  edges={cg._total_edges}"
        )

        for ef in ef_values:
            rows.append({
                "condition": "adaptive_current",
                "epoch": epoch_idx,
                "ef": ef,
                "recall": ef_results[ef]["recall"],
                "edge_count": cg._total_edges,
            })

    return rows


def run_adaptive_improved(dataset, index_path, cfg):
    """ImprovedAdaptationManager with all four fixes."""
    k = cfg["k"]
    ef_values = cfg["ef_values"]
    primary_ef = cfg["primary_ef"]
    n_calib = cfg["n_calibration_epochs"]
    base = dataset["base"]

    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.load_index(str(ROOT / index_path))

    print("  [adaptive_improved] Building spatial index...")
    centroids, cell_labels = build_spatial_index(base, n_cells=cfg["n_cells"], seed=42)

    detector = DriftDetector(
        window_size=cfg["window_size"],
        n_cells=cfg["n_cells"],
        n_rff=cfg["n_rff"],
    )
    detector.hot_cell_lambda = cfg["hot_cell_lambda"]

    mgr_cfg = {
        "M_candidates":                   cfg["M_candidates"],
        "repair_ef_search":               cfg["repair_ef_search"],
        "max_repair_nodes":               cfg["max_repair_nodes"],
        "eh_threshold_percentile":        cfg["eh_threshold_percentile"],
        "rng_relaxation":                 cfg["rng_relaxation"],
        "n_epochs":                       len(dataset["epochs"]),
        "use_diversity":                  False,
        "two_hop":                        cfg["two_hop"],
        "repair_cooldown":                cfg["repair_cooldown"],
        "recalibration_enabled":          False,
        "primary_ef":                     cfg["primary_ef"],
        "k":                              cfg["k"],
        "mini_batch_size":                cfg["mini_batch_size"],
        "min_repair_queries":             cfg["min_repair_queries"],
        "use_visited_expansion":          cfg["use_visited_expansion"],
        "eh_individual_threshold_percentile": cfg["eh_individual_threshold_percentile"],
        "mode_dispatch_sample_size":      cfg["mode_dispatch_sample_size"],
        "mode_dispatch_ef_multiplier":    cfg["mode_dispatch_ef_multiplier"],
        "mode_dispatch_eh_ratio":         cfg["mode_dispatch_eh_ratio"],
        "escalated_ef":                   cfg["escalated_ef"],
    }

    cg = ConjugateGraph(M_conj=cfg["M_conj"], max_total_edges=cfg["max_total_edges"])
    mgr = ImprovedAdaptationManager(
        index=index,
        base=base,
        conjugate_graph=cg,
        detector=detector,
        cell_labels=cell_labels,
        centroids=centroids,
        config=mgr_cfg,
    )

    print(f"  [adaptive_improved] Calibrating on first {n_calib} epochs...")
    calib_eh, calib_cell_ids = mgr.run_calibration_epochs(dataset, n_calib, k, primary_ef)
    mgr.calibrate(calib_eh, calib_cell_ids)

    rows = []
    for epoch_idx, (queries, gt) in enumerate(zip(dataset["epochs"], dataset["groundtruth"])):
        all_ids_by_ef, _, stats = mgr.process_epoch_online(
            queries, k, ef_values
        )

        # Matched recall comparison: adaptive (conjugate-augmented) vs. a fresh raw
        # static search, both restricted to the SAME hard-query indices this epoch.
        # This is the recall-based counterpart to mean_hardonly_static_dist — same
        # matched-population design, but using the thesis's primary metric (recall)
        # instead of raw L2 distance, which is less sensitive to the intrinsic
        # drift-driven distance inflation that swamps the distance-based signal.
        # Ground truth is used here in the evaluation harness only, never inside
        # the adapter itself, preserving the ground-truth-free design of the
        # deployed adaptation mechanism.
        hard_idx = stats.get("hard_query_indices") or []
        if hard_idx:
            hard_idx = np.array(hard_idx, dtype=np.int64)
            recall_adaptive_hardonly = _compute_recall(
                all_ids_by_ef[primary_ef][hard_idx], gt[hard_idx], k
            )
            index.set_ef(primary_ef)
            raw_hard_ids, _ = index.knn_query(queries[hard_idx], k=k, num_threads=1)
            recall_static_hardonly = _compute_recall(raw_hard_ids, gt[hard_idx], k)
        else:
            recall_adaptive_hardonly = None
            recall_static_hardonly = None

        for ef in ef_values:
            recall = _compute_recall(all_ids_by_ef[ef], gt, k)
            rows.append({
                "condition": "adaptive_improved",
                "epoch": epoch_idx,
                "ef": ef,
                "recall": recall,
                "edge_count": cg._total_edges,
                "mean_source_query_dist": stats.get("mean_source_query_dist"),
                "mean_static_topk_dist": stats.get("mean_static_topk_dist"),
                "mean_hardonly_static_dist": stats.get("mean_hardonly_static_dist"),
                "n_hard_queries": len(hard_idx),
                "recall_adaptive_hardonly": recall_adaptive_hardonly,
                "recall_static_hardonly": recall_static_hardonly,
            })

        primary_recall = _compute_recall(all_ids_by_ef[primary_ef], gt, k)
        src_dist = stats.get("mean_source_query_dist")
        src_dist_str = f"{src_dist:.4f}" if src_dist is not None else "n/a"
        hardonly_dist = stats.get("mean_hardonly_static_dist")
        hardonly_dist_str = f"{hardonly_dist:.4f}" if hardonly_dist is not None else "n/a"
        rah_str = f"{recall_adaptive_hardonly:.4f}" if recall_adaptive_hardonly is not None else "n/a"
        rsh_str = f"{recall_static_hardonly:.4f}" if recall_static_hardonly is not None else "n/a"
        print(
            f"  [adaptive_improved] epoch {epoch_idx:2d}  recall@{primary_ef}={primary_recall:.4f}"
            f"  edges_added={stats['total_edges_added']}  repairs={stats['repair_count']}"
            f"  mode={stats['mode']}"
            f"  src_dist={src_dist_str}  static_topk_dist={stats['mean_static_topk_dist']:.4f}"
            f"  hardonly_static_dist={hardonly_dist_str}"
            f"  n_hard={len(hard_idx)}  recall_adaptive_hardonly={rah_str}  recall_static_hardonly={rsh_str}"
        )

    return rows


RUNNERS = {
    "static": run_static,
    "adaptive_current": run_adaptive_current,
    "adaptive_improved": run_adaptive_improved,
}


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(EXP_DIR / "config.yaml")
    cfg = _load_config(config_path)

    # Optional: restrict which conditions run, e.g. to rerun only adaptive_improved
    # after instrumenting it, without redoing the unchanged static/adaptive_current
    # baselines. Defaults to all three for backward compatibility.
    conditions = cfg.get("conditions", list(RUNNERS.keys()))
    unknown = set(conditions) - set(RUNNERS)
    if unknown:
        raise ValueError(f"Unknown condition(s) in config 'conditions': {unknown}")

    schedules = [
        ("gradual", cfg["dataset_gradual"], cfg["results_gradual"]),
        ("sudden",  cfg["dataset_sudden"],  cfg["results_sudden"]),
    ]

    for schedule_name, dataset_key, out_path in schedules:
        print(f"\n{'#'*70}")
        print(f"SCHEDULE: {schedule_name.upper()}")
        print(f"{'#'*70}")

        dataset = load_drift_dataset(str(ROOT / dataset_key))
        index_path = cfg["index_path"]

        all_rows = []
        for condition in conditions:
            print(f"\n--- {condition} ---")
            all_rows.extend(RUNNERS[condition](dataset, index_path, cfg))

        _save_rows(all_rows, str(ROOT / out_path))


if __name__ == "__main__":
    main()
