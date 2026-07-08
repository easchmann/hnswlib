"""Experiment 53: static vs adaptive_improved (ungated) vs adaptive_improved_gated
(population EH mean gate) on clustered structural hardness drift (exp52 dataset)."""

import csv
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hnswlib

_spec = importlib.util.spec_from_file_location(
    "improved_adapter",
    ROOT / "src" / "drift" / "improved_adapter.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ImprovedAdaptationManager = _mod.ImprovedAdaptationManager

from src.drift.cluster_drift import load_cluster_drift_dataset
from src.drift.conjugate_graph import ConjugateGraph
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
    base = dataset["base"]

    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.load_index(index_path, max_elements=base.shape[0])

    rows = []
    for epoch_idx, (queries, gt) in enumerate(zip(dataset["epochs"], dataset["groundtruth"])):
        for ef in ef_values:
            index.set_ef(ef)
            ids, _ = index.knn_query(queries, k=k)
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


def _build_mgr_cfg(cfg):
    return {
        "M_candidates":                       cfg["M_candidates"],
        "repair_ef_search":                   cfg["repair_ef_search"],
        "max_repair_nodes":                   cfg["max_repair_nodes"],
        "eh_threshold_percentile":            cfg["eh_threshold_percentile"],
        "rng_relaxation":                     cfg["rng_relaxation"],
        "n_epochs":                           25,
        "use_diversity":                      False,
        "two_hop":                            cfg["two_hop"],
        "repair_cooldown":                    cfg["repair_cooldown"],
        "recalibration_enabled":              False,
        "primary_ef":                         cfg["primary_ef"],
        "k":                                  cfg["k"],
        "mini_batch_size":                    cfg["mini_batch_size"],
        "min_repair_queries":                 cfg["min_repair_queries"],
        "use_visited_expansion":              cfg["use_visited_expansion"],
        "eh_individual_threshold_percentile": cfg["eh_individual_threshold_percentile"],
        "mode_dispatch_sample_size":          cfg["mode_dispatch_sample_size"],
        "mode_dispatch_ef_multiplier":        cfg["mode_dispatch_ef_multiplier"],
        "mode_dispatch_eh_ratio":             cfg["mode_dispatch_eh_ratio"],
        "escalated_ef":                       cfg["escalated_ef"],
    }


def _run_improved(dataset, index_path, cfg, mgr_cfg, condition_name):
    """Shared logic for ungated and gated ImprovedAdaptationManager runs."""
    k = cfg["k"]
    ef_values = cfg["ef_values"]
    primary_ef = cfg["primary_ef"]
    n_calib = cfg["n_calibration_epochs"]
    base = dataset["base"]
    is_gated = mgr_cfg.get("eh_population_gate", False)

    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.load_index(index_path, max_elements=base.shape[0])

    print(f"  [{condition_name}] Building spatial index...")
    centroids, cell_labels = build_spatial_index(base, n_cells=cfg["n_cells"], seed=42)

    detector = DriftDetector(
        window_size=cfg["window_size"],
        n_cells=cfg["n_cells"],
        n_rff=cfg["n_rff"],
    )
    detector.hot_cell_lambda = cfg["hot_cell_lambda"]

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

    print(f"  [{condition_name}] Calibrating on first {n_calib} epochs...")
    calib_eh, calib_cell_ids = mgr.run_calibration_epochs(dataset, n_calib, k, primary_ef)
    mgr.calibrate(calib_eh, calib_cell_ids)

    rows = []
    for epoch_idx, (queries, gt) in enumerate(zip(dataset["epochs"], dataset["groundtruth"])):
        all_ids_by_ef, _, stats = mgr.process_epoch_online(queries, k, ef_values)

        for ef in ef_values:
            recall = _compute_recall(all_ids_by_ef[ef], gt, k)
            row = {
                "condition": condition_name,
                "epoch": epoch_idx,
                "ef": ef,
                "recall": recall,
                "edge_count": cg._total_edges,
            }
            if is_gated:
                row["eh_gate_open"] = stats["eh_population_gate_open"]
                row["mean_eh_window"] = stats["mean_eh_window"]
            rows.append(row)

        primary_recall = _compute_recall(all_ids_by_ef[primary_ef], gt, k)
        gate_info = (f"  gate={'open' if stats['eh_population_gate_open'] else 'closed'}"
                     f"  mean_eh_win={stats['mean_eh_window']:.4f}") if is_gated else ""
        print(
            f"  [{condition_name}] epoch {epoch_idx:2d}  "
            f"recall@{primary_ef}={primary_recall:.4f}"
            f"  edges_added={stats['total_edges_added']}  repairs={stats['repair_count']}"
            f"  mode={stats['mode']}{gate_info}"
        )

    return rows


def run_adaptive_improved_ungated(dataset, index_path, cfg):
    """ImprovedAdaptationManager with no population gate."""
    mgr_cfg = _build_mgr_cfg(cfg)
    mgr_cfg["eh_population_gate"] = False
    return _run_improved(dataset, index_path, cfg, mgr_cfg, "adaptive_improved")


def run_adaptive_improved_gated(dataset, index_path, cfg):
    """ImprovedAdaptationManager with population EH mean gate."""
    mgr_cfg = _build_mgr_cfg(cfg)
    mgr_cfg["eh_population_gate"] = cfg["eh_population_gate"]
    mgr_cfg["eh_gate_window_size"] = cfg["eh_gate_window_size"]
    mgr_cfg["eh_gate_sigma"] = cfg["eh_gate_sigma"]
    return _run_improved(dataset, index_path, cfg, mgr_cfg, "adaptive_improved_gated")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(EXP_DIR / "config.yaml")
    schedule_filter = sys.argv[2] if len(sys.argv) > 2 else None
    cfg = _load_config(config_path)

    base = np.load(os.path.join(cfg["data_dir"], "index_vectors.npy"), mmap_mode='r')

    all_schedules = [
        ("gradual", cfg["dataset_gradual"], cfg["results_gradual"]),
        ("sudden",  cfg["dataset_sudden"],  cfg["results_sudden"]),
    ]
    schedules = (
        [s for s in all_schedules if s[0] == schedule_filter]
        if schedule_filter else all_schedules
    )
    if not schedules:
        print(f"ERROR: unknown schedule {schedule_filter!r}; choose 'gradual' or 'sudden'")
        sys.exit(1)

    for schedule_name, dataset_key, out_path in schedules:
        print(f"\n{'#' * 70}")
        print(f"SCHEDULE: {schedule_name.upper()}")
        print(f"{'#' * 70}")

        dataset = load_cluster_drift_dataset(str(ROOT / dataset_key))
        dataset["base"] = base

        index_path = cfg["index_path"]

        all_rows = []

        print("\n--- static ---")
        all_rows.extend(run_static(dataset, index_path, cfg))

        print("\n--- adaptive_improved (ungated) ---")
        all_rows.extend(run_adaptive_improved_ungated(dataset, index_path, cfg))

        print("\n--- adaptive_improved_gated (population EH gate) ---")
        all_rows.extend(run_adaptive_improved_gated(dataset, index_path, cfg))

        _save_rows(all_rows, str(ROOT / out_path))


if __name__ == "__main__":
    main()
