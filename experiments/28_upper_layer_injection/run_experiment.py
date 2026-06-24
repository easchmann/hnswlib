"""Experiment 28: upper-layer edge injection vs conjugate EH vs layer-0 vs static."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hnswlib
from src.drift.adapter import AdaptationManager, find_repair_candidates, compute_candidate_edges
from src.drift.conjugate_graph import ConjugateGraph
from src.drift.dataset import load_drift_dataset
from src.drift.detector import DriftDetector, assign_cell, build_spatial_index, compute_eh_batch
from src.drift.layer0_eh_adapter import Layer0EHAdapter
from src.drift.upper_layer_eh_adapter import UpperLayerEHAdapter
from src.drift.hybrid_l0_ln_adapter import HybridL0LNAdapter


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _compute_recall(ids, gt, k):
    hits = [len(set(ids[i].tolist()) & set(gt[i, :k].tolist())) / k for i in range(len(ids))]
    return float(np.mean(hits))


def run_static(dataset, index_path, cfg):
    k = cfg["k"]
    ef_values = cfg["ef_values"]
    index = hnswlib.Index(space="l2", dim=dataset["base"].shape[1])
    index.load_index(index_path)
    rows = []
    for epoch, (queries, gt) in enumerate(zip(dataset["epochs"], dataset["groundtruth"])):
        for ef in ef_values:
            index.set_ef(ef)
            ids, _ = index.knn_query(queries, k=k, num_threads=1)
            rows.append({"condition": "static", "epoch": epoch, "ef": ef,
                         "recall": _compute_recall(ids, gt, k), "edge_count": 0})
        print(f"  epoch {epoch:2d}  recall@{cfg['primary_ef']}="
              f"{rows[-len(ef_values)]['recall']:.4f}")
    return rows


def run_adaptive_eh(dataset, index_path, cfg):
    k = cfg["k"]
    ef_values = cfg["ef_values"]
    primary_ef = cfg["primary_ef"]
    n_calib = cfg["eh_n_calibration_epochs"]
    base = dataset["base"]

    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.load_index(index_path)

    print("  Building spatial index...")
    centroids, cell_labels = build_spatial_index(base, n_cells=cfg["eh_n_cells"], seed=42)

    print(f"  Calibrating on first {n_calib} epochs...")
    calib_eh, calib_cids = [], []
    index.set_ef(primary_ef)
    for i in range(n_calib):
        queries = dataset["epochs"][i]
        labels, _ = index.knn_query(queries, k=k)
        eh_vals = compute_eh_batch([base[labels[j]] for j in range(len(queries))])
        cids = assign_cell(queries, centroids)
        calib_eh.extend(eh_vals)
        calib_cids.extend(cids.tolist())

    detector = DriftDetector(window_size=cfg["eh_window_size"], n_cells=cfg["eh_n_cells"],
                             n_rff=cfg["eh_n_rff"])
    detector.hot_cell_lambda = cfg["eh_hot_cell_lambda"]
    detector.calibrate(np.array(calib_eh), np.array(calib_cids, dtype=np.int32))

    cg = ConjugateGraph(M_conj=cfg["eh_M_conj"], max_total_edges=cfg["eh_max_total_edges"])
    mgr_cfg = {
        "M_candidates": cfg["eh_M_candidates"],
        "repair_ef_search": cfg["eh_ef_repair"],
        "max_repair_nodes": cfg["eh_max_repair_nodes"],
        "eh_threshold_percentile": cfg["eh_threshold_percentile"],
        "rng_relaxation": cfg["eh_rng_relaxation"],
        "n_epochs": len(dataset["epochs"]),
        "use_diversity": False,
        "two_hop": cfg["eh_two_hop"],
        "repair_cooldown": cfg["eh_cooldown"],
    }
    mgr = AdaptationManager(index=index, base=base, conjugate_graph=cg,
                            detector=detector, cell_labels=cell_labels,
                            centroids=centroids, config=mgr_cfg)

    rows = []
    for epoch, (queries, gt) in enumerate(zip(dataset["epochs"], dataset["groundtruth"])):
        ef_results = {}
        for ef in ef_values:
            ids, dists = mgr.search_enhanced(queries, k=k, ef_search=ef)
            ef_results[ef] = (ids, dists)

        primary_ids, primary_dists = ef_results[primary_ef]
        ep = mgr.process_epoch(queries, primary_ids, primary_dists, k)

        print(f"  epoch {epoch:2d}  recall@{primary_ef}="
              f"{_compute_recall(primary_ids, gt, k):.4f}  "
              f"drift={ep['drift_detected']}  edges={cg._total_edges}")

        for ef in ef_values:
            ids, _ = ef_results[ef]
            rows.append({"condition": "adaptive_eh", "epoch": epoch, "ef": ef,
                         "recall": _compute_recall(ids, gt, k),
                         "edge_count": cg._total_edges})
    return rows


def run_adaptive_layer0_eh(dataset, index_path, cfg):
    k = cfg["k"]
    ef_values = cfg["ef_values"]
    primary_ef = cfg["primary_ef"]
    n_calib = cfg["eh_n_calibration_epochs"]
    base = dataset["base"]

    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.load_index(index_path)

    print("  Building spatial index...")
    centroids, cell_labels = build_spatial_index(base, n_cells=cfg["eh_n_cells"], seed=42)

    print(f"  Calibrating on first {n_calib} epochs...")
    calib_eh, calib_cids = [], []
    index.set_ef(primary_ef)
    for i in range(n_calib):
        queries = dataset["epochs"][i]
        labels, _ = index.knn_query(queries, k=k)
        eh_vals = compute_eh_batch([base[labels[j]] for j in range(len(queries))])
        cids = assign_cell(queries, centroids)
        calib_eh.extend(eh_vals)
        calib_cids.extend(cids.tolist())

    detector = DriftDetector(window_size=cfg["eh_window_size"], n_cells=cfg["eh_n_cells"],
                             n_rff=cfg["eh_n_rff"])
    detector.hot_cell_lambda = cfg["eh_hot_cell_lambda"]
    detector.calibrate(np.array(calib_eh), np.array(calib_cids, dtype=np.int32))

    adapter_cfg = {
        "M_candidates": cfg["l0_M_candidates"],
        "repair_ef_search": cfg["l0_ef_repair"],
        "max_repair_nodes": cfg["l0_max_repair_nodes"],
        "eh_threshold_percentile": cfg["l0_threshold_percentile"],
        "rng_relaxation": cfg["l0_rng_relaxation"],
        "two_hop": cfg["l0_two_hop"],
        "repair_cooldown": cfg["l0_cooldown"],
    }
    adapter = Layer0EHAdapter(index=index, base=base, detector=detector,
                              cell_labels=cell_labels, centroids=centroids,
                              config=adapter_cfg)

    rows = []
    for epoch, (queries, gt) in enumerate(zip(dataset["epochs"], dataset["groundtruth"])):
        ef_results = {}
        for ef in ef_values:
            ids, dists = adapter.search_enhanced(queries, k=k, ef_search=ef)
            ef_results[ef] = (ids, dists)

        primary_ids, primary_dists = ef_results[primary_ef]
        ep = adapter.process_epoch(queries, primary_ids, primary_dists, k)

        print(f"  epoch {epoch:2d}  recall@{primary_ef}="
              f"{_compute_recall(primary_ids, gt, k):.4f}  "
              f"drift={ep['drift_detected']}  edges={adapter.edge_count()}")

        for ef in ef_values:
            ids, _ = ef_results[ef]
            rows.append({"condition": "adaptive_layer0_eh", "epoch": epoch, "ef": ef,
                         "recall": _compute_recall(ids, gt, k),
                         "edge_count": adapter.edge_count()})
    return rows


def run_adaptive_layerN_eh(dataset, index_path, cfg):
    k = cfg["k"]
    ef_values = cfg["ef_values"]
    primary_ef = cfg["primary_ef"]
    n_calib = cfg["eh_n_calibration_epochs"]
    base = dataset["base"]

    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.load_index(index_path)

    print("  Building spatial index...")
    centroids, cell_labels = build_spatial_index(base, n_cells=cfg["eh_n_cells"], seed=42)

    print(f"  Calibrating on first {n_calib} epochs...")
    calib_eh, calib_cids = [], []
    index.set_ef(primary_ef)
    for i in range(n_calib):
        queries = dataset["epochs"][i]
        labels, _ = index.knn_query(queries, k=k)
        eh_vals = compute_eh_batch([base[labels[j]] for j in range(len(queries))])
        cids = assign_cell(queries, centroids)
        calib_eh.extend(eh_vals)
        calib_cids.extend(cids.tolist())

    detector = DriftDetector(window_size=cfg["eh_window_size"], n_cells=cfg["eh_n_cells"],
                             n_rff=cfg["eh_n_rff"])
    detector.hot_cell_lambda = cfg["eh_hot_cell_lambda"]
    detector.calibrate(np.array(calib_eh), np.array(calib_cids, dtype=np.int32))

    adapter_cfg = {
        "M_candidates": cfg["lN_M_candidates"],
        "repair_ef_search": cfg["lN_ef_repair"],
        "max_repair_nodes": cfg["lN_max_repair_nodes"],
        "eh_threshold_percentile": cfg["lN_threshold_percentile"],
        "repair_cooldown": cfg["lN_cooldown"],
        "inject_level": cfg["lN_inject_level"],
    }
    adapter = UpperLayerEHAdapter(index=index, base=base, detector=detector,
                                   cell_labels=cell_labels, centroids=centroids,
                                   config=adapter_cfg)

    rows = []
    for epoch, (queries, gt) in enumerate(zip(dataset["epochs"], dataset["groundtruth"])):
        ef_results = {}
        for ef in ef_values:
            ids, dists = adapter.search_enhanced(queries, k=k, ef_search=ef)
            ef_results[ef] = (ids, dists)

        primary_ids, primary_dists = ef_results[primary_ef]
        ep = adapter.process_epoch(queries, primary_ids, primary_dists, k)

        print(f"  epoch {epoch:2d}  recall@{primary_ef}="
              f"{_compute_recall(primary_ids, gt, k):.4f}  "
              f"drift={ep['drift_detected']}  "
              f"added={ep['edges_added']}  edges={adapter.edge_count()}  "
              f"attempts={ep['attempts']}")

        for ef in ef_values:
            ids, _ = ef_results[ef]
            rows.append({"condition": "adaptive_layerN_eh", "epoch": epoch, "ef": ef,
                         "recall": _compute_recall(ids, gt, k),
                         "edge_count": adapter.edge_count(),
                         "lN_attempts": adapter.attempt_count()})
    return rows


def run_hybrid_l0_lN(dataset, index_path, cfg):
    k = cfg["k"]
    ef_values = cfg["ef_values"]
    primary_ef = cfg["primary_ef"]
    n_calib = cfg["eh_n_calibration_epochs"]
    base = dataset["base"]

    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.load_index(index_path)

    print("  Building spatial index...")
    centroids, cell_labels = build_spatial_index(base, n_cells=cfg["eh_n_cells"], seed=42)

    print(f"  Calibrating on first {n_calib} epochs...")
    calib_eh, calib_cids = [], []
    index.set_ef(primary_ef)
    for i in range(n_calib):
        queries = dataset["epochs"][i]
        labels, _ = index.knn_query(queries, k=k)
        eh_vals = compute_eh_batch([base[labels[j]] for j in range(len(queries))])
        cids = assign_cell(queries, centroids)
        calib_eh.extend(eh_vals)
        calib_cids.extend(cids.tolist())

    detector = DriftDetector(window_size=cfg["eh_window_size"], n_cells=cfg["eh_n_cells"],
                             n_rff=cfg["eh_n_rff"])
    detector.hot_cell_lambda = cfg["eh_hot_cell_lambda"]
    detector.calibrate(np.array(calib_eh), np.array(calib_cids, dtype=np.int32))

    adapter_cfg = {
        "M_candidates": cfg["lN_M_candidates"],
        "repair_ef_search": cfg["lN_ef_repair"],
        "max_repair_nodes": cfg["lN_max_repair_nodes"],
        "eh_threshold_percentile": cfg["lN_threshold_percentile"],
        "repair_cooldown": cfg["lN_cooldown"],
        "inject_level": cfg["lN_inject_level"],
    }
    adapter = HybridL0LNAdapter(index=index, base=base, detector=detector,
                                 cell_labels=cell_labels, centroids=centroids,
                                 config=adapter_cfg)

    rows = []
    for epoch, (queries, gt) in enumerate(zip(dataset["epochs"], dataset["groundtruth"])):
        ef_results = {}
        for ef in ef_values:
            ids, dists = adapter.search_enhanced(queries, k=k, ef_search=ef)
            ef_results[ef] = (ids, dists)

        primary_ids, primary_dists = ef_results[primary_ef]
        ep = adapter.process_epoch(queries, primary_ids, primary_dists, k)

        print(f"  epoch {epoch:2d}  recall@{primary_ef}="
              f"{_compute_recall(primary_ids, gt, k):.4f}  "
              f"drift={ep['drift_detected']}  "
              f"l0={ep['l0_edges_added']}  lN={ep['lN_edges_added']}  "
              f"L0_total={adapter.l0_edge_count()}  LN_total={adapter.lN_edge_count()}  "
              f"lN_attempts={ep['lN_attempts']}")

        for ef in ef_values:
            ids, _ = ef_results[ef]
            rows.append({"condition": "hybrid_l0_lN", "epoch": epoch, "ef": ef,
                         "recall": _compute_recall(ids, gt, k),
                         "edge_count": adapter.lN_edge_count(),
                         "l0_edge_count": adapter.l0_edge_count(),
                         "lN_edge_count": adapter.lN_edge_count()})
    return rows


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "config.yaml"
    cfg = _load_config(config_path)
    index_path = str(ROOT / cfg["index_path"])

    for schedule, ds_key, results_key in [
        ("gradual", "dataset_grad",   "results_gradual"),
        ("sudden",  "dataset_sudden", "results_sudden"),
    ]:
        dataset = load_drift_dataset(str(ROOT / cfg[ds_key]))

        print(f"\n=== {schedule} — Static ===")
        rows = run_static(dataset, index_path, cfg)

        print(f"\n=== {schedule} — Adaptive EH (conjugate) ===")
        rows.extend(run_adaptive_eh(dataset, index_path, cfg))

        print(f"\n=== {schedule} — Adaptive Layer0 EH ===")
        rows.extend(run_adaptive_layer0_eh(dataset, index_path, cfg))

        print(f"\n=== {schedule} — Adaptive LayerN EH ===")
        rows.extend(run_adaptive_layerN_eh(dataset, index_path, cfg))

        print(f"\n=== {schedule} — Hybrid L0+LN EH ===")
        rows.extend(run_hybrid_l0_lN(dataset, index_path, cfg))

        out = ROOT / cfg[results_key]
        os.makedirs(out.parent, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"Saved {len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
