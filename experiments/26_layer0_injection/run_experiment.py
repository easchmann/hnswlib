"""Experiment 26: layer-0 edge injection vs EH adapter vs static HNSW."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hnswlib
from src.drift.adapter import AdaptationManager
from src.drift.conjugate_graph import ConjugateGraph
from src.drift.dataset import load_drift_dataset
from src.drift.detector import DriftDetector, assign_cell, build_spatial_index, compute_eh_batch
from src.drift.layer0_adapter import Layer0Adapter


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _compute_recall(ids, gt, k):
    hits = [len(set(ids[i].tolist()) & set(gt[i, :k].tolist())) / k for i in range(len(ids))]
    return float(np.mean(hits))


def run_static(dataset, schedule, index_path, cfg):
    k = cfg["k"]
    ef_values = cfg["ef_values"]
    index = hnswlib.Index(space="l2", dim=dataset["base"].shape[1])
    index.load_index(index_path)
    results = []
    for epoch, (queries, gt) in enumerate(zip(dataset["epochs"], dataset["groundtruth"])):
        for ef in ef_values:
            index.set_ef(ef)
            ids, _ = index.knn_query(queries, k=k, num_threads=1)
            results.append({"schedule": schedule, "condition": "static", "epoch": epoch,
                            "ef": ef, "recall": _compute_recall(ids, gt, k), "edge_count": 0})
        print(f"  epoch {epoch:2d}  recall@{cfg['primary_ef']}="
              f"{results[-len(ef_values)]['recall']:.4f}")
    return results


def run_adaptive_eh(dataset, schedule, index_path, cfg):
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

    results = []
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
            results.append({"schedule": schedule, "condition": "adaptive_eh", "epoch": epoch,
                            "ef": ef, "recall": _compute_recall(ids, gt, k),
                            "edge_count": cg._total_edges})
    return results


def run_adaptive_layer0(dataset, schedule, index_path, cfg):
    k = cfg["k"]
    ef_values = cfg["ef_values"]
    primary_ef = cfg["primary_ef"]
    base = dataset["base"]

    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.load_index(index_path)

    adapter = Layer0Adapter(
        index=index,
        base=base,
        hot_k=cfg["l0_hot_k"],
        ef_repair=cfg["l0_ef_repair"],
        n_neighbors=cfg["l0_n_neighbors"],
    )

    results = []
    for epoch, (queries, gt) in enumerate(zip(dataset["epochs"], dataset["groundtruth"])):
        epoch_rows = []
        for ef in ef_values:
            ids_list = []
            for q in queries:
                ret = adapter.search_enhanced(q[np.newaxis].astype(np.float32), k, ef)
                ids_list.append(ret[0])
            ids = np.array(ids_list)
            epoch_rows.append({"schedule": schedule, "condition": "adaptive_layer0",
                               "epoch": epoch, "ef": ef,
                               "recall": _compute_recall(ids, gt, k), "edge_count": None})

        edges_added = adapter.repair()
        ec = adapter.edge_count()
        for row in epoch_rows:
            row["edge_count"] = ec

        print(f"  epoch {epoch:2d}  recall@{primary_ef}="
              f"{epoch_rows[0]['recall']:.4f}  "
              f"repair_added={edges_added}  edges={ec}")
        results.extend(epoch_rows)

    return results


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "config.yaml"
    cfg = _load_config(config_path)

    index_path = str(ROOT / cfg["index_path"])
    all_results = []

    for schedule, ds_key in [("gradual", "dataset_grad"), ("sudden", "dataset_sudden")]:
        dataset = load_drift_dataset(str(ROOT / cfg[ds_key]))

        print(f"\n=== {schedule} — Static ===")
        all_results.extend(run_static(dataset, schedule, index_path, cfg))

        print(f"\n=== {schedule} — Adaptive EH ===")
        all_results.extend(run_adaptive_eh(dataset, schedule, index_path, cfg))

        print(f"\n=== {schedule} — Adaptive Layer0 ===")
        all_results.extend(run_adaptive_layer0(dataset, schedule, index_path, cfg))

    out_path = ROOT / cfg["results_path"]
    os.makedirs(out_path.parent, exist_ok=True)
    pd.DataFrame(all_results).to_csv(out_path, index=False)
    print(f"\nSaved {len(all_results)} rows to {out_path}")


if __name__ == "__main__":
    main()
