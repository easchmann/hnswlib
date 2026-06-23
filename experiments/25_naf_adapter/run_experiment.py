"""Experiment 25: NAF adapter vs EH adapter vs static HNSW."""

import json
import os
import sys
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
from src.drift.detector import DriftDetector, assign_cell, build_spatial_index, compute_eh_batch
from src.drift.naf_adapter import NAFAdapter


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
    results = []
    for epoch, (queries, gt) in enumerate(zip(dataset["epochs"], dataset["groundtruth"])):
        for ef in ef_values:
            index.set_ef(ef)
            ids, _ = index.knn_query(queries, k=k, num_threads=1)
            results.append({"epoch": epoch, "condition": "static", "ef": ef,
                            "recall": _compute_recall(ids, gt, k), "edge_count": 0})
        print(f"  epoch {epoch:2d}  recall@{cfg['primary_ef']}="
              f"{results[-len(ef_values)]['recall']:.4f}")
    return results


def run_adaptive_eh(dataset, index_path, cfg):
    k = cfg["k"]
    ef_values = cfg["ef_values"]
    primary_ef = cfg["primary_ef"]
    n_calib = cfg["n_calibration_epochs"]
    base = dataset["base"]

    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.load_index(index_path)

    print("  Building spatial index...")
    centroids, cell_labels = build_spatial_index(base, n_cells=cfg["n_cells"], seed=42)

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

    detector = DriftDetector(window_size=cfg["window_size"], n_cells=cfg["n_cells"], n_rff=cfg["n_rff"])
    detector.hot_cell_lambda = cfg["hot_cell_lambda"]
    detector.calibrate(np.array(calib_eh), np.array(calib_cids, dtype=np.int32))

    cg = ConjugateGraph(M_conj=cfg["M_conj"], max_total_edges=cfg["max_total_edges"])
    mgr_cfg = {
        "M_candidates": cfg["M_candidates"],
        "repair_ef_search": cfg["ef_repair"],
        "max_repair_nodes": cfg["max_repair_nodes"],
        "eh_threshold_percentile": cfg["eh_threshold_percentile"],
        "rng_relaxation": cfg["rng_relaxation"],
        "n_epochs": len(dataset["epochs"]),
        "use_diversity": False,
        "two_hop": cfg["two_hop"],
        "repair_cooldown": cfg["cooldown"],
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
            results.append({"epoch": epoch, "condition": "adaptive_eh", "ef": ef,
                            "recall": _compute_recall(ids, gt, k),
                            "edge_count": cg._total_edges})
    return results


def _build_faiss_index(base):
    """Build an exact L2 FAISS index over base vectors."""
    fi = faiss.IndexFlatL2(base.shape[1])
    fi.add(base.astype(np.float32))
    return fi


def run_adaptive_naf(dataset, index_path, cfg, faiss_index=None, condition_name="adaptive_naf"):
    k = cfg["k"]
    ef_values = cfg["ef_values"]
    primary_ef = cfg["primary_ef"]
    base = dataset["base"]

    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.load_index(index_path)

    prefix = "naf_exact" if faiss_index is not None else "naf"
    adapter = NAFAdapter(
        index=index,
        base=base,
        hot_k=cfg[f"{prefix}_hot_k"],
        ef_repair=cfg[f"{prefix}_ef_repair"],
        n_neighbors=cfg[f"{prefix}_n_neighbors"],
        M_conj=cfg[f"{prefix}_M_conj"],
        faiss_index=faiss_index,
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
            epoch_rows.append({"epoch": epoch, "condition": condition_name, "ef": ef,
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
    config_path = Path(__file__).parent / "config.yaml"
    cfg = _load_config(config_path)

    dataset_path = str(ROOT / cfg["dataset_path"])
    index_path = str(ROOT / cfg["index_path"])
    dataset = load_drift_dataset(dataset_path)

    all_results = []

    print("\n=== Static ===")
    all_results.extend(run_static(dataset, index_path, cfg))

    print("\n=== Adaptive EH ===")
    all_results.extend(run_adaptive_eh(dataset, index_path, cfg))

    print("\n=== Adaptive NAF (HNSW pool) ===")
    all_results.extend(run_adaptive_naf(dataset, index_path, cfg))

    print("\n=== Adaptive NAF (FAISS exact pool) ===")
    print("  Building FAISS index...")
    fi = _build_faiss_index(dataset["base"])
    all_results.extend(run_adaptive_naf(dataset, index_path, cfg,
                                        faiss_index=fi, condition_name="adaptive_naf_exact"))

    out_path = ROOT / cfg["results_path"]
    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} rows to {out_path}")


if __name__ == "__main__":
    main()
