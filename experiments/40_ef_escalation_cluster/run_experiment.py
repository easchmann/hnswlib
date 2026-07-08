"""Experiment 39: per-query ef escalation on DEEP-96 cluster drift.

Conditions:
  static          — plain HNSW, ef_values = [32, 64, 128]
  adaptive_eh     — conjugate M_conj=48 (replicates exp30 conjugate condition)
  ef_escalation_2x / ef_escalation_4x — EfEscalationAdapter; hard queries
                    (EH > p75 calibration threshold) re-run at ef_base * factor
  oracle_ef128    — all queries run at ef=128 regardless of EH (upper bound)
"""

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
from src.drift.cluster_drift import load_cluster_drift_dataset
from src.drift.conjugate_graph import ConjugateGraph
from src.drift.detector import (
    DriftDetector,
    assign_cell,
    build_spatial_index,
    compute_eh_batch,
)
from src.drift.ef_escalation_adapter import EfEscalationAdapter


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _compute_recall(ids, gt, k):
    hits = [len(set(ids[i].tolist()) & set(gt[i, :k].tolist())) / k for i in range(len(ids))]
    return float(np.mean(hits))


def _build_calibration(dataset, index_path, cfg):
    """Load a fresh index, build spatial index, collect calibration EH data."""
    base = dataset["base"]
    n_calib = cfg["eh_n_calibration_epochs"]
    primary_ef = cfg["primary_ef"]
    k = cfg["k"]

    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.load_index(index_path, max_elements=base.shape[0])

    print("  Building spatial index...")
    centroids, cell_labels = build_spatial_index(
        base, n_cells=cfg["eh_n_cells"], seed=42
    )

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

    return index, base, centroids, cell_labels, np.array(calib_eh), np.array(calib_cids, dtype=np.int32)


def run_static(dataset, index_path, cfg):
    k = cfg["k"]
    ef_values = cfg["ef_values"]
    base = dataset["base"]

    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.load_index(index_path, max_elements=base.shape[0])

    rows = []
    for epoch, (queries, gt) in enumerate(zip(dataset["epochs"], dataset["groundtruth"])):
        for ef in ef_values:
            index.set_ef(ef)
            ids, _ = index.knn_query(queries, k=k)
            rows.append({
                "condition": "static",
                "epoch": epoch,
                "ef": ef,
                "recall": _compute_recall(ids, gt, k),
                "edge_count": 0,
                "n_escalated": 0,
            })
        primary_recall = next(r["recall"] for r in reversed(rows) if r["ef"] == cfg["primary_ef"])
        print(f"  epoch {epoch:2d}  recall@{cfg['primary_ef']}={primary_recall:.4f}")
    return rows


def run_adaptive_eh(dataset, index_path, cfg):
    k = cfg["k"]
    ef_values = cfg["ef_values"]
    primary_ef = cfg["primary_ef"]

    index, base, centroids, cell_labels, calib_eh, calib_cids = _build_calibration(
        dataset, index_path, cfg
    )

    detector = DriftDetector(
        window_size=cfg["eh_window_size"],
        n_cells=cfg["eh_n_cells"],
        n_rff=cfg["eh_n_rff"],
    )
    detector.hot_cell_lambda = cfg["eh_hot_cell_lambda"]
    detector.calibrate(calib_eh, calib_cids)

    cg = ConjugateGraph(M_conj=cfg["eh_M_conj"], max_total_edges=cfg["eh_max_total_edges"])
    mgr_cfg = {
        "M_candidates":          cfg["eh_M_candidates"],
        "repair_ef_search":      cfg["eh_ef_repair"],
        "max_repair_nodes":      cfg["eh_max_repair_nodes"],
        "eh_threshold_percentile": cfg["eh_threshold_percentile"],
        "rng_relaxation":        cfg["eh_rng_relaxation"],
        "n_epochs":              len(dataset["epochs"]),
        "use_diversity":         False,
        "two_hop":               cfg["eh_two_hop"],
        "repair_cooldown":       cfg["eh_cooldown"],
        "primary_ef":            primary_ef,
    }
    mgr = AdaptationManager(
        index=index, base=base, conjugate_graph=cg,
        detector=detector, cell_labels=cell_labels,
        centroids=centroids, config=mgr_cfg,
    )

    rows = []
    for epoch, (queries, gt) in enumerate(zip(dataset["epochs"], dataset["groundtruth"])):
        ef_results = {}
        for ef in ef_values:
            ids, dists = mgr.search_enhanced(queries, k=k, ef_search=ef)
            ef_results[ef] = (ids, dists)

        primary_ids, primary_dists = ef_results[primary_ef]
        ep = mgr.process_epoch(queries, primary_ids, primary_dists, k)

        print(
            f"  epoch {epoch:2d}  recall@{primary_ef}="
            f"{_compute_recall(primary_ids, gt, k):.4f}  "
            f"drift={ep['drift_detected']}  edges={cg._total_edges}"
        )

        for ef in ef_values:
            ids, _ = ef_results[ef]
            rows.append({
                "condition": "adaptive_eh",
                "epoch": epoch,
                "ef": ef,
                "recall": _compute_recall(ids, gt, k),
                "edge_count": cg._total_edges,
                "n_escalated": 0,
            })
    return rows


def run_ef_escalation(factor, dataset, index_path, cfg):
    """EfEscalationAdapter: hard queries re-run at ef_base * factor."""
    k = cfg["k"]
    ef_values = cfg["ef_values"]
    primary_ef = cfg["primary_ef"]
    condition_name = f"ef_escalation_{factor}x"

    index, base, centroids, cell_labels, calib_eh, calib_cids = _build_calibration(
        dataset, index_path, cfg
    )

    detector = DriftDetector(
        window_size=cfg["eh_window_size"],
        n_cells=cfg["eh_n_cells"],
        n_rff=cfg["eh_n_rff"],
    )
    detector.hot_cell_lambda = cfg["eh_hot_cell_lambda"]

    adapter_cfg = {
        "eh_threshold_percentile": cfg["eh_threshold_percentile"],
        "escalation_factor":       factor,
    }
    adapter = EfEscalationAdapter(
        index=index, base=base, detector=detector,
        centroids=centroids, config=adapter_cfg,
    )
    adapter.calibrate(calib_eh, calib_cids)

    rows = []
    for epoch, (queries, gt) in enumerate(zip(dataset["epochs"], dataset["groundtruth"])):
        epoch_rows = []
        for ef in ef_values:
            all_ids, all_dists, mean_eh, n_esc = adapter.search(queries, k, ef)
            epoch_rows.append((ef, all_ids, mean_eh, n_esc, _compute_recall(all_ids, gt, k)))

        # update detector state once (using primary ef results)
        primary_row = next(r for r in epoch_rows if r[0] == primary_ef)
        _, primary_ids, mean_eh, _, _ = primary_row
        cell_ids = assign_cell(queries, centroids)
        drift_result = adapter.process_epoch(queries, primary_ids, mean_eh, cell_ids)

        primary_recall = primary_row[4]
        primary_n_esc = primary_row[3]
        print(
            f"  epoch {epoch:2d}  recall@{primary_ef}={primary_recall:.4f}  "
            f"n_escalated={primary_n_esc}  drift={drift_result is not None and drift_result.get('drift_detected', False)}"
        )

        for ef, all_ids, mean_eh, n_esc, recall in epoch_rows:
            rows.append({
                "condition":   condition_name,
                "epoch":       epoch,
                "ef":          ef,
                "recall":      recall,
                "edge_count":  0,
                "n_escalated": n_esc,
            })
    return rows


def run_oracle_ef128(dataset, index_path, cfg):
    """All queries run at ef=oracle_ef, regardless of EH."""
    k = cfg["k"]
    ef_oracle = cfg["ef_oracle"]
    base = dataset["base"]

    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.load_index(index_path, max_elements=base.shape[0])
    index.set_ef(ef_oracle)

    rows = []
    for epoch, (queries, gt) in enumerate(zip(dataset["epochs"], dataset["groundtruth"])):
        ids, _ = index.knn_query(queries, k=k)
        recall = _compute_recall(ids, gt, k)
        rows.append({
            "condition":   "oracle_ef128",
            "epoch":       epoch,
            "ef":          ef_oracle,
            "recall":      recall,
            "edge_count":  0,
            "n_escalated": len(queries),
        })
        print(f"  epoch {epoch:2d}  recall@{ef_oracle}={recall:.4f}")
    return rows


def main():
    config_path = (
        sys.argv[1] if len(sys.argv) > 1
        else Path(__file__).parent / "config.yaml"
    )
    cfg = _load_config(config_path)
    index_path = str(ROOT / cfg["index_path"])

    for schedule, ds_key, results_key in [
        ("gradual", "dataset_grad",   "results_gradual"),
        ("sudden",  "dataset_sudden", "results_sudden"),
    ]:
        dataset = load_cluster_drift_dataset(str(ROOT / cfg[ds_key]))
        all_rows = []

        print(f"\n{'='*60}")
        print(f"SCHEDULE: {schedule.upper()}")

        print("\n--- static ---")
        all_rows.extend(run_static(dataset, index_path, cfg))

        print("\n--- adaptive_eh (conjugate) ---")
        all_rows.extend(run_adaptive_eh(dataset, index_path, cfg))

        for factor in cfg["ef_escalation_factors"]:
            print(f"\n--- ef_escalation_{factor}x ---")
            all_rows.extend(run_ef_escalation(factor, dataset, index_path, cfg))

        print("\n--- oracle_ef128 ---")
        all_rows.extend(run_oracle_ef128(dataset, index_path, cfg))

        out = ROOT / cfg[results_key]
        os.makedirs(out.parent, exist_ok=True)
        pd.DataFrame(all_rows).to_csv(out, index=False)
        print(f"\nSaved {len(all_rows)} rows → {out}")


if __name__ == "__main__":
    main()
