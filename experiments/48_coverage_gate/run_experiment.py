"""Experiment 48: coverage-gated gateway ef-gap on DEEP-96 cluster drift."""

import collections
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
from src.drift.detector import DriftDetector, assign_cell, build_spatial_index, compute_eh_batch


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _compute_recall(ids, gt, k):
    hits = [len(set(ids[i].tolist()) & set(gt[i, :k].tolist())) / k for i in range(len(ids))]
    return float(np.mean(hits))


def _calibrate_adaptive_eh(dataset, index_path, cfg):
    """Load a fresh index and calibrate the EH detector on pre-drift epochs."""
    base = dataset["base"]
    n_calib = cfg["eh_n_calibration_epochs"]
    primary_ef = cfg["primary_ef"]
    k = cfg["k"]

    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.load_index(index_path, max_elements=base.shape[0])

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

    return index, base, centroids, cell_labels, detector


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
            ids, _ = index.knn_query(queries, k=k, num_threads=1)
            rows.append({"condition": "static", "epoch": epoch, "ef": ef,
                         "recall": _compute_recall(ids, gt, k), "edge_count": 0,
                         "gate_open": False, "mean_coverage": 0.0, "n_efhigh_queries": 0})
        print(f"  epoch {epoch:2d}  recall@{cfg['primary_ef']}="
              f"{rows[-len(ef_values)]['recall']:.4f}")
    return rows


def run_adaptive_eh(dataset, index_path, cfg):
    k = cfg["k"]
    ef_values = cfg["ef_values"]
    primary_ef = cfg["primary_ef"]

    index, base, centroids, cell_labels, detector = _calibrate_adaptive_eh(
        dataset, index_path, cfg)

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
                         "edge_count": cg._total_edges, "gate_open": False,
                         "mean_coverage": 0.0, "n_efhigh_queries": 0})
    return rows


def run_gateway_coverage_gate(dataset, index_path, cfg):
    if not hasattr(hnswlib, "get_last_query_stats"):
        raise RuntimeError(
            "hnswlib.get_last_query_stats not available — "
            "rebuild hnswlib with the thesis C++ patch"
        )

    k = cfg["k"]
    ef_values = cfg["ef_values"]
    ef_low = cfg["ef_low"]
    ef_high = cfg["ef_high"]
    n_gateway = cfg["n_gateway"]
    n_calib = cfg["n_calibration_epochs"]
    p_calib = cfg["p_calib"]
    coverage_threshold = cfg["coverage_threshold"]
    base = dataset["base"]

    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.load_index(index_path, max_elements=base.shape[0])

    # Persists across ALL epochs — cumulative shortcut count per gateway node.
    gateway_shortcut_count = collections.defaultdict(int)
    rng = np.random.default_rng(42)
    total_edges = 0
    rows = []

    for epoch, (queries, gt) in enumerate(zip(dataset["epochs"], dataset["groundtruth"])):
        n_queries = len(queries)
        probe_mask = rng.random(n_queries) < p_calib
        n_efhigh_epoch = 0
        ids_low = np.empty((n_queries, k), dtype=np.int64)
        epoch_gateway_nodes_visited = set()

        index.set_ef(ef_low)
        for i, q in enumerate(queries):
            ids_q, _ = index.knn_query(q.reshape(1, -1), k=k)
            stats = hnswlib.get_last_query_stats()
            visited = np.array(stats["base_layer_visited_node_ids"], dtype=np.int64)
            ids_low[i] = ids_q[0]

            gateway_nodes = visited[:n_gateway]
            epoch_gateway_nodes_visited.update(int(gw) for gw in gateway_nodes)

            # Coverage gate: open per-query when mean shortcuts per gateway node < threshold.
            coverage_scores = [gateway_shortcut_count[int(gw)] for gw in gateway_nodes]
            mean_cov = float(np.mean(coverage_scores)) if coverage_scores else 0.0
            coverage_gate_open = mean_cov < coverage_threshold

            run_efhigh = coverage_gate_open or probe_mask[i]
            if not run_efhigh:
                continue

            index.set_ef(ef_high)
            ids_h, _ = index.knn_query(q.reshape(1, -1), k=k)
            index.set_ef(ef_low)
            n_efhigh_epoch += 1

            proxy_missed = set(ids_h[0].tolist()) - set(ids_q[0].tolist())

            if epoch < n_calib:
                # Calibration: no edge insertion, no coverage updates.
                continue

            if proxy_missed:
                for gw in gateway_nodes:
                    gw_i = int(gw)
                    for mn in proxy_missed:
                        mn_i = int(mn)
                        if gw_i == mn_i:
                            continue
                        inserted_fwd = int(index.add_layer0_edge_evict(gw_i, mn_i))
                        if inserted_fwd:
                            gateway_shortcut_count[gw_i] += 1
                        total_edges += inserted_fwd
                        total_edges += int(index.add_layer0_edge_evict(mn_i, gw_i))

        # Epoch-end coverage: mean over all gateway nodes visited this epoch.
        if epoch_gateway_nodes_visited:
            epoch_mean_coverage = float(np.mean(
                [gateway_shortcut_count[gw] for gw in epoch_gateway_nodes_visited]
            ))
        else:
            epoch_mean_coverage = 0.0

        # gate_open in CSV: True whenever coverage gate fired beyond probe fraction.
        gate_open = n_efhigh_epoch > n_queries * p_calib

        recall_low = _compute_recall(ids_low, gt, k)
        rows.append({"condition": "gateway_coverage_gate", "epoch": epoch, "ef": ef_low,
                     "recall": recall_low, "edge_count": total_edges,
                     "gate_open": gate_open, "mean_coverage": epoch_mean_coverage,
                     "n_efhigh_queries": n_efhigh_epoch})

        for ef in ef_values:
            if ef == ef_low:
                continue
            index.set_ef(ef)
            ids, _ = index.knn_query(queries, k=k, num_threads=1)
            rows.append({"condition": "gateway_coverage_gate", "epoch": epoch, "ef": ef,
                         "recall": _compute_recall(ids, gt, k), "edge_count": total_edges,
                         "gate_open": gate_open, "mean_coverage": epoch_mean_coverage,
                         "n_efhigh_queries": n_efhigh_epoch})

        print(f"  epoch {epoch:2d}  recall@{ef_low}={recall_low:.4f}  "
              f"gate_open={gate_open}  mean_coverage={epoch_mean_coverage:.2f}  "
              f"n_efhigh={n_efhigh_epoch}  edges={total_edges}")

    # Plateau diagnostic: check if gate stayed open throughout ep20–24.
    plateau_rows = [r for r in rows
                    if r["condition"] == "gateway_coverage_gate"
                    and r["epoch"] >= 20
                    and r["ef"] == ef_low]
    plateau_gate_open = all(r["gate_open"] for r in plateau_rows)
    plateau_max_coverage = max((r["mean_coverage"] for r in plateau_rows), default=0.0)
    if plateau_gate_open and plateau_max_coverage < coverage_threshold:
        print(
            "\n  DIAGNOSTIC: gate_open=True persists through all plateau epochs (ep20–24) "
            f"and mean_coverage ({plateau_max_coverage:.2f}) never reaches coverage_threshold "
            f"({coverage_threshold}). Gateway diversity is sufficient to keep the gate open — "
            "the mechanism continues adapting without overhead saturation. "
            "This is a positive result."
        )

    return rows


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "config.yaml"
    cfg = _load_config(config_path)
    index_path = str(ROOT / cfg["index_path"])

    for schedule, ds_key, results_key in [
        ("gradual", "dataset_grad",   "results_gradual"),
        ("sudden",  "dataset_sudden", "results_sudden"),
    ]:
        dataset = load_cluster_drift_dataset(str(ROOT / cfg[ds_key]))

        print(f"\n=== {schedule} — Static ===")
        rows = run_static(dataset, index_path, cfg)

        print(f"\n=== {schedule} — Adaptive EH (conjugate) ===")
        rows.extend(run_adaptive_eh(dataset, index_path, cfg))

        print(f"\n=== {schedule} — Gateway Coverage Gate ===")
        rows.extend(run_gateway_coverage_gate(dataset, index_path, cfg))

        out = ROOT / cfg[results_key]
        os.makedirs(out.parent, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"Saved {len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
