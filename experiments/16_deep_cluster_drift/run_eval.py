"""Experiment 16: DEEP-96 cluster-reweighted drift — static vs periodic rebuild vs adaptive."""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import faiss
import hnswlib

from src.drift.adapter import AdaptationManager
from src.drift.cluster_drift import load_cluster_drift_dataset
from src.drift.naf_adapter import NAFAdapter
from src.drift.conjugate_graph import ConjugateGraph
from src.drift.detector import (
    DriftDetector,
    assign_cell,
    build_spatial_index,
    compute_eh_batch,
)
from node_promoter import NodePromoter


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _compute_recall_stats(result_ids, groundtruth, k):
    """Returns (mean_recall, std_recall) across queries."""
    hits = []
    for i in range(len(result_ids)):
        true_set = set(groundtruth[i, :k].tolist())
        hits.append(len(set(result_ids[i].tolist()) & true_set) / k)
    arr = np.array(hits)
    return float(arr.mean()), float(arr.std())


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
    print(f"  Saved {len(rows)} rows to {out_path}")


def run_static(dataset, index_path, base, config):
    """Plain HNSW search — no adaptation."""
    k = config["eval"]["recall_k"]
    ef_values = config["eval"]["ef_search_values"]
    primary_ef = config["eval"]["primary_ef"]

    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.load_index(index_path, max_elements=base.shape[0])

    results = []
    for epoch_idx, (queries, gt) in enumerate(
            zip(dataset["epochs"], dataset["groundtruth"])):

        for ef in ef_values:
            index.set_ef(ef)
            t0 = time.perf_counter()
            ids, _ = index.knn_query(queries, k=k)
            elapsed_ms = (time.perf_counter() - t0) * 1000 / len(queries)
            recall, recall_std = _compute_recall_stats(ids, gt, k)

            results.append({
                "epoch_idx": epoch_idx,
                "condition": "static",
                "recall_at_k": recall,
                "per_query_recall_std": recall_std,
                "mean_latency_ms": elapsed_ms,
                "queries_per_second": 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0,
                "ef_search": ef,
                "n_conjugate_edges": 0,
                "repair_happened": False,
                "n_index_rebuilds": 0,
                "drift_detected": False,
                "mmd_squared": 0.0,
                "mean_eh": 0.0,
                "n_promoted_nodes": 0,
            })

        print(
            f"  epoch {epoch_idx:2d}  recall@{primary_ef}="
            f"{results[-len(ef_values)]['recall_at_k']:.4f}"
        )

    return results


def run_periodic_rebuild(dataset, index_path, base, config, condition):
    """Full index rebuild from base vectors every rebuild_interval epochs."""
    k = config["eval"]["recall_k"]
    ef_values = config["eval"]["ef_search_values"]
    rebuild_interval = condition["rebuild_interval"]
    idx_cfg = config["index"]
    primary_ef = config["eval"]["primary_ef"]

    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.load_index(index_path, max_elements=base.shape[0])

    n_rebuilds = 0
    results = []
    for epoch_idx, (queries, gt) in enumerate(
            zip(dataset["epochs"], dataset["groundtruth"])):

        if epoch_idx > 0 and epoch_idx % rebuild_interval == 0:
            print(f"  epoch {epoch_idx}: rebuilding index from scratch...")
            index = hnswlib.Index(space="l2", dim=base.shape[1])
            index.init_index(
                max_elements=base.shape[0],
                ef_construction=idx_cfg["ef_construction"],
                M=idx_cfg["M"],
                random_seed=idx_cfg["seed"],
            )
            index.add_items(base, num_threads=-1)
            n_rebuilds += 1

        primary_recall = None
        for ef in ef_values:
            index.set_ef(ef)
            t0 = time.perf_counter()
            ids, _ = index.knn_query(queries, k=k)
            elapsed_ms = (time.perf_counter() - t0) * 1000 / len(queries)
            recall, recall_std = _compute_recall_stats(ids, gt, k)
            if ef == primary_ef:
                primary_recall = recall

            results.append({
                "epoch_idx": epoch_idx,
                "condition": "periodic_rebuild",
                "recall_at_k": recall,
                "per_query_recall_std": recall_std,
                "mean_latency_ms": elapsed_ms,
                "queries_per_second": 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0,
                "ef_search": ef,
                "n_conjugate_edges": 0,
                "repair_happened": False,
                "n_index_rebuilds": n_rebuilds,
                "drift_detected": False,
                "mmd_squared": 0.0,
                "mean_eh": 0.0,
                "n_promoted_nodes": 0,
            })

        print(
            f"  epoch {epoch_idx:2d}  recall@{primary_ef}={primary_recall:.4f}  "
            f"rebuilds={n_rebuilds}"
        )

    return results


def run_adaptive(dataset, index_path, base, config, condition):
    """Conjugate graph adaptation with drift detection."""
    adapt_cfg = config["adaptation"]
    k = config["eval"]["recall_k"]
    primary_ef = config["eval"]["primary_ef"]
    ef_values = config["eval"]["ef_search_values"]
    n_calib = adapt_cfg.get("n_calibration_epochs", 5)

    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.load_index(index_path, max_elements=base.shape[0])

    print("  Building spatial index...")
    centroids, cell_labels = build_spatial_index(
        base, n_cells=adapt_cfg["n_cells"], seed=42
    )

    print(f"  Calibrating detector on first {n_calib} epochs...")
    calib_eh = []
    calib_cell_ids = []
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

    # Per-condition overrides take precedence over the shared adaptation block.
    def _get(key, default=None):
        return condition.get(key, adapt_cfg.get(key, default))

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
        "upper_layer_rewire": _get("upper_layer_rewire", False),
        "rewire_alpha": _get("rewire_alpha", adapt_cfg.get("rewire_alpha", 1.3)),
        "rewire_max_layer": _get("rewire_max_layer", adapt_cfg.get("rewire_max_layer", 2)),
        "rewire_max_queries": _get("rewire_max_queries", adapt_cfg.get("rewire_max_queries", 100)),
        "rewire_eh_percentile": _get("rewire_eh_percentile", adapt_cfg.get("rewire_eh_percentile", 50.0)),
        "entry_point_adaptation": _get("entry_point_adaptation", False),
        "primary_ef": config["eval"]["primary_ef"],
    }

    cg = ConjugateGraph(
        M_conj=condition["M_conj"],
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

    use_promotion = condition.get("node_promotion", False)
    promoter = None
    if use_promotion:
        promoter = NodePromoter(
            index=index,
            base=base,
            centroids=centroids,
            cell_labels=cell_labels,
            config=condition,
        )
        print(f"  NodePromoter enabled: n_promote={condition.get('n_promote', 200)}, "
              f"promote_layer={condition.get('promote_layer', 1)}")

    results = []
    for epoch_idx, (queries, gt) in enumerate(
            zip(dataset["epochs"], dataset["groundtruth"])):

        ef_results = {}
        for ef in ef_values:
            t0 = time.perf_counter()
            ids, dists = mgr.search_enhanced(queries, k=k, ef_search=ef)
            elapsed_ms = (time.perf_counter() - t0) * 1000 / len(queries)
            recall, recall_std = _compute_recall_stats(ids, gt, k)
            ef_results[ef] = {
                "ids": ids,
                "dists": dists,
                "recall": recall,
                "recall_std": recall_std,
                "elapsed_ms": elapsed_ms,
            }

        primary = ef_results[primary_ef]
        ep = mgr.process_epoch(queries, primary["ids"], primary["dists"], k)

        # Node promotion: elevate hot-cell representative nodes into upper layers
        if promoter is not None and ep["drift_detected"] and ep["hot_cells"]:
            candidates = promoter.select_promotion_candidates(ep["hot_cells"])
            n_this = promoter.promote(candidates)
            if n_this > 0:
                print(f"  epoch {epoch_idx:2d}: promoted {n_this} nodes "
                      f"(total={promoter.n_promoted})")

        rew = ep.get("upper_rewire_stats") or {}
        print(
            f"  epoch {epoch_idx:2d}  recall@{primary_ef}={primary['recall']:.4f}  "
            f"drift={ep['drift_detected']}  edges={cg._total_edges}"
            + (f"  rewire+{rew.get('edges_added', 0)}" if rew.get("edges_added") else "")
            + (f"  promoted={promoter.n_promoted}" if promoter else "")
        )

        for ef in ef_values:
            r = ef_results[ef]
            results.append({
                "epoch_idx": epoch_idx,
                "condition": condition["name"],
                "recall_at_k": r["recall"],
                "per_query_recall_std": r["recall_std"],
                "mean_latency_ms": r["elapsed_ms"],
                "queries_per_second": 1000.0 / r["elapsed_ms"] if r["elapsed_ms"] > 0 else 0.0,
                "ef_search": ef,
                "drift_detected": ep["drift_detected"],
                "mmd_squared": ep["mmd_squared"],
                "n_conjugate_edges": cg._total_edges,
                "n_upper_rewire_edges": rew.get("edges_added", 0),
                "repair_happened": ep["repair_stats"] is not None,
                "n_index_rebuilds": 0,
                "mean_eh": ep["mean_eh_this_epoch"],
                "n_promoted_nodes": promoter.n_promoted if promoter else 0,
            })

    return results


def _build_faiss_index(base):
    """Build an exact L2 FAISS index over base vectors."""
    fi = faiss.IndexFlatL2(base.shape[1])
    fi.add(base.astype(np.float32))
    return fi


def run_naf(dataset, index_path, base, config, faiss_index=None, condition_name="adaptive_naf"):
    """NAF-guided conjugate graph adaptation — no drift detector required."""
    k = config["eval"]["recall_k"]
    primary_ef = config["eval"]["primary_ef"]
    ef_values = config["eval"]["ef_search_values"]

    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.load_index(index_path, max_elements=base.shape[0])

    prefix = "naf_exact" if faiss_index is not None else "naf"
    adapter = NAFAdapter(
        index=index,
        base=base,
        hot_k=config[f"{prefix}_hot_k"],
        ef_repair=config[f"{prefix}_ef_repair"],
        n_neighbors=config[f"{prefix}_n_neighbors"],
        M_conj=config[f"{prefix}_M_conj"],
        faiss_index=faiss_index,
    )

    results = []
    for epoch_idx, (queries, gt) in enumerate(
            zip(dataset["epochs"], dataset["groundtruth"])):

        epoch_rows = []
        for ef in ef_values:
            t0 = time.perf_counter()
            ids_list = []
            for q in queries:
                ret = adapter.search_enhanced(q[np.newaxis].astype(np.float32), k, ef)
                ids_list.append(ret[0])
            ids = np.array(ids_list)
            elapsed_ms = (time.perf_counter() - t0) * 1000 / len(queries)
            recall, recall_std = _compute_recall_stats(ids, gt, k)
            epoch_rows.append({
                "epoch_idx": epoch_idx,
                "condition": condition_name,
                "recall_at_k": recall,
                "per_query_recall_std": recall_std,
                "mean_latency_ms": elapsed_ms,
                "queries_per_second": 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0,
                "ef_search": ef,
                "n_conjugate_edges": None,
                "repair_happened": None,
                "drift_detected": False,
                "mmd_squared": 0.0,
                "n_index_rebuilds": 0,
                "mean_eh": 0.0,
                "n_promoted_nodes": 0,
                "n_upper_rewire_edges": 0,
            })

        edges_added = adapter.repair()
        ec = adapter.edge_count()
        for row in epoch_rows:
            row["n_conjugate_edges"] = ec
            row["repair_happened"] = edges_added > 0

        primary_recall = next(r["recall_at_k"] for r in epoch_rows if r["ef_search"] == primary_ef)
        print(
            f"  epoch {epoch_idx:2d}  recall@{primary_ef}={primary_recall:.4f}  "
            f"repair_added={edges_added}  edges={ec}"
        )
        results.extend(epoch_rows)

    return results


def run_all(config_path):
    cfg = _load_config(config_path)
    results_dir = ROOT / cfg["output"]["results_dir"]
    index_path = str(ROOT / cfg["output"]["index_path"])

    os.makedirs(results_dir, exist_ok=True)
    with open(results_dir / "_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    schedules = {
        "gradual": str(ROOT / cfg["output"]["dataset_gradual_path"]),
        "sudden": str(ROOT / cfg["output"]["dataset_sudden_path"]),
    }

    for schedule_name, dataset_path in schedules.items():
        print(f"\n{'#'*70}")
        print(f"SCHEDULE: {schedule_name.upper()}")
        print("#" * 70)

        dataset = load_cluster_drift_dataset(dataset_path)
        base = dataset["base"]

        # Build FAISS index once per schedule if any condition needs exact search
        _faiss_index = None
        if any(c["method"] == "naf_exact" for c in cfg["conditions"]):
            print("  Building FAISS exact index...")
            _faiss_index = _build_faiss_index(base)
            print(f"  FAISS index ready ({base.shape[0]:,} vectors)")

        for condition in cfg["conditions"]:
            name = condition["name"]
            method = condition["method"]
            print(f"\n{'='*60}")
            print(f"CONDITION: {name}  ({schedule_name})")
            print("=" * 60)

            if method == "static":
                rows = run_static(dataset, index_path, base, cfg)
            elif method == "periodic_rebuild":
                rows = run_periodic_rebuild(dataset, index_path, base, cfg, condition)
            elif method == "adaptive":
                rows = run_adaptive(dataset, index_path, base, cfg, condition)
            elif method == "naf":
                rows = run_naf(dataset, index_path, base, cfg)
            elif method == "naf_exact":
                rows = run_naf(dataset, index_path, base, cfg,
                               faiss_index=_faiss_index, condition_name=name)
            else:
                raise ValueError(f"Unknown method: {method}")

            out_csv = str(results_dir / f"{schedule_name}_{name}.csv")
            _save_results(rows, out_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.yaml"),
    )
    args = parser.parse_args()
    run_all(args.config)


if __name__ == "__main__":
    main()
