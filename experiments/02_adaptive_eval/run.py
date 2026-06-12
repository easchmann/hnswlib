"""Experiment 02: compare static HNSW, periodic rebuild, and adaptive repair."""

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

from src.baselines.periodic_rebuild import PeriodicRebuildBaseline
from src.drift.adapter import AdaptationManager, find_repair_candidates
from src.drift.conjugate_graph import ConjugateGraph
from src.drift.dataset import load_drift_dataset
from src.drift.detector import (
    DriftDetector,
    assign_cell,
    build_spatial_index,
    compute_eh_batch,
)
from src.eval.drift_eval import evaluate_epoch


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _compute_recall(result_ids, groundtruth, k):
    hits = []
    for i in range(len(result_ids)):
        true_set = set(groundtruth[i, :k].tolist())
        hits.append(len(set(result_ids[i].tolist()) & true_set) / k)
    return float(np.mean(hits))


def run_static_hnsw(dataset, index, config, scenario):
    """Load experiment-01 results if available; otherwise re-run evaluation."""
    ef_values = config["eval"]["ef_search_values"]
    k = config["eval"]["recall_k"]

    results_01_key = f"results_01_{scenario}"
    csv_path = ROOT / config["output"][results_01_key]

    if csv_path.exists():
        print(f"  Loading static HNSW results from {csv_path}")
        rows = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                ef = int(row["ef_search"])
                if ef not in ef_values:
                    continue
                rows.append({
                    "epoch_idx": int(row["epoch"]),
                    "recall_at_k": float(row["recall"]),
                    "mean_latency_ms": float(row["mean_query_time_ms"]),
                    "queries_per_second": 1000.0 / float(row["mean_query_time_ms"])
                        if float(row["mean_query_time_ms"]) > 0 else 0.0,
                    "ef_search": ef,
                    "method": "static_hnsw",
                    "drift_detected": False,
                    "mmd_squared": None,
                    "n_conjugate_edges": 0,
                    "repair_happened": False,
                    "repair_stats": None,
                })
        if rows:
            return rows
        print("  No matching ef_search values in cached results; re-running.")

    print("  Running static HNSW evaluation...")
    results = []
    for ef in ef_values:
        for epoch_idx, (queries, gt) in enumerate(
                zip(dataset["epochs"], dataset["groundtruth"])):
            row = evaluate_epoch(index, queries, gt, ef_search=ef, k=k)
            results.append({
                "epoch_idx": epoch_idx,
                "recall_at_k": row["recall"],
                "mean_latency_ms": row["mean_query_time_ms"],
                "queries_per_second": 1000.0 / row["mean_query_time_ms"]
                    if row["mean_query_time_ms"] > 0 else 0.0,
                "ef_search": ef,
                "method": "static_hnsw",
                "drift_detected": False,
                "mmd_squared": None,
                "n_conjugate_edges": 0,
                "repair_happened": False,
                "repair_stats": None,
            })
    return results


def run_periodic_rebuild(dataset, base, config):
    """Run PeriodicRebuildBaseline for each ef_search value."""
    ef_values = config["eval"]["ef_search_values"]
    k = config["eval"]["recall_k"]
    interval = config["periodic_rebuild"]["rebuild_interval"]
    idx_cfg = config["index"]

    results = []
    for ef in ef_values:
        baseline = PeriodicRebuildBaseline(
            base=base,
            M=idx_cfg["M"],
            ef_construction=idx_cfg["ef_construction"],
            rebuild_interval=interval,
            seed=idx_cfg["seed"],
        )
        for epoch_idx, (queries, gt) in enumerate(
                zip(dataset["epochs"], dataset["groundtruth"])):
            row = baseline.process_epoch(epoch_idx, queries, gt, k=k, ef_search=ef)
            results.append({
                "epoch_idx": epoch_idx,
                "recall_at_k": row["recall_at_k"],
                "mean_latency_ms": row["mean_latency_ms"],
                "queries_per_second": row["queries_per_second"],
                "ef_search": ef,
                "method": "periodic_rebuild",
                "drift_detected": False,
                "mmd_squared": None,
                "n_conjugate_edges": 0,
                "repair_happened": row["rebuilt_this_epoch"],
                "repair_stats": None,
                "rebuild_time_seconds": row["rebuild_time_seconds"],
            })
        print(
            f"  periodic_rebuild ef={ef}: total rebuild time "
            f"{baseline.total_rebuild_time():.1f}s"
        )
    return results


def run_adaptive(dataset, index, base, config):
    """Run adaptive method: DriftDetector + ConjugateGraph + AdaptationManager."""
    adapt_cfg = config["adaptation"]
    k = config["eval"]["recall_k"]
    primary_ef = config["eval"]["primary_ef"]
    ef_values = config["eval"]["ef_search_values"]
    n_calib = adapt_cfg.get("n_calibration_epochs", 5)

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

    mgr_config = {
        "M_candidates": adapt_cfg["M_candidates"],
        "repair_ef_search": adapt_cfg["ef_repair"],
        "max_repair_nodes": adapt_cfg["max_repair_nodes"],
        "rng_relaxation": adapt_cfg["rng_relaxation"],
        "n_epochs": len(dataset["epochs"]),
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

    # Evaluate at all ef values before adaptation so all share the same
    # conjugate graph snapshot. Adaptation itself uses primary_ef results.
    results = []
    for epoch_idx, (queries, gt) in enumerate(
            zip(dataset["epochs"], dataset["groundtruth"])):

        ef_results = {}
        for ef in ef_values:
            t0 = time.perf_counter()
            ids, dists = mgr.search_enhanced(queries, k=k, ef_search=ef)
            elapsed_ms = (time.perf_counter() - t0) * 1000 / len(queries)
            ef_results[ef] = {
                "ids": ids,
                "dists": dists,
                "recall": _compute_recall(ids, gt, k),
                "elapsed_ms": elapsed_ms,
            }

        # adaptation: detection + repair driven by primary_ef search results
        primary = ef_results[primary_ef]
        ep = mgr.process_epoch(queries, primary["ids"], primary["dists"], k)

        print(
            f"  epoch {epoch_idx:2d}  recall@{primary_ef}={primary['recall']:.4f}  "
            f"drift={ep['drift_detected']}  edges={cg._total_edges}"
        )

        for ef in ef_values:
            r = ef_results[ef]
            results.append({
                "epoch_idx": epoch_idx,
                "recall_at_k": r["recall"],
                "mean_latency_ms": r["elapsed_ms"],
                "queries_per_second": 1000.0 / r["elapsed_ms"] if r["elapsed_ms"] > 0 else 0.0,
                "ef_search": ef,
                "method": "adaptive",
                "drift_detected": ep["drift_detected"],
                "mmd_squared": ep["mmd_squared"],
                "n_conjugate_edges": cg._total_edges,
                "repair_happened": ep["repair_stats"] is not None,
                "repair_stats": str(ep["repair_stats"]) if ep["repair_stats"] else None,
                "mean_eh": ep["mean_eh_this_epoch"],
            })

    return results


def _print_comparison(all_rows, scenario, k):
    methods = sorted(set(r["method"] for r in all_rows))
    ef = 32  # primary ef for comparison table

    header = (
        f"{'Method':<22} | {'Pre R@'+str(k):>10} | {'Post R@'+str(k):>10} | "
        f"{'Drop':>6} | {'Events':>7} | {'Latency':>10}"
    )
    print(f"\n{'='*len(header)}")
    print(f"Scenario: {scenario}  (ef_search={ef})")
    print(header)
    print("-" * len(header))

    for method in methods:
        rows = [r for r in all_rows if r["method"] == method and r["ef_search"] == ef]
        pre = np.mean([r["recall_at_k"] for r in rows if r["epoch_idx"] <= 4])
        post = np.mean([r["recall_at_k"] for r in rows if r["epoch_idx"] >= 20])
        drop = pre - post
        events = sum(1 for r in rows if r.get("repair_happened") or r.get("rebuilt_this_epoch"))
        lat = np.mean([r["mean_latency_ms"] for r in rows])
        print(
            f"{method:<22} | {pre:>10.4f} | {post:>10.4f} | "
            f"{drop:>6.3f} | {events:>7} | {lat:>8.2f}ms"
        )
    print("=" * len(header))


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


def run_all(config_path):
    cfg = _load_config(config_path)
    k = cfg["eval"]["recall_k"]
    results_dir = str(ROOT / cfg["output"]["results_dir"])
    index_path = str(ROOT / cfg["output"]["index_path"])

    for scenario in ["gradual", "sudden"]:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario}")
        print("=" * 60)

        dataset_key = f"dataset_{scenario}_path"
        dataset = load_drift_dataset(str(ROOT / cfg["output"][dataset_key]))
        base = dataset["base"]

        index = hnswlib.Index(space="l2", dim=base.shape[1])
        index.load_index(index_path)

        all_rows = []

        print("\n[1/3] Static HNSW")
        static_rows = run_static_hnsw(dataset, index, cfg, scenario)
        all_rows.extend(static_rows)
        print(f"  {len(static_rows)} rows")

        print("\n[2/3] Periodic rebuild")
        rebuild_rows = run_periodic_rebuild(dataset, base, cfg)
        all_rows.extend(rebuild_rows)
        print(f"  {len(rebuild_rows)} rows")

        print("\n[3/3] Adaptive")
        adaptive_rows = run_adaptive(dataset, index, base, cfg)
        all_rows.extend(adaptive_rows)
        print(f"  {len(adaptive_rows)} rows")

        out_csv = os.path.join(results_dir, f"{scenario}_all.csv")
        _save_results(all_rows, out_csv)
        _print_comparison(all_rows, scenario, k)


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
