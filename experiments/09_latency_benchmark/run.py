"""Experiment 09: latency benchmark — query overhead and repair cost.

Three measurement parts:
  A. Latency vs accumulated edge count (ef=32, 4 methods, 5 CG checkpoints)
  B. Latency vs ef_search (epoch-24 CG, 4 methods, ef in {16,32,64,128})
  C. Repair event cost (compute_candidate_edges + apply_repairs, N repetitions)

Four query-time methods compared:
  batch_knn          — batch C++ knn_query (static baseline)
  perquery_knn       — per-query Python loop, no conjugate (isolates loop overhead)
  perquery_enhanced  — current search_enhanced implementation
  batch_knn_conj     — batch knn + vectorised numpy conjugate expansion (proposed)
"""

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

from src.drift.adapter import (
    AdaptationManager,
    compute_candidate_edges,
    apply_repairs,
    find_repair_candidates,
)
from src.drift.conjugate_graph import ConjugateGraph
from src.drift.dataset import load_drift_dataset
from src.drift.detector import (
    DriftDetector,
    assign_cell,
    build_spatial_index,
    compute_eh_batch,
)


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _out_dir(cfg):
    p = ROOT / cfg["output"]["results_dir"]
    if p.exists():
        return p
    # Fallback to script directory if results/ not present
    return Path(__file__).resolve().parent


def _save_csv(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = sorted(set(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    print(f"  Saved {len(rows)} rows -> {path}")


# ---------------------------------------------------------------------------
# Query methods
# ---------------------------------------------------------------------------

def _method_batch_knn(index, queries, k, ef):
    """Single batch C++ knn_query."""
    index.set_ef(ef)
    return index.knn_query(queries, k=k)


def _method_perquery_knn(index, queries, k, ef):
    """Per-query Python loop, no conjugate expansion."""
    index.set_ef(ef)
    n = len(queries)
    all_ids = np.empty((n, k), dtype=np.int64)
    all_dists = np.empty((n, k), dtype=np.float32)
    for i, q in enumerate(queries):
        lbl, dst = index.knn_query(q.reshape(1, -1), k=k)
        all_ids[i] = lbl[0]
        all_dists[i] = dst[0]
    return all_ids, all_dists


def _method_perquery_enhanced(index, cg, base, queries, k, ef, current_epoch, two_hop):
    """Current production path: per-query knn + scalar conjugate expansion."""
    index.set_ef(ef)
    n = len(queries)
    all_ids = np.empty((n, k), dtype=np.int64)
    all_dists = np.empty((n, k), dtype=np.float32)
    for i, q in enumerate(queries):
        lbl, dst = index.knn_query(q.reshape(1, -1), k=k)
        aug_ids, aug_dists = cg.enhanced_search(
            lbl[0], dst[0], base, q, k, current_epoch, two_hop=two_hop
        )
        all_ids[i] = aug_ids
        all_dists[i] = aug_dists
    return all_ids, all_dists


def _method_batch_knn_conj(index, cg, base, queries, k, ef, current_epoch, two_hop):
    """Batch knn + vectorised numpy distances for conjugate expansion."""
    index.set_ef(ef)
    batch_labels, batch_dists = index.knn_query(queries, k=k)
    n = len(queries)
    all_ids = np.empty((n, k), dtype=np.int64)
    all_dists = np.empty((n, k), dtype=np.float32)

    for i in range(n):
        labels = batch_labels[i]
        distances = batch_dists[i]
        q = queries[i]

        candidate_ids = list(labels)
        candidate_dists = list(distances)
        seen = set(labels.tolist())

        hop1_new = []
        for node_id in labels:
            for edge in cg._edges.get(int(node_id), []):
                nb = edge.neighbor_id
                if nb not in seen:
                    seen.add(nb)
                    hop1_new.append(nb)

        if hop1_new:
            vecs = base[hop1_new]
            diffs = vecs - q
            dists_v = np.einsum("id,id->i", diffs, diffs)
            candidate_ids.extend(hop1_new)
            candidate_dists.extend(dists_v.tolist())

        if two_hop:
            hop2_new = []
            for node_id in hop1_new:
                for edge in cg._edges.get(node_id, []):
                    nb = edge.neighbor_id
                    if nb not in seen:
                        seen.add(nb)
                        hop2_new.append(nb)
            if hop2_new:
                vecs2 = base[hop2_new]
                diffs2 = vecs2 - q
                dists2 = np.einsum("id,id->i", diffs2, diffs2)
                candidate_ids.extend(hop2_new)
                candidate_dists.extend(dists2.tolist())

        dists_arr = np.array(candidate_dists, dtype=np.float32)
        ids_arr = np.array(candidate_ids, dtype=np.int64)
        if len(candidate_ids) > k:
            top_k_idx = np.argpartition(dists_arr, k)[:k]
            top_k_idx = top_k_idx[np.argsort(dists_arr[top_k_idx])]
        else:
            top_k_idx = np.argsort(dists_arr)
        all_ids[i] = ids_arr[top_k_idx]
        all_dists[i] = dists_arr[top_k_idx]

    return all_ids, all_dists


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _time_fn(fn, n_queries, n_warmup, n_bench):
    """Return list of per-query latencies (ms) over n_bench rounds."""
    times = []
    for i in range(n_warmup + n_bench):
        t0 = time.perf_counter()
        fn()
        dt = (time.perf_counter() - t0) * 1000 / n_queries
        if i >= n_warmup:
            times.append(dt)
    return times


def _recall(ids, gt, k):
    hits = [len(set(ids[i].tolist()) & set(gt[i, :k].tolist())) / k
            for i in range(len(ids))]
    return float(np.mean(hits))


def _stats_row(latencies, n_queries):
    arr = np.array(latencies)
    mean_ms = float(arr.mean())
    return {
        "mean_ms_per_query": mean_ms,
        "std_ms_per_query": float(arr.std()),
        "p50_ms_per_query": float(np.percentile(arr, 50)),
        "p95_ms_per_query": float(np.percentile(arr, 95)),
        "qps": float(1000.0 / mean_ms) if mean_ms > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Parts A & B: latency measurements
# ---------------------------------------------------------------------------

def run_latency_benchmarks(cfg, index, base, queries, gt, out_dir):
    """Parts A and B: time all 4 methods across edge counts and ef values."""
    bench_cfg = cfg["benchmark"]
    adapt_cfg = cfg["adaptation"]
    n_warmup = bench_cfg["n_warmup_rounds"]
    n_bench = bench_cfg["n_bench_rounds"]
    k = bench_cfg["recall_k"]
    primary_ef = bench_cfg["primary_ef"]
    ef_values = bench_cfg["ef_search_values"]
    two_hop = adapt_cfg.get("two_hop", True)
    current_epoch = bench_cfg["bench_epoch"]
    n_q = len(queries)

    cg_dir = out_dir / "cg_states"
    checkpoint_epochs = cfg["checkpoint_epochs"] + [0]  # 0 = zero-edge CG

    # Map checkpoint epoch -> (CG path, n_edges label)
    cg_files = {}
    zero_path = cg_dir / "cg_ep00_zero.json"
    if zero_path.exists():
        cg_files[0] = zero_path
    for ep in cfg["checkpoint_epochs"]:
        p = cg_dir / f"cg_ep{ep:02d}.json"
        if p.exists():
            cg_files[ep] = p

    rows = []

    # ------------------------------------------------------------------
    # Part A: latency vs edge count (primary_ef, all 4 methods)
    # ------------------------------------------------------------------
    print("\n=== Part A: latency vs edge count ===")
    for ep_key in sorted(cg_files):
        cg = ConjugateGraph.load(str(cg_files[ep_key]))
        n_edges = cg._total_edges
        print(f"\n  CG ep{ep_key:02d}: {n_edges} edges")

        methods = {
            "batch_knn": lambda: _method_batch_knn(index, queries, k, primary_ef),
            "perquery_knn": lambda: _method_perquery_knn(index, queries, k, primary_ef),
            "perquery_enhanced": lambda: _method_perquery_enhanced(
                index, cg, base, queries, k, primary_ef, current_epoch, two_hop),
            "batch_knn_conj": lambda: _method_batch_knn_conj(
                index, cg, base, queries, k, primary_ef, current_epoch, two_hop),
        }

        for method_name, fn in methods.items():
            latencies = _time_fn(fn, n_q, n_warmup, n_bench)
            result_ids, _ = fn()
            rec = _recall(result_ids, gt, k) if method_name not in ("batch_knn", "perquery_knn") else None
            row = {
                "part": "latency_vs_edges",
                "method": method_name,
                "n_edges": n_edges,
                "ef_search": primary_ef,
                **_stats_row(latencies, n_q),
            }
            if rec is not None:
                row["recall_at_k"] = rec
            rows.append(row)
            print(f"    {method_name:<22} {row['mean_ms_per_query']:.4f} ms/q  "
                  f"({row['qps']:.0f} QPS)")

    # ------------------------------------------------------------------
    # Part B: latency vs ef_search (epoch-24 CG, all 4 methods)
    # ------------------------------------------------------------------
    print("\n=== Part B: latency vs ef_search (epoch-24 CG) ===")
    ep24_key = max(cg_files)
    cg24 = ConjugateGraph.load(str(cg_files[ep24_key]))
    n_edges_24 = cg24._total_edges

    for ef in ef_values:
        print(f"\n  ef={ef}")
        methods = {
            "batch_knn": lambda ef=ef: _method_batch_knn(index, queries, k, ef),
            "perquery_knn": lambda ef=ef: _method_perquery_knn(index, queries, k, ef),
            "perquery_enhanced": lambda ef=ef: _method_perquery_enhanced(
                index, cg24, base, queries, k, ef, current_epoch, two_hop),
            "batch_knn_conj": lambda ef=ef: _method_batch_knn_conj(
                index, cg24, base, queries, k, ef, current_epoch, two_hop),
        }
        for method_name, fn in methods.items():
            latencies = _time_fn(fn, n_q, n_warmup, n_bench)
            result_ids, _ = fn()
            rec = _recall(result_ids, gt, k) if method_name not in ("batch_knn", "perquery_knn") else None
            row = {
                "part": "latency_vs_ef",
                "method": method_name,
                "n_edges": n_edges_24,
                "ef_search": ef,
                **_stats_row(latencies, n_q),
            }
            if rec is not None:
                row["recall_at_k"] = rec
            rows.append(row)
            print(f"    {method_name:<22} {row['mean_ms_per_query']:.4f} ms/q")

    return rows


# ---------------------------------------------------------------------------
# Part C: repair event cost
# ---------------------------------------------------------------------------

def run_repair_benchmark(cfg, index, base, queries, hnsw_ids, hnsw_dists, out_dir,
                         detector, cell_labels, centroids):
    """Time compute_candidate_edges + apply_repairs independently."""
    adapt_cfg = cfg["adaptation"]
    bench_cfg = cfg["benchmark"]
    n_reps = bench_cfg["n_repair_reps"]
    k = bench_cfg["recall_k"]
    epoch_size = bench_cfg["epoch_size"]
    current_epoch = bench_cfg["bench_epoch"]

    print("\n=== Part C: repair event cost ===")

    # Use drift detector to get hot cells from post-drift queries
    eh_vals = compute_eh_batch([base[hnsw_ids[j]] for j in range(len(queries))])
    cell_ids = assign_cell(queries, centroids)
    detector.update_batch(eh_vals, cell_ids)
    drift_result = detector.check_drift()
    hot_cells = drift_result["hot_cells"] if drift_result else list(range(adapt_cfg["n_cells"]))
    if not hot_cells:
        hot_cells = list(range(adapt_cfg["n_cells"]))

    node_eh = np.zeros(base.shape[0], dtype=np.float64)
    for i, result_row in enumerate(hnsw_ids):
        for nid in result_row:
            node_eh[nid] = 0.9 * node_eh[nid] + 0.1 * float(eh_vals[i])

    repair_nodes = find_repair_candidates(
        hot_cells, cell_labels, node_eh,
        eh_threshold_percentile=adapt_cfg["eh_threshold_percentile"],
        max_nodes=adapt_cfg["max_repair_nodes"],
        last_repaired=np.full(base.shape[0], -1, dtype=np.int32),
        use_diversity=False,
    )
    print(f"  repair_nodes selected: {len(repair_nodes)}")

    rows = []

    # Time compute_candidate_edges
    cce_times = []
    print(f"  Timing compute_candidate_edges ({n_reps} reps)...")
    for rep in range(n_reps):
        t0 = time.perf_counter()
        cands = compute_candidate_edges(
            repair_nodes, index, base,
            M_candidates=adapt_cfg["M_candidates"],
            ef_search=adapt_cfg["ef_repair"],
            queries=queries,
            result_ids=hnsw_ids,
        )
        dt = (time.perf_counter() - t0) * 1000
        cce_times.append(dt)
        print(f"    rep {rep}: {dt:.1f} ms")

    cce_arr = np.array(cce_times)
    rows.append({
        "part": "repair",
        "component": "compute_candidate_edges",
        "n_repair_nodes": len(repair_nodes),
        "mean_ms": float(cce_arr.mean()),
        "std_ms": float(cce_arr.std()),
        "p50_ms": float(np.percentile(cce_arr, 50)),
        "p95_ms": float(np.percentile(cce_arr, 95)),
        "amortized_ms_per_query": float(cce_arr.mean() / epoch_size),
    })

    # Time apply_repairs (use fresh CG each rep)
    ar_times = []
    print(f"  Timing apply_repairs ({n_reps} reps)...")
    for rep in range(n_reps):
        cg_fresh = ConjugateGraph(
            M_conj=adapt_cfg["M_conj"],
            max_total_edges=adapt_cfg["max_total_edges"],
        )
        t0 = time.perf_counter()
        apply_repairs(
            cands, cg_fresh,
            t_added=current_epoch / 25.0,
            epoch_added=current_epoch,
            current_epoch=current_epoch,
            base=base,
            rng_relaxation=adapt_cfg["rng_relaxation"],
        )
        dt = (time.perf_counter() - t0) * 1000
        ar_times.append(dt)

    ar_arr = np.array(ar_times)
    rows.append({
        "part": "repair",
        "component": "apply_repairs",
        "n_repair_nodes": len(repair_nodes),
        "mean_ms": float(ar_arr.mean()),
        "std_ms": float(ar_arr.std()),
        "p50_ms": float(np.percentile(ar_arr, 50)),
        "p95_ms": float(np.percentile(ar_arr, 95)),
        "amortized_ms_per_query": float(ar_arr.mean() / epoch_size),
    })

    total_repair_ms = cce_arr.mean() + ar_arr.mean()
    rows.append({
        "part": "repair",
        "component": "total_repair_event",
        "n_repair_nodes": len(repair_nodes),
        "mean_ms": total_repair_ms,
        "std_ms": float(np.sqrt(cce_arr.var() + ar_arr.var())),
        "p50_ms": float(np.percentile(cce_arr + ar_arr, 50)),
        "p95_ms": float(np.percentile(cce_arr + ar_arr, 95)),
        "amortized_ms_per_query": total_repair_ms / epoch_size,
    })

    print(f"\n  compute_candidate_edges: {cce_arr.mean():.1f} ± {cce_arr.std():.1f} ms")
    print(f"  apply_repairs:           {ar_arr.mean():.1f} ± {ar_arr.std():.1f} ms")
    print(f"  total repair event:      {total_repair_ms:.1f} ms")
    print(f"  amortized per query:     {total_repair_ms / epoch_size:.3f} ms")

    return rows


# ---------------------------------------------------------------------------
# Setup: rebuild detector state for Part C
# ---------------------------------------------------------------------------

def _build_detector(cfg, index, base, dataset):
    """Calibrate a fresh detector on the first n_calib epochs."""
    adapt_cfg = cfg["adaptation"]
    bench_cfg = cfg["benchmark"]
    k = bench_cfg["recall_k"]
    primary_ef = bench_cfg["primary_ef"]
    n_calib = adapt_cfg["n_calibration_epochs"]

    centroids, cell_labels = build_spatial_index(
        base, n_cells=adapt_cfg["n_cells"], seed=42
    )
    calib_eh, calib_cell_ids = [], []
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
    return detector, cell_labels, centroids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    args = parser.parse_args()

    cfg = _load_config(args.config)
    out_dir = _out_dir(cfg)
    cg_dir = out_dir / "cg_states"

    if not cg_dir.exists() or not any(cg_dir.glob("*.json")):
        print("ERROR: CG state files not found. Run build_cg_state.py first.")
        sys.exit(1)

    dataset = load_drift_dataset(str(ROOT / cfg["output"]["dataset_gradual_path"]))
    base = dataset["base"]
    index_path = str(ROOT / cfg["output"]["index_path"])
    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.load_index(index_path)

    # Load pre-saved benchmark data
    queries = np.load(str(out_dir / "bench_queries.npy"))
    gt = np.load(str(out_dir / "bench_gt.npy"))
    hnsw_ids = np.load(str(out_dir / "bench_hnsw_ids.npy"))
    hnsw_dists = np.load(str(out_dir / "bench_hnsw_dists.npy"))

    print(f"Benchmark queries: {queries.shape}  edges in cg_states: "
          f"{len(list(cg_dir.glob('*.json')))} checkpoints")

    # Rebuild detector for Part C
    detector, cell_labels, centroids = _build_detector(cfg, index, base, dataset)

    # Parts A & B
    latency_rows = run_latency_benchmarks(cfg, index, base, queries, gt, out_dir)
    _save_csv(latency_rows, str(out_dir / "latency_results.csv"))

    # Part C
    repair_rows = run_repair_benchmark(
        cfg, index, base, queries, hnsw_ids, hnsw_dists, out_dir,
        detector, cell_labels, centroids,
    )
    _save_csv(repair_rows, str(out_dir / "repair_results.csv"))

    print("\nDone.")


if __name__ == "__main__":
    main()
