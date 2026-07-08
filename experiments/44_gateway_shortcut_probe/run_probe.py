"""Experiment 44 oracle probe: gateway→missed-NN shortcuts for cluster drift.

Tests whether inserting edges from early-visited (gateway) nodes to missed true NNs
improves recall at ef_low without increasing the beam width.

Three-phase structure on a single fully-drifted epoch:
  1. baseline  — HNSW at ef_low and ef_values sweep (characterise the gap)
  2. collect   — for each query: identify gateway nodes (first n_gateway visited at ef_low)
                 and missed NNs (FAISS true NNs absent from ef_low result);
                 accumulate (gateway, missed_NN) pairs without modifying the index
  3. insert    — insert all pairs via add_layer0_edge_evict
  4. shortcut  — HNSW at ef_low on the same queries after insertions

"""

import os
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hnswlib
from src.drift.cluster_drift import load_cluster_drift_dataset


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _compute_recall(ids, gt, k):
    hits = [len(set(ids[i].tolist()) & set(gt[i, :k].tolist())) / k for i in range(len(ids))]
    return float(np.mean(hits))


def run_probe(dataset, index_path, cfg, schedule_name):
    k = cfg["k"]
    ef_low = cfg["ef_low"]
    ef_values = cfg["ef_values"]
    n_gateway = cfg["n_gateway"]
    probe_epoch = cfg["probe_epoch"]

    base = dataset["base"]
    queries = dataset["epochs"][probe_epoch]
    gt = dataset["groundtruth"][probe_epoch]
    n_queries = len(queries)

    if not hasattr(hnswlib, "get_last_query_stats"):
        raise RuntimeError(
            "hnswlib.get_last_query_stats not available — "
            "rebuild hnswlib with the thesis C++ patch"
        )

    print(f"  Building FAISS flat index over {base.shape[0]:,} vectors...")
    fi = faiss.IndexFlatL2(base.shape[1])
    fi.add(base.astype(np.float32))

    # load a fresh index (insertions modify it in place)
    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.load_index(index_path, max_elements=base.shape[0])

    rows = []

    # ---- Phase 1: baseline ef sweep ----
    print("  Phase 1: baseline")
    for ef in ef_values:
        index.set_ef(ef)
        ids, _ = index.knn_query(queries, k=k, num_threads=1)
        recall = _compute_recall(ids, gt, k)
        rows.append({"schedule": schedule_name, "phase": "baseline", "ef": ef, "recall": recall,
                     "edges_inserted": 0})
        print(f"    ef={ef:4d}  recall={recall:.4f}")

    # ---- Phase 2: collect gateway→missed-NN pairs ----
    print(f"  Phase 2: collecting pairs (ef_low={ef_low}, n_gateway={n_gateway})")
    pairs = set()   # (gateway_node, missed_nn) — deduplicated across queries
    n_queries_with_misses = 0

    for i in range(n_queries):
        q = queries[i]

        # ef_low search — must be single-query for get_last_query_stats
        index.set_ef(ef_low)
        ids_q, _ = index.knn_query(q.reshape(1, -1), k=k)
        stats = hnswlib.get_last_query_stats()
        visited = np.array(stats["base_layer_visited_node_ids"], dtype=np.int64)

        gateway_nodes = visited[:n_gateway].tolist()

        # FAISS exact: which true NNs did ef_low miss?
        _, true_nn = fi.search(q.reshape(1, -1).astype(np.float32), k)
        result_set = set(ids_q[0].tolist())
        missed = [int(n) for n in true_nn[0] if n not in result_set]

        if not missed:
            continue
        n_queries_with_misses += 1

        for gw in gateway_nodes:
            for mn in missed:
                if int(gw) != mn:
                    pairs.add((int(gw), mn))

    print(f"    queries with misses: {n_queries_with_misses}/{n_queries}  "
          f"unique pairs: {len(pairs)}")

    # ---- Phase 3: insert all pairs ----
    print("  Phase 3: inserting edges")
    n_inserted = 0
    for gw, mn in pairs:
        ok_fwd = index.add_layer0_edge_evict(gw, mn)
        ok_rev = index.add_layer0_edge_evict(mn, gw)
        n_inserted += int(ok_fwd) + int(ok_rev)
    print(f"    edges inserted: {n_inserted} / {2 * len(pairs)} attempted "
          f"({100 * n_inserted / max(1, 2 * len(pairs)):.1f}% acceptance)")

    # ---- Phase 4: recall after shortcuts ----
    print("  Phase 4: recall after shortcuts")
    for ef in ef_values:
        index.set_ef(ef)
        ids, _ = index.knn_query(queries, k=k, num_threads=1)
        recall = _compute_recall(ids, gt, k)
        rows.append({"schedule": schedule_name, "phase": "shortcut", "ef": ef, "recall": recall,
                     "edges_inserted": n_inserted})
        baseline = next(r["recall"] for r in rows if r["phase"] == "baseline" and r["ef"] == ef)
        print(f"    ef={ef:4d}  recall={recall:.4f}  (delta={recall - baseline:+.4f})")

    return rows


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "config.yaml"
    cfg = _load_config(config_path)
    index_path = str(ROOT / cfg["index_path"])

    all_rows = []
    for schedule, ds_key in [("gradual", "dataset_grad"), ("sudden", "dataset_sudden")]:
        print(f"\n=== {schedule} — probe epoch {cfg['probe_epoch']} ===")
        dataset = load_cluster_drift_dataset(str(ROOT / cfg[ds_key]))
        rows = run_probe(dataset, index_path, cfg, schedule)
        all_rows.extend(rows)

    out = ROOT / cfg["results"]
    os.makedirs(out.parent, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"\nSaved → {out}")
    print(pd.DataFrame(all_rows).to_string(index=False))


if __name__ == "__main__":
    main()
