import csv
import json
import os
import time
from datetime import datetime

import hnswlib
import numpy as np

from src.drift.dataset import load_drift_dataset


def evaluate_epoch(index, queries, groundtruth, ef_search, k):
    index.set_ef(ef_search)
    query_times = []
    hits = []
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        labels, _ = index.knn_query(q.reshape(1, -1), k=k)
        query_times.append((time.perf_counter() - t0) * 1000)
        true_set = set(groundtruth[i, :k].tolist())
        found = len(set(labels[0].tolist()) & true_set)
        hits.append(found / k)
    return dict(
        recall=float(np.mean(hits)),
        mean_query_time_ms=float(np.mean(query_times)),
        ef_search=ef_search,
        n_queries=len(queries),
    )


def run_drift_evaluation(index_path, dataset_path, ef_values, k):
    dataset = load_drift_dataset(dataset_path)
    epochs = dataset["epochs"]
    groundtruth = dataset["groundtruth"]  # list of per-epoch arrays
    diagnostics = dataset["diagnostics"]
    config = dataset["config"]

    index = hnswlib.Index(space="l2", dim=epochs[0].shape[1])
    index.load_index(index_path)

    results = []
    for epoch_idx, epoch_queries in enumerate(epochs):
        t_val = diagnostics[epoch_idx].get("t_value") if epoch_idx < len(diagnostics) else None
        for ef in ef_values:
            row = evaluate_epoch(index, epoch_queries, groundtruth[epoch_idx], ef, k)
            row["epoch"] = epoch_idx
            row["t"] = t_val
            row.update(config)
            results.append(row)
    return results


def save_results(results, out_path):
    if not results:
        return
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    meta = {
        "timestamp": datetime.utcnow().isoformat(),
        "out_path": out_path,
        "config": results[0],
    }
    with open(out_path.replace(".csv", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
