"""DEEP-96 loader and recall-based hot/cold cluster selection for exp16."""

import faiss
import h5py
import numpy as np

from src.drift.cluster_drift import (
    build_cluster_drift_sequence,
    build_query_clusters,
    characterise_cluster_drift,
    compute_groundtruth,
    load_cluster_drift_dataset,
    save_cluster_drift_dataset,
    sample_epoch_queries,
)


def load_deep_hdf5(path, n_vectors=None):
    """Returns (base, query_pool) as float32 arrays, L2-normalised."""
    with h5py.File(path, "r") as f:
        base = f["train"][:n_vectors].astype(np.float32)
        queries = f["test"][:].astype(np.float32)
    base = base / np.linalg.norm(base, axis=1, keepdims=True)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    return base, queries


def select_hot_clusters_by_recall(index, base, query_pool, cluster_labels, n_hot, k, ef):
    """Return (hot_cluster_ids, per_cluster_recall_dict).

    Selects n_hot query clusters with lowest HNSW recall@k at ef_search=ef.
    Prints a diagnostic table. Call this before building the drift sequence.
    """
    n_clusters = len(np.unique(cluster_labels))

    faiss_index = faiss.IndexFlatL2(base.shape[1])
    faiss_index.add(base.astype(np.float32))

    index.set_ef(ef)

    per_cluster_recall = {}
    for c in range(n_clusters):
        mask = cluster_labels == c
        q_c = query_pool[mask]
        if len(q_c) == 0:
            per_cluster_recall[c] = 1.0
            continue
        _, gt = faiss_index.search(q_c.astype(np.float32), k)
        ids, _ = index.knn_query(q_c, k=k, num_threads=1)
        recall = float(np.mean([
            len(set(ids[i].tolist()) & set(gt[i, :k].tolist())) / k
            for i in range(len(q_c))
        ]))
        per_cluster_recall[c] = recall

    print(f"\nPer-cluster recall@{k} at ef={ef}  (pre-drift diagnostic):")
    print(f"  {'Cluster':>8}  {'Size':>7}  {'Recall':>8}")
    for c in range(n_clusters):
        size = int((cluster_labels == c).sum())
        print(f"  {c:>8}  {size:>7}  {per_cluster_recall[c]:>8.4f}")

    sorted_clusters = sorted(per_cluster_recall, key=per_cluster_recall.__getitem__)
    hot_ids = sorted_clusters[:n_hot]

    hot_recall_max = per_cluster_recall[hot_ids[-1]]
    cold_recall_min = per_cluster_recall[sorted_clusters[n_hot]]
    print(f"\n  Selected hot clusters (lowest recall): {hot_ids}")
    print(f"  Max hot-cluster recall:  {hot_recall_max:.4f}")
    print(f"  Min cold-cluster recall: {cold_recall_min:.4f}")
    print(f"  Gap: {cold_recall_min - hot_recall_max:.4f} pp")
    print()

    return hot_ids, per_cluster_recall


def select_cold_clusters_by_recall(per_cluster_recall, n_cold, exclude_ids):
    """Return n_cold cluster IDs with highest recall, excluding hot clusters."""
    candidates = {c: r for c, r in per_cluster_recall.items() if c not in set(exclude_ids)}
    sorted_desc = sorted(candidates, key=candidates.__getitem__, reverse=True)
    cold_ids = sorted_desc[:n_cold]
    cold_recall_min = per_cluster_recall[cold_ids[-1]]
    hot_recall_max = per_cluster_recall[exclude_ids[-1]]
    print(f"  Selected cold clusters (highest recall): {cold_ids}")
    print(f"  Min cold-cluster recall: {cold_recall_min:.4f}")
    print(f"  Max hot-cluster recall:  {hot_recall_max:.4f}")
    print(f"  Bimodal gap (cold_min - hot_max): {cold_recall_min - hot_recall_max:.4f} pp")
    print()
    return cold_ids


def sample_epoch_queries_bimodal(query_pool, cluster_labels, hot_cluster_ids,
                                  cold_cluster_ids, t, epoch_size, rng):
    """t=0 → draw exclusively from cold (easy) clusters; t=1 → from hot (hard) clusters."""
    hot_set = set(hot_cluster_ids)
    cold_set = set(cold_cluster_ids)
    n_hot = len(hot_cluster_ids)
    n_cold = len(cold_cluster_ids)

    weights = np.zeros(len(query_pool))
    for i, lbl in enumerate(cluster_labels):
        if lbl in hot_set:
            weights[i] = t / n_hot
        elif lbl in cold_set:
            weights[i] = (1.0 - t) / n_cold

    total = weights.sum()
    if total <= 0:
        raise ValueError(f"All weights zero at t={t} — check hot/cold cluster IDs")
    weights /= total

    idx = rng.choice(len(query_pool), size=epoch_size, replace=True, p=weights)
    return query_pool[idx]


def build_cluster_drift_sequence_bimodal(query_pool, cluster_labels, hot_cluster_ids,
                                          cold_cluster_ids, schedule, epoch_size, base_seed):
    """Build drift sequence using bimodal sampling (cold at t=0, hot at t=1)."""
    epochs = []
    for epoch_idx, t in enumerate(schedule):
        rng = np.random.default_rng(base_seed + epoch_idx)
        q = sample_epoch_queries_bimodal(
            query_pool, cluster_labels, hot_cluster_ids, cold_cluster_ids,
            t, epoch_size, rng
        )
        epochs.append(q)
    return epochs
