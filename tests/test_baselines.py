"""Tests for PeriodicRebuildBaseline."""

import numpy as np

from src.baselines.periodic_rebuild import PeriodicRebuildBaseline


def _make_baseline(n=200, dim=8, interval=5):
    rng = np.random.default_rng(0)
    base = rng.random((n, dim), dtype=np.float32)
    return PeriodicRebuildBaseline(base=base, M=8, ef_construction=50,
                                   rebuild_interval=interval, seed=0), base


def _make_groundtruth(base, queries, k):
    """Exact brute-force ground truth."""
    gt = []
    for q in queries:
        dists = np.linalg.norm(base - q, axis=1)
        gt.append(np.argsort(dists)[:k])
    return np.array(gt, dtype=np.int32)


def test_rebuild_happens_at_interval():
    baseline, base = _make_baseline(interval=5)
    rng = np.random.default_rng(1)
    dim = base.shape[1]
    k = 5
    rebuild_epochs = []

    for epoch_idx in range(11):
        queries = rng.random((20, dim), dtype=np.float32)
        gt = _make_groundtruth(base, queries, k)
        row = baseline.process_epoch(epoch_idx, queries, gt, k=k, ef_search=50)
        if row["rebuilt_this_epoch"]:
            rebuild_epochs.append(epoch_idx)

    assert rebuild_epochs == [5, 10], f"Expected rebuilds at [5, 10], got {rebuild_epochs}"


def test_recall_reasonable_after_rebuild():
    baseline, base = _make_baseline(n=300, interval=5)
    rng = np.random.default_rng(2)
    dim = base.shape[1]
    k = 5

    queries = rng.random((50, dim), dtype=np.float32)
    gt = _make_groundtruth(base, queries, k)

    # Epoch 5 triggers a rebuild
    for epoch_idx in range(6):
        row = baseline.process_epoch(epoch_idx, queries, gt, k=k, ef_search=100)

    assert row["rebuilt_this_epoch"]
    assert row["recall_at_k"] >= 0.7, f"Recall after rebuild too low: {row['recall_at_k']}"
