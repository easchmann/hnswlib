"""Hardness-based query drift construction for exp33.

Queries are sorted ascending by hardness (index 0 = easiest, last = hardest).
Drift is constructed by mixing easy and hard pool queries per a t-value schedule.
"""

import numpy as np


def build_hardness_drift_schedule(n_epochs, schedule):
    """Build list of t values; schedule is 'gradual' or 'sudden'."""
    if schedule == "gradual":
        return (
            [0.0] * 5
            + list(np.linspace(0.0, 0.5, 5))
            + [0.5] * 5
            + list(np.linspace(0.5, 1.0, 5))
            + [1.0] * 5
        )
    if schedule == "sudden":
        return [0.0] * 13 + [1.0] * 12
    raise ValueError(f"Unknown schedule: {schedule!r}")


def build_hardness_drift_dataset(queries_sorted, ground_truth, schedule, epoch_size,
                                  easy_pool_size, hard_pool_size, seed):
    """Build epoch arrays by mixing easy/hard pool queries at each t value in schedule.

    Returns dict with keys base (placeholder), epochs, groundtruth.
    base is an empty (0, dim) array; caller must substitute real base before eval.
    """
    n_total = len(queries_sorted)
    hard_start = n_total - hard_pool_size
    dim = queries_sorted.shape[1]

    epochs, gts = [], []
    for epoch_idx, t in enumerate(schedule):
        rng = np.random.default_rng(seed + epoch_idx)
        n_hard = round(epoch_size * t)
        n_easy = epoch_size - n_hard

        easy_idx = rng.integers(0, easy_pool_size, size=n_easy) if n_easy > 0 else np.empty(0, dtype=np.int64)
        hard_idx = rng.integers(hard_start, n_total, size=n_hard) if n_hard > 0 else np.empty(0, dtype=np.int64)

        idx = np.concatenate([easy_idx, hard_idx]).astype(np.int64)
        idx = idx[rng.permutation(len(idx))]

        epochs.append(queries_sorted[idx].copy().astype(np.float32))
        gts.append(np.array(ground_truth[idx]))

    placeholder_base = np.zeros((0, dim), dtype=np.float32)
    return dict(base=placeholder_base, epochs=epochs, groundtruth=gts)
