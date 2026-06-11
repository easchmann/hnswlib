"""Tests for src/drift/detector.py"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.drift.detector import (
    SlidingWindowBuffer,
    assign_cell,
    build_spatial_index,
    compute_eh,
    compute_eh_batch,
)


def test_eh_zero_for_identical_vectors():
    k, dim = 5, 8
    vecs = np.ones((k, dim), dtype=np.float32)
    assert compute_eh(vecs) == pytest.approx(0.0)


def test_eh_positive_for_spread_vectors():
    rng = np.random.default_rng(0)
    vecs = rng.uniform(0, 100, size=(10, 8)).astype(np.float32)
    assert compute_eh(vecs) > 0.0


def test_eh_increases_with_spread():
    rng = np.random.default_rng(42)
    dim = 16
    tight = rng.normal(0, 0.01, size=(10, dim)).astype(np.float32)
    spread = rng.normal(0, 10.0, size=(10, dim)).astype(np.float32)
    assert compute_eh(spread) > compute_eh(tight)


def test_sliding_window_circular():
    buf = SlidingWindowBuffer(window_size=5, n_cells=10)
    values = [float(i) for i in range(8)]
    for v in values:
        buf.update(v, cell_id=0)
    result = buf.get_eh_values()
    assert len(result) == 5
    np.testing.assert_array_almost_equal(sorted(result), [3.0, 4.0, 5.0, 6.0, 7.0])


def test_sliding_window_ready():
    window_size = 10
    buf = SlidingWindowBuffer(window_size=window_size, n_cells=5)
    assert not buf.is_ready()
    threshold = window_size // 2 + 1
    for i in range(threshold):
        buf.update(float(i), cell_id=0)
    assert buf.is_ready()


def test_cell_assignment_correct():
    centroids = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], dtype=np.float32)
    queries = np.array([[0.5, 0.5], [9.5, 0.5]], dtype=np.float32)
    cells = assign_cell(queries, centroids)
    assert cells[0] == 0
    assert cells[1] == 1


def test_build_spatial_index_shapes():
    rng = np.random.default_rng(7)
    n_base, dim, n_cells = 500, 16, 10
    base = rng.normal(size=(n_base, dim)).astype(np.float32)
    centroids, labels = build_spatial_index(base, n_cells=n_cells, seed=0)
    assert centroids.shape == (n_cells, dim)
    assert labels.shape == (n_base,)
    assert labels.min() >= 0
    assert labels.max() < n_cells
