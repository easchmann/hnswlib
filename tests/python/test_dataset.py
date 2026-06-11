"""Tests for src/drift/dataset.py"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.drift.dataset import (
    _rotation_cache,
    build_rotation_drift_sequence,
    characterise_drift,
    compute_groundtruth,
    load_drift_dataset,
    load_fvecs,
    make_rotation_generator,
    rotate_queries,
    save_drift_dataset,
)


# Module-level fixtures

@pytest.fixture(scope="module")
def queries_small():
    return np.random.default_rng(42).random((1000, 128)).astype(np.float32)


@pytest.fixture(scope="module")
def A():
    return make_rotation_generator(128, seed=42)


@pytest.fixture(scope="module")
def base_small():
    return np.random.default_rng(7).random((2000, 128)).astype(np.float32)



# Helpers
def _make_fvecs(path: str, vectors: np.ndarray) -> None:
    with open(path, "wb") as f:
        for v in vectors:
            f.write(np.array([len(v)], dtype=np.int32).tobytes())
            f.write(v.astype(np.float32).tobytes())


# load_fvecs
class TestLoadFvecs:
    def test_loads_all(self, tmp_path):
        data = np.random.default_rng(0).random((10, 4)).astype(np.float32)
        p = str(tmp_path / "test.fvecs")
        _make_fvecs(p, data)
        loaded = load_fvecs(p)
        assert loaded.shape == (10, 4)
        np.testing.assert_allclose(loaded, data, atol=1e-6)

    def test_loads_n_vectors(self, tmp_path):
        data = np.random.default_rng(0).random((10, 4)).astype(np.float32)
        p = str(tmp_path / "test.fvecs")
        _make_fvecs(p, data)
        loaded = load_fvecs(p, n_vectors=3)
        assert loaded.shape == (3, 4)

    def test_returns_float32(self, tmp_path):
        data = np.ones((5, 8), dtype=np.float32)
        p = str(tmp_path / "t.fvecs")
        _make_fvecs(p, data)
        loaded = load_fvecs(p)
        assert loaded.dtype == np.float32



# make_rotation_generator

class TestMakeRotationGenerator:
    def test_skew_symmetric(self):
        A = make_rotation_generator(16, seed=7)
        np.testing.assert_allclose(A + A.T, np.zeros_like(A), atol=1e-12)

    def test_spectral_norm_scaled(self):
        # spectral norm should equal pi - i.e. expm(1*A) rotates 180° in principal plane
        A = make_rotation_generator(16, seed=7)
        np.testing.assert_allclose(np.linalg.norm(A, ord=2), np.pi, atol=1e-12)

    def test_shape(self):
        A = make_rotation_generator(32, seed=0)
        assert A.shape == (32, 32)

    def test_dtype_float64(self):
        A = make_rotation_generator(8, seed=1)
        assert A.dtype == np.float64

    def test_reproducible(self):
        A1 = make_rotation_generator(16, seed=42)
        A2 = make_rotation_generator(16, seed=42)
        np.testing.assert_array_equal(A1, A2)

    def test_different_seeds_differ(self):
        A1 = make_rotation_generator(16, seed=1)
        A2 = make_rotation_generator(16, seed=2)
        assert not np.allclose(A1, A2)



# rotate_queries
class TestRotateQueries:
    def setup_method(self):
        _rotation_cache.clear()

    def test_t0_identity(self):
        rng = np.random.default_rng(0)
        queries = rng.random((20, 16)).astype(np.float32)
        A = make_rotation_generator(16, seed=0)
        rotated = rotate_queries(queries, A, t=0.0)
        np.testing.assert_allclose(rotated, queries, atol=1e-5)

    def test_output_shape(self):
        queries = np.random.default_rng(0).random((30, 8)).astype(np.float32)
        A = make_rotation_generator(8, seed=0)
        out = rotate_queries(queries, A, t=0.5)
        assert out.shape == (30, 8)

    def test_output_dtype_float32(self):
        queries = np.random.default_rng(0).random((10, 8)).astype(np.float32)
        A = make_rotation_generator(8, seed=0)
        out = rotate_queries(queries, A, t=0.3)
        assert out.dtype == np.float32

    def test_preserves_norms(self):
        queries = np.random.default_rng(0).random((50, 16)).astype(np.float32)
        A = make_rotation_generator(16, seed=0)
        out = rotate_queries(queries, A, t=0.7)
        in_norms = np.linalg.norm(queries.astype(np.float64), axis=1)
        out_norms = np.linalg.norm(out.astype(np.float64), axis=1)
        np.testing.assert_allclose(in_norms, out_norms, rtol=1e-4)

    def test_caches_rotation_matrix(self):
        _rotation_cache.clear()
        queries = np.random.default_rng(0).random((10, 8)).astype(np.float32)
        A = make_rotation_generator(8, seed=0)
        rotate_queries(queries, A, t=0.5)
        assert 0.5 in _rotation_cache
        cached = _rotation_cache[0.5].copy()
        rotate_queries(queries, A, t=0.5)
        np.testing.assert_array_equal(_rotation_cache[0.5], cached)

    def test_different_t_different_result(self):
        queries = np.random.default_rng(0).random((20, 16)).astype(np.float32)
        A = make_rotation_generator(16, seed=0)
        out1 = rotate_queries(queries, A, t=0.3)
        out2 = rotate_queries(queries, A, t=0.8)
        assert not np.allclose(out1, out2)



# build_rotation_drift_sequence
class TestBuildRotationDriftSequence:
    def setup_method(self):
        _rotation_cache.clear()

    def test_returns_correct_length(self):
        queries = np.random.default_rng(0).random((100, 8)).astype(np.float32)
        A = make_rotation_generator(8, seed=0)
        schedule = [0.0] * 5 + [0.5] * 5
        epochs = build_rotation_drift_sequence(queries, A, schedule, epoch_size=20, seed=1)
        assert len(epochs) == 10

    def test_epoch_shape(self):
        queries = np.random.default_rng(0).random((200, 16)).astype(np.float32)
        A = make_rotation_generator(16, seed=0)
        schedule = [float(i) / 9 for i in range(10)]
        epochs = build_rotation_drift_sequence(queries, A, schedule, epoch_size=50, seed=7)
        for ep in epochs:
            assert ep.shape == (50, 16)

    def test_epoch_dtype_float32(self):
        queries = np.random.default_rng(0).random((100, 8)).astype(np.float32)
        A = make_rotation_generator(8, seed=0)
        epochs = build_rotation_drift_sequence(queries, A, [0.0, 0.5], epoch_size=30, seed=3)
        for ep in epochs:
            assert ep.dtype == np.float32

    def test_sampling_with_replacement(self):
        # With replace=True, epoch_size can exceed pool size
        queries = np.random.default_rng(0).random((10, 8)).astype(np.float32)
        A = make_rotation_generator(8, seed=0)
        epochs = build_rotation_drift_sequence(queries, A, [0.0], epoch_size=50, seed=0)
        assert epochs[0].shape == (50, 8)


# compute_groundtruth
class TestComputeGroundtruth:
    def test_shape(self):
        base = np.random.default_rng(0).random((200, 8)).astype(np.float32)
        queries = np.random.default_rng(1).random((10, 8)).astype(np.float32)
        gt = compute_groundtruth(base, queries, k=5)
        assert gt.shape == (10, 5)

    def test_dtype_int32(self):
        base = np.random.default_rng(0).random((100, 4)).astype(np.float32)
        queries = np.random.default_rng(1).random((5, 4)).astype(np.float32)
        gt = compute_groundtruth(base, queries, k=3)
        assert gt.dtype == np.int32

    def test_exact_nn(self):
        base = np.eye(10, dtype=np.float32)
        queries = base[:3].copy()
        gt = compute_groundtruth(base, queries, k=1)
        np.testing.assert_array_equal(gt[:, 0], [0, 1, 2])

    def test_indices_in_range(self):
        base = np.random.default_rng(0).random((50, 8)).astype(np.float32)
        queries = np.random.default_rng(1).random((20, 8)).astype(np.float32)
        gt = compute_groundtruth(base, queries, k=5)
        assert gt.min() >= 0
        assert gt.max() < 50

    def test_batching_consistent(self):
        base = np.random.default_rng(0).random((100, 8)).astype(np.float32)
        queries = np.random.default_rng(1).random((25, 8)).astype(np.float32)
        gt1 = compute_groundtruth(base, queries, k=5, batch_size=1000)
        gt2 = compute_groundtruth(base, queries, k=5, batch_size=7)
        np.testing.assert_array_equal(gt1, gt2)

    def test_dimension_mismatch_raises(self):
        base = np.random.default_rng(0).random((50, 8)).astype(np.float32)
        queries = np.random.default_rng(1).random((10, 16)).astype(np.float32)
        with pytest.raises(AssertionError):
            compute_groundtruth(base, queries, k=3)



# characterise_drift

class TestCharacteriseDrift:
    def _make_epochs(self, dim=16, n_epochs=5, epoch_size=50):
        _rotation_cache.clear()
        queries = np.random.default_rng(0).random((200, dim)).astype(np.float32)
        A = make_rotation_generator(dim, seed=0)
        schedule = [i / (n_epochs - 1) for i in range(n_epochs)]
        return build_rotation_drift_sequence(queries, A, schedule, epoch_size, seed=0)

    def test_returns_list_of_dicts(self):
        base = np.random.default_rng(9).random((500, 16)).astype(np.float32)
        epochs = self._make_epochs()
        result = characterise_drift(base, epochs)
        assert isinstance(result, list)
        assert len(result) == 5
        for d in result:
            assert isinstance(d, dict)

    def test_required_keys(self):
        base = np.random.default_rng(9).random((500, 16)).astype(np.float32)
        epochs = self._make_epochs()
        result = characterise_drift(base, epochs)
        for d in result:
            for key in ("epoch_idx", "t_value", "ood_distance", "mmd_squared", "mean_nn_distance"):
                assert key in d, f"missing key: {key}"

    def test_epoch_indices(self):
        base = np.random.default_rng(9).random((500, 16)).astype(np.float32)
        epochs = self._make_epochs()
        result = characterise_drift(base, epochs)
        for i, d in enumerate(result):
            assert d["epoch_idx"] == i

    def test_metrics_nonnegative(self):
        base = np.random.default_rng(9).random((500, 16)).astype(np.float32)
        epochs = self._make_epochs()
        result = characterise_drift(base, epochs)
        for d in result:
            assert d["ood_distance"] >= 0
            assert d["mean_nn_distance"] >= 0

    def test_mmd_squared_zero_for_same_distribution(self):
        base = np.random.default_rng(9).random((500, 16)).astype(np.float32)
        q = np.random.default_rng(1).random((50, 16)).astype(np.float32)
        epochs = [q.copy(), q.copy()]
        result = characterise_drift(base, epochs)
        assert abs(result[0]["mmd_squared"]) < 1e-6

    def test_metrics_increase_with_drift(self):
        _rotation_cache.clear()
        dim = 32
        base = np.random.default_rng(0).random((1000, dim)).astype(np.float32)
        queries = np.random.default_rng(1).random((500, dim)).astype(np.float32)
        A = make_rotation_generator(dim, seed=0)
        t_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        epochs = build_rotation_drift_sequence(queries, A, t_values, 100, seed=0)
        result = characterise_drift(base, epochs)
        ood = [d["ood_distance"] for d in result]
        mmd = [d["mmd_squared"] for d in result]
        assert all(ood[i] <= ood[i + 1] + 1e-4 for i in range(len(ood) - 1))
        assert mmd[-1] > mmd[0]

    def test_t_value_from_schedule(self):
        base = np.random.default_rng(9).random((500, 16)).astype(np.float32)
        epochs = self._make_epochs()
        sched = [0.0, 0.25, 0.5, 0.75, 1.0]
        result = characterise_drift(base, epochs, schedule=sched)
        for i, d in enumerate(result):
            assert d["t_value"] == pytest.approx(sched[i])



# save / load round-trip 

class TestSaveLoad:
    def _make_dataset(self, n_epochs=3, epoch_size=10, dim=4):
        base = np.random.default_rng(0).random((50, dim)).astype(np.float32)
        epochs = [np.random.default_rng(i).random((epoch_size, dim)).astype(np.float32)
                  for i in range(n_epochs)]
        gt = [np.zeros((epoch_size, 5), dtype=np.int32) for _ in range(n_epochs)]
        config = {"n_epochs": n_epochs, "dim": dim}
        diagnostics = [{"epoch_idx": i, "ood_distance": float(i)} for i in range(n_epochs)]
        return base, epochs, gt, config, diagnostics

    def test_roundtrip(self, tmp_path):
        base, epochs, gt, config, diagnostics = self._make_dataset()
        path = str(tmp_path / "dataset")
        save_drift_dataset(path, base, epochs, gt, config, diagnostics)
        loaded = load_drift_dataset(path)
        np.testing.assert_array_equal(loaded["base"], base)
        assert len(loaded["epochs"]) == 3
        for i in range(3):
            np.testing.assert_array_equal(loaded["epochs"][i], epochs[i])
        for i in range(3):
            np.testing.assert_array_equal(loaded["groundtruth"][i], gt[i])
        assert loaded["config"] == config

    def test_config_preserved(self, tmp_path):
        base, epochs, gt, config, diagnostics = self._make_dataset()
        config["nested"] = {"a": 1}
        path = str(tmp_path / "ds")
        save_drift_dataset(path, base, epochs, gt, config, diagnostics)
        loaded = load_drift_dataset(path)
        assert loaded["config"]["nested"]["a"] == 1

    def test_diagnostics_preserved(self, tmp_path):
        base, epochs, gt, config, diagnostics = self._make_dataset()
        path = str(tmp_path / "ds2")
        save_drift_dataset(path, base, epochs, gt, config, diagnostics)
        loaded = load_drift_dataset(path)
        assert len(loaded["diagnostics"]) == 3
        assert loaded["diagnostics"][1]["ood_distance"] == pytest.approx(1.0)

    def test_length_mismatch_raises(self, tmp_path):
        base, epochs, gt, config, diagnostics = self._make_dataset(n_epochs=3)
        path = str(tmp_path / "bad")
        # Manually corrupt: save with config claiming 5 epochs but only 3 stored
        config["n_epochs"] = 5
        save_drift_dataset(path, base, epochs, gt, config, diagnostics)
        # Override config file to claim 5 epochs (save already wrote n_epochs=5)
        with pytest.raises(ValueError):
            load_drift_dataset(path)


class TestNewFunctions:
    def test_drift_sequence_length(self, queries_small, A):
        _rotation_cache.clear()
        schedule = [0.0, 0.5, 1.0]
        seq = build_rotation_drift_sequence(queries_small, A, schedule, epoch_size=200)
        assert len(seq) == 3
        assert all(e.shape == (200, 128) for e in seq)

    def test_drift_sequence_reproducible(self, queries_small, A):
        _rotation_cache.clear()
        schedule = [0.0, 0.5, 1.0]
        seq1 = build_rotation_drift_sequence(queries_small, A, schedule, epoch_size=100, seed=99)
        _rotation_cache.clear()
        seq2 = build_rotation_drift_sequence(queries_small, A, schedule, epoch_size=100, seed=99)
        for e1, e2 in zip(seq1, seq2):
            np.testing.assert_array_equal(e1, e2)

    def test_groundtruth_in_bounds(self, base_small, queries_small):
        gt = compute_groundtruth(base_small, queries_small[:50], k=10)
        assert gt.shape == (50, 10)
        assert gt.min() >= 0
        assert gt.max() < len(base_small)

    def test_characterise_drift_monotone(self, base_small, queries_small, A):
        _rotation_cache.clear()
        schedule = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        epochs = build_rotation_drift_sequence(queries_small, A, schedule, epoch_size=200, seed=0)
        diagnostics = characterise_drift(base_small, epochs, schedule=schedule)
        ood = [d["ood_distance"] for d in diagnostics]
        mmd = [d["mmd_squared"] for d in diagnostics]
        assert all(ood[i] <= ood[i + 1] + 0.01 for i in range(len(ood) - 1))
        assert mmd[0] < 0.01

    def test_save_load_roundtrip(self, tmp_path, base_small, queries_small):
        n_epochs = 3
        epoch_size = 50
        epochs = [queries_small[i * epoch_size:(i + 1) * epoch_size] for i in range(n_epochs)]
        gt = [compute_groundtruth(base_small, ep, k=5) for ep in epochs]
        config = {"n_epochs": n_epochs, "epoch_size": epoch_size}
        diagnostics = [{"epoch_idx": i} for i in range(n_epochs)]
        path = str(tmp_path / "full_dataset")
        save_drift_dataset(path, base_small, epochs, gt, config, diagnostics)
        loaded = load_drift_dataset(path)
        np.testing.assert_array_equal(loaded["base"], base_small)
        assert len(loaded["epochs"]) == n_epochs
        for i in range(n_epochs):
            np.testing.assert_array_equal(loaded["epochs"][i], epochs[i])
            np.testing.assert_array_equal(loaded["groundtruth"][i], gt[i])
        assert loaded["config"]["n_epochs"] == n_epochs
