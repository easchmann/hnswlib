"""Drift detection: Escape Hardness (EH), sliding window buffer, spatial cells."""

import time

import faiss
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.cluster import MiniBatchKMeans


def compute_eh(result_vectors):
    """Mean pairwise L2 distance among top-k result vectors.
    Caller must pass base[result_ids] — hnswlib returns IDs, not vectors.
    """
    k = result_vectors.shape[0]
    if k <= 1:
        return 0.0
    return float(np.mean(pdist(result_vectors, metric="euclidean")))


def compute_eh_batch(result_vectors_batch):
    """EH for a full epoch. Each element must be base[result_ids] for one query."""
    return np.array([compute_eh(rv) for rv in result_vectors_batch])


class SlidingWindowBuffer:
    """Circular buffer of EH values with a spatial cell histogram."""

    def __init__(self, window_size, n_cells):
        self._window_size = window_size
        self._buffer = np.zeros(window_size, dtype=np.float64)
        self._cell_histogram = np.zeros(n_cells, dtype=np.int64)
        self._head = 0
        self._n_obs = 0

    def update(self, eh_value, cell_id):
        self._buffer[self._head] = eh_value
        self._cell_histogram[cell_id] += 1
        self._head = (self._head + 1) % self._window_size
        self._n_obs += 1

    def update_batch(self, eh_values, cell_ids):
        for eh, cell in zip(eh_values, cell_ids):
            self.update(float(eh), int(cell))

    def get_eh_values(self):
        if self._n_obs < self._window_size:
            return self._buffer[: self._n_obs].copy()
        return np.roll(self._buffer, -self._head).copy()

    def get_cell_histogram(self):
        return self._cell_histogram.copy()

    def reset_histogram(self):
        self._cell_histogram[:] = 0

    def is_ready(self):
        return self._n_obs >= self._window_size // 2 + 1

    @property
    def n_observations(self):
        return self._n_obs


def build_spatial_index(base, n_cells=100, seed=42):
    """K-means on base vectors; returns (centroids, labels)."""
    n_base = base.shape[0]
    print(f"Building spatial index: {n_cells} cells on {n_base} vectors...", end=" ", flush=True)
    t0 = time.time()
    km = MiniBatchKMeans(n_clusters=n_cells, random_state=seed, n_init=3)
    km.fit(base)
    print(f"done in {time.time() - t0:.1f}s")
    return km.cluster_centers_.astype(np.float32), km.labels_.astype(np.int32)


def assign_cell(query_vectors, centroids):
    """Nearest centroid for each query. Uses faiss when n_queries > 100."""
    if query_vectors.shape[0] > 100:
        index = faiss.IndexFlatL2(centroids.shape[1])
        index.add(centroids.astype(np.float32))
        _, cell_ids = index.search(query_vectors.astype(np.float32), 1)
        return cell_ids.ravel().astype(np.int32)
    diff = query_vectors[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    return np.argmin(np.sum(diff ** 2, axis=-1), axis=1).astype(np.int32)


class RFFKernel:
    """Gaussian RFF kernel for scalar EH values."""

    def __init__(self, sigma, n_features=256, seed=0):
        rng = np.random.default_rng(seed)
        self.n_features = n_features
        self._omega = rng.normal(0, 1.0 / sigma, size=n_features)
        self._b = rng.uniform(0, 2 * np.pi, size=n_features)

    def transform(self, x):
        """x: 1D array of scalars. Returns shape (n, n_features)."""
        z = np.sqrt(2.0 / self.n_features) * np.cos(
            np.outer(x, self._omega) + self._b
        )
        return z

    def mean_embedding(self, x):
        """Mean of transform(x) over the sample dimension. Shape (n_features,)."""
        return self.transform(x).mean(axis=0)


def compute_mmd_squared(reference, current, kernel):
    """MMD² estimate between two 1D arrays of EH values via RFF mean embeddings."""
    diff = kernel.mean_embedding(reference) - kernel.mean_embedding(current)
    return float(np.dot(diff, diff))


class DriftDetector:
    """MMD²-based drift detector on EH values with spatial cell localisation."""

    def __init__(self, window_size, n_cells, n_rff=256, rff_seed=0):
        self._buffer = SlidingWindowBuffer(window_size, n_cells)
        self._n_rff = n_rff
        self._rff_seed = rff_seed
        self.kernel = None
        self.reference_embedding = None
        self.threshold = None
        self.reference_cell_histogram = None
        self.hot_cell_lambda = 2.0
        self._sigma = None

    def calibrate(self, reference_eh, reference_cell_ids, n_null_samples=200, seed=42):
        """Calibrate on reference-distribution EH values. Call once after index construction."""
        rng = np.random.default_rng(seed)

        # Estimate bandwidth via median heuristic
        n = len(reference_eh)
        n_pairs = min(500, n * (n - 1) // 2)
        idx = rng.integers(0, n, size=(n_pairs, 2))
        diffs = np.abs(reference_eh[idx[:, 0]] - reference_eh[idx[:, 1]])
        sigma = float(np.median(diffs))
        if sigma < 1e-8:
            print("Warning: sigma near zero, setting sigma=1.0")
            sigma = 1.0

        self._sigma = sigma
        self.kernel = RFFKernel(sigma, self._n_rff, self._rff_seed)
        self.reference_embedding = self.kernel.mean_embedding(reference_eh)

        # Null distribution: shuffle + split into halves
        null_mmds = []
        for _ in range(n_null_samples):
            perm = rng.permutation(reference_eh)
            half = len(perm) // 2
            null_mmds.append(compute_mmd_squared(perm[:half], perm[half:], self.kernel))
        self.threshold = float(np.percentile(null_mmds, 95))

        # Reference cell distribution (normalised)
        counts = np.bincount(reference_cell_ids, minlength=self._buffer._cell_histogram.shape[0])
        self.reference_cell_histogram = counts / (counts.sum() + 1e-10)

        # Seed buffer so it starts full
        self._buffer.update_batch(reference_eh, reference_cell_ids)

        stats = {
            "sigma": sigma,
            "threshold": self.threshold,
            "n_reference_queries": len(reference_eh),
            "reference_eh_mean": float(reference_eh.mean()),
            "reference_eh_std": float(reference_eh.std()),
        }
        print(
            f"Calibration complete | sigma={sigma:.4f} | threshold={self.threshold:.6f} | "
            f"n={len(reference_eh)} | EH mean={stats['reference_eh_mean']:.4f} ± {stats['reference_eh_std']:.4f}"
        )
        return stats

    def update(self, eh_value, cell_id):
        self._buffer.update(eh_value, cell_id)

    def update_batch(self, eh_values, cell_ids):
        self._buffer.update_batch(eh_values, cell_ids)

    def check_drift(self):
        """Compare current window against reference. Returns None if buffer not ready."""
        if not self._buffer.is_ready():
            return None

        current_eh = self._buffer.get_eh_values()
        current_embedding = self.kernel.mean_embedding(current_eh)
        diff = self.reference_embedding - current_embedding
        mmd_sq = float(np.dot(diff, diff))
        drift_detected = mmd_sq > self.threshold

        hot_cells = []
        hot_ratios = []
        if drift_detected:
            current_hist = self._buffer.get_cell_histogram().astype(float)
            current_hist_norm = current_hist / (current_hist.sum() + 1e-10)
            ratios = current_hist_norm / (self.reference_cell_histogram + 1e-10)
            hot_idx = np.where(ratios > self.hot_cell_lambda)[0]
            order = np.argsort(ratios[hot_idx])[::-1]
            hot_cells = hot_idx[order].tolist()
            hot_ratios = ratios[hot_idx[order]].tolist()

        self._buffer.reset_histogram()

        return {
            "drift_detected": drift_detected,
            "mmd_squared": mmd_sq,
            "threshold": self.threshold,
            "hot_cells": hot_cells,
            "hot_cell_ratios": hot_ratios,
            "n_observations": self._buffer.n_observations,
        }

    def recalibrate(self, current_eh, current_cell_ids, epoch=None, n_null_samples=100, seed=42):
        """Reset reference to current distribution; recompute threshold cheaply."""
        current_eh = np.asarray(current_eh, dtype=np.float64)
        current_cell_ids = np.asarray(current_cell_ids, dtype=np.int32)
        rng = np.random.default_rng(seed)

        self.reference_embedding = self.kernel.mean_embedding(current_eh)

        n_cells = self._buffer._cell_histogram.shape[0]
        counts = np.bincount(current_cell_ids, minlength=n_cells)
        self.reference_cell_histogram = counts / (counts.sum() + 1e-10)

        null_mmds = []
        for _ in range(n_null_samples):
            perm = rng.permutation(current_eh)
            half = len(perm) // 2
            null_mmds.append(compute_mmd_squared(perm[:half], perm[half:], self.kernel))
        self.threshold = float(np.percentile(null_mmds, 95))

        # Drop stale pre-recalibration observations
        window_size = self._buffer._window_size
        self._buffer = SlidingWindowBuffer(window_size, n_cells)
        self._buffer.update_batch(current_eh, current_cell_ids)

        epoch_str = f"epoch {epoch}" if epoch is not None else "?"
        print(
            f"Recalibrated at {epoch_str} | new EH mean={float(current_eh.mean()):.4f} | "
            f"new threshold={self.threshold:.6f}"
        )

        return {
            "sigma": self._sigma,
            "threshold": self.threshold,
            "n_reference_queries": len(current_eh),
            "reference_eh_mean": float(current_eh.mean()),
            "reference_eh_std": float(current_eh.std()),
        }

    def is_calibrated(self):
        return self.threshold is not None
