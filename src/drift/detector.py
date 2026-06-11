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
