# Per-query entry point routing via a pool of representative index nodes.
# This approach maintains a pool of M nodes that cover the current query region. 
# Before each query we do a brute-force nearest-neighbour lookup over the pool (O(M x dim) and use the closest pool node as the entry point. 
# Since it's almost always a layer-0 node, HNSW skips all upper-layer traversal and base-layer search
# starts directly from a node close to the query.
#
# Pool construction: run warmup queries with high ef and collect the returned nearest-neighbour nodes. 
# After dedup these are exactly the data nodes that live near the current query distribution, so they're the right starting
# points for future queries from the same distribution.
#
# There's also an online mode (adapt=True in query()) that monitors base_layer_entry_distance and rebuilds the pool automatically when drift
# is detected, same sliding-window trigger as the other strategies.
#
# Usage:
#   router = PoolRouter(raw_index, pool_size=500, ef_build=2000)
#   router.build_pool(warmup_queries)
#   labels, dists = router.query(q, k=10, ef=50)

from __future__ import annotations

import numpy as np
import hnswlib
from collections import deque


class PoolRouter:

    def __init__(
        self,
        index: hnswlib.Index,
        pool_size: int = 500,
        ef_build: int = 2000,
        k_build: int = 5,
        window_size: int = 200,
        threshold_factor: float = 1.5,
        query_buffer_size: int = 500,
    ):
        self.index = index
        self.pool_size = pool_size
        self.ef_build = ef_build
        self.k_build = k_build
        self.window_size = window_size
        self.threshold_factor = threshold_factor
        self.query_buffer_size = query_buffer_size

        # save so we can reset before pool searches
        self.original_ep = index.enterpoint_node

        self.pool_ids = None # np.ndarray shape (M,), int64
        self.pool_vecs = None  # np.ndarray shape (M, dim), float32

        # drift detection
        self.baseline = None
        self.threshold = None
        self.window = deque(maxlen=window_size)
        self.query_buffer = deque(maxlen=query_buffer_size)

        self.adaptation_log = []

    # setup

    def calibrate(self, queries, ef=200, k=10):
        queries = np.asarray(queries, dtype=np.float32)
        self.index.set_ef(ef)
        dists = []
        for q in queries:
            self.index.knn_query(q.reshape(1, -1), k=k)
            dists.append(float(hnswlib.get_last_query_stats()["base_layer_entry_distance"]))
        self.baseline = float(np.mean(dists))
        self.threshold = self.baseline * self.threshold_factor
        self.window.clear()
        return self.baseline

    # Run warmup_queries with high ef and collect the returned NN nodes as pool candidates. Each unique node gets added until we hit pool_size.
    # We reset to original_ep before each search so the pool is always seeded from a consistent, known-good position regardless of what routing calls
    # might have changed the global EP to in between.
    def build_pool(self, warmup_queries):
        warmup_queries = np.asarray(warmup_queries, dtype=np.float32)
        saved_ef = self.index.ef
        self.index.set_ef(self.ef_build)

        seen = set()
        candidates = []

        for q in warmup_queries:
            # always start from the original EP so pool searches are consistent
            self.index.set_entry_point(self.original_ep)
            labels, _ = self.index.knn_query(q.reshape(1, -1), k=self.k_build)
            for nid in labels[0].tolist():
                if nid not in seen:
                    seen.add(nid)
                    candidates.append(nid)
            if len(candidates) >= self.pool_size:
                break

        self.index.set_ef(saved_ef)

        if not candidates:
            return 0

        self.pool_ids = np.array(candidates[:self.pool_size], dtype=np.int64)
        self.pool_vecs = np.array(self.index.get_items(self.pool_ids.tolist()), dtype=np.float32)
        return len(self.pool_ids)


    # querying
    # Route q to the nearest pool node and search from there.
    # If adapt=True, also monitor bl_entry_dist and rebuild the pool on drift.
    def query(self, q, k, ef, adapt=False):
        q = np.asarray(q, dtype=np.float32).ravel()

        if self.pool_ids is not None:
            sq = ((self.pool_vecs - q) ** 2).sum(axis=1)
            best_node = int(self.pool_ids[int(np.argmin(sq))])
            self.index.set_entry_point(best_node)

        self.index.set_ef(ef)
        labels, dists = self.index.knn_query(q.reshape(1, -1), k=k)

        if adapt and self.threshold is not None:
            stats = hnswlib.get_last_query_stats()
            bl = float(stats["base_layer_entry_distance"])
            self.window.append(bl)
            self.query_buffer.append(q.copy())
            if self._drift_detected():
                self._rebuild()

        return labels, dists


    # drift detection / online adaptation

    def _drift_detected(self):
        if len(self.window) < self.window_size:
            return False
        return float(np.mean(self.window)) > self.threshold

    def window_mean(self):
        if self.window:
            return float(np.mean(self.window))
        return float("nan")

    # rebuild pool from the most recent query buffer
    def _rebuild(self):
        buffer = np.array(self.query_buffer, dtype=np.float32)
        wm = self.window_mean()

        n = self.build_pool(buffer)

        entry = {
            "adaptation_n": len(self.adaptation_log) + 1,
            "pool_size_achieved": n,
            "window_mean_before": wm,
            "threshold": self.threshold,
        }
        self.adaptation_log.append(entry)
        print(
            f"[PoolRouter] rebuild #{entry['adaptation_n']}  "
            f"pool_size={n}  window_mean={wm:.1f}  threshold={self.threshold:.1f}"
        )
        self.window.clear()
