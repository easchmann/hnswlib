# Per-query entry point routing via a pool of upper-layer HNSW nodes.
#
# Pool construction: collect all level-≥1 nodes (HNSW navigation hubs) and
# select the pool_size nearest to the warmup query centroid.  Setting one of
# these as the entry point before each search lets searchKnn run a real
# upper-layer descent from the query region instead of skipping all levels.
#
# Why upper-layer nodes are required:
#   set_entry_point(layer-0 node) causes searchKnn to skip every level ≥1
#   (guard: `if level > element_levels_[currObj] continue`) and seed the
#   base-layer beam search from a node with only M₀ connections — no hub
#   structure, no long-range shortcuts → massive recall drop.
#   A level-≥1 pool node has proper upper-layer link lists so searchKnn
#   descends through those levels toward the query before hitting layer 0.
#
# NOTE: get_nodes_at_layer returns *internal* IDs.  This implementation
# assumes internal ID == external label (true when add_items uses np.arange).
#
# There is also an online mode (adapt=True in query()) that monitors
# base_layer_entry_distance and rebuilds the pool automatically on drift.
#
# Usage:
#   router = PoolRouter(raw_index, pool_size=500)
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

    # Select pool_size upper-layer (level ≥ 1) nodes nearest to the centroid
    # of warmup_queries.  Upper-layer nodes have proper link lists at level ≥ 1,
    # so set_entry_point on them lets searchKnn run real upper-layer descent.
    def build_pool(self, warmup_queries):
        warmup_queries = np.asarray(warmup_queries, dtype=np.float32)

        # instead of selecting pool nodes from layer >=1 nodes, we try starting at the highest layer
        # then go to lower layer if the pool is not full enough. -> goal to have nodes at highest possible level.
        # Assumes internal ID == external label (add_items called with np.arange).
        for min_layer in range(self.index.max_level, 0, -1):
            candidates = self.index.get_nodes_at_layer(min_layer)
            if len(candidates) >= self.pool_size//2:
                break

        # upper_internal = self.index.get_nodes_at_layer(1)
        # if not upper_internal:
        #     return 0

        upper_ids = np.array(candidates, dtype=np.int64)
        upper_vecs = np.array(self.index.get_items(upper_ids.tolist()), dtype=np.float32)

        # Score each hub by squared distance to the warmup centroid and take
        # the pool_size nearest. these are the hubs closest to the query region.
        centroid = warmup_queries.mean(axis=0).astype(np.float32)
        sq = ((upper_vecs - centroid) ** 2).sum(axis=1)

        n_select = min(self.pool_size, len(upper_ids))
        if n_select < len(upper_ids):
            sel = np.argpartition(sq, n_select)[:n_select]
        else:
            sel = np.arange(len(upper_ids))

        self.pool_ids = upper_ids[sel]
        self.pool_vecs = upper_vecs[sel]
        return int(len(self.pool_ids))


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
