# MultiEPAdaptiveHNSW -- drift-aware HNSW wrapper with a growing entry-point pool.

# Why single-centroid adaptation fails for multimodal data
# AdaptiveHNSW computes the centroid of buffered queries and promotes the data node nearest to that centroid as the new global entry point.  
# For a unimodal (single Gaussian) distribution this works: the centroid is near real data and
# the promoted node gives the greedy descent a good starting position.

# For a mixture-of-Gaussians distribution the centroid lies in empty space between cluster centres.  
# The node nearest the centroid is from some random boundary cluster and not the home cluster of any individual query. 
# Setting it as EP starts every query far from its true nearest neighbours, raising bl_entry_dist above the no-adaptation baseline and hurting recall.

# Strategy
# 1.Drift detection: same sliding window on base_layer_entry_distance.
# 2. On drift: build an entry-point pool:
# For each query q in the buffer, run knn_query(q, k=1, ef=ef_seed) to find the actual nearest data node to q.
# With ef_seed >> ef_query this should succeeds also under drift. 
# Deduplicate across queries (many queries from the same cluster map to the same node), promote each unique result to max_level, and add it to the pool. 
# After one adaptation the pool contains one representative per observed cluster in the buffer.

# 3.At query time: route each query to its nearest pool member:
# Compute sq-L2 from the query to every pooled EP vector (numpy array). 
# Call set_entry_point(nearest_pool_member) then run the knn_query. 
# The greedy upper-layer descent now begins inside the query's home cluster, so bl_entry_dist should drop to roughly the
# within-cluster distance and recall should be better also at lower ef.

# Routing cost
# O(pool_size x dim) per query for the nearest-EP distance computation.
# With pool_size <= query_buffer_size (default 500) and dim = 128 this is <= 64k float multiplications
# negligible relative to the HNSW search


from __future__ import annotations

import numpy as np
import hnswlib
from collections import deque
from typing import Optional


class MultiEPAdaptiveHNSW:
    def __init__(self, index,*, window_size=200, threshold_factor=1.5, ef_seed=2000, query_buffer_size=500):
        self.index = index
        self.window_size = window_size
        self.threshold_factor = threshold_factor
        self.ef_seed = ef_seed
        self.query_buffer_size = query_buffer_size

        self.baseline = None
        self.threshold = None
        self.window = deque(maxlen=window_size)
        self.query_buffer = deque(maxlen=query_buffer_size)

        self.original_ep = index.enterpoint_node
        self.extra_eps = []

        # Cached after adaptation for O(pool x dim) per-query routing. None means no routing (pool not yet built).
        self.ep_vecs = None
        self.ep_ids = None

        self.adaptation_log = []


      # run queries and record mean base layer entry distance as baseline
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


    # Query the index. When the entry-point pool is active (after the first adaptation event), each query is routed to its nearest pool member
    # before the HNSW search runs regardless of the `adapt` flag.
    def knn_query(self, queries, k=10, ef=None, adapt=True):
        queries = np.asarray(queries, dtype=np.float32)
        single = queries.ndim == 1
        if single:
            queries = queries.reshape(1, -1)

        if ef is not None:
            self.index.set_ef(ef)

        all_labels = []
        all_dists = []

        for q in queries:
            # Route to the nearest entry point in the pool.
            if self.ep_vecs is not None:
                sq_dists = ((self.ep_vecs - q) ** 2).sum(axis=1)
                best_idx = int(np.argmin(sq_dists))
                best_ep = self.ep_ids[best_idx]
                if best_ep != self.index.enterpoint_node:
                    self.index.set_entry_point(best_ep)

            labels, dists = self.index.knn_query(q.reshape(1, -1), k=k)
            all_labels.append(labels[0])
            all_dists.append(dists[0])

            if adapt and self.threshold is not None:
                stats = hnswlib.get_last_query_stats()
                bl_entry = float(stats["base_layer_entry_distance"])
                self.window.append(bl_entry)
                self.query_buffer.append(q.copy())
                if self.drift_detected():
                    self.adapt()

        labels_out = np.array(all_labels)
        dists_out = np.array(all_dists)
        if single:
            return labels_out[:1], dists_out[:1]
        return labels_out, dists_out


    # Current mean of the drift-detection window
    def window_mean(self):
        if self.window:
            return float(np.mean(self.window))
        else: 
            return float("nan")

    def drift_detected(self):
        if len(self.window) < self.window_size:
            return False
        return self.window_mean() > self.threshold



    def adapt(self):
        buffer = np.array(self.query_buffer, dtype=np.float32)
        max_level = self.index.max_level

        # Run a high-ef knn search for each buffered query to find the nearest existing data node.
        # Each result is a cluster representative: individual queries actually lie near real data points, so the found node is in the right cluster.
        saved_ef = self.index.ef
        self.index.set_ef(self.ef_seed)

        already_seen: set = set(self.extra_eps)
        already_seen.add(self.original_ep)
        new_eps = []

        for seed in buffer:
            labels, _ = self.index.knn_query(seed.reshape(1, -1), k=1)
            node = int(labels[0][0])
            if node not in already_seen:
                already_seen.add(node)
                new_eps.append(node)

        self.index.set_ef(saved_ef)

        # Promote each new node to max_level so searchKnn runs a full
        # upper-layer descent from it (not skip all levels as for a L0 node).
        for node in new_eps:
            self.index.promote_node(node, max_level)

        self.extra_eps.extend(new_eps)

        # Cache all EP vectors for fast per-query routing.
        all_eps = [self.original_ep] + self.extra_eps
        self.ep_vecs = np.array(self.index.get_items(all_eps), dtype=np.float32)
        self.ep_ids = all_eps

        entry = {
            "adaptation_n": len(self.adaptation_log) + 1,
            "n_new_eps": len(new_eps),
            "total_eps": len(all_eps),
            "max_level": max_level,
            "window_mean_before": self.window_mean(),
            "threshold": self.threshold,
        }
        self.adaptation_log.append(entry)
        print(
            f"[MultiEPAdaptiveHNSW] adaptation #{entry['adaptation_n']}  "
            f"new_eps={len(new_eps)}  pool_size={len(all_eps)}  "
            f"window_mean={entry['window_mean_before']:.1f}  "
            f"threshold={self.threshold:.1f}"
        )

        self.window.clear()
