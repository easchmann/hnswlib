import numpy as np
import hnswlib


def measure_baseline(index, queries, ef=200, k=10):
    index.set_ef(ef)
    dists = []
    for q in queries:
        index.knn_query(q.reshape(1, -1), k=k)
        dists.append(float(hnswlib.get_last_query_stats()["base_layer_entry_distance"]))
    return float(np.mean(dists))


class PoolAndRewireController:
    """
    Two-component online adaptation controller.

    Component 1 — query-driven upper-layer rewiring:
        After drift is detected, calls rewire_for_query for each recent query,
        adding corrective edges at upper layers toward the drifted query region.

    Component 2 — entry-point pool with centroid-distance eviction:
        A bounded set of promoted nodes that serve as geometrically closer
        entry points for drifted queries.

        Pool management (Python-side):
          - self._pool is a plain set of node IDs.
          - _pool_add(node_id): no-op if already present.
          - _pool_evict_by_centroid(centroid): while pool exceeds max_pool_size,
            fetch vectors for all pool members, compute squared-L2 distance to
            the current query centroid, and evict the farthest node.
            Rationale: a node far from the centroid is unlikely to be the
            closest pool member for any upcoming query, so it wastes a slot.
            This is proactive (acts immediately on geometry) vs. time-based LRU
            which is reactive (requires query traffic to reveal staleness).
          - _pool_best_entry(query): scan pool vectors, return the node ID
            with the smallest squared-L2 distance to query.

    Drift is detected via a sliding window on base_layer_entry_distance.
    """

    def __init__(self, index, reference_queries,
                 # drift detection
                 window_size=200,
                 bl_entry_threshold=None,   # absolute; default = 1.5x baseline
                 centroid_threshold=None,   # L2 displacement; None = disabled
                 # rewiring
                 alpha=1.5,
                 max_layer=3,
                 queries_per_rewire=10,
                 cooldown=100,
                 # entry pool
                 max_pool_size=20):

        self.index              = index
        self.alpha              = alpha
        self.max_layer          = max_layer
        self.queries_per_rewire = queries_per_rewire
        self.cooldown           = cooldown
        self.max_pool_size      = max_pool_size

        self.reference_centroid = reference_queries.mean(axis=0).astype(np.float32)

        baseline = measure_baseline(index, reference_queries)
        self.baseline_bl_entry  = baseline
        self.bl_entry_threshold = bl_entry_threshold or baseline * 1.5
        self.centroid_threshold = centroid_threshold

        self.query_window    = []
        self.bl_entry_window = []
        self.window_size     = window_size

        self.queries_since_last_adapt = 0
        self.update_count      = 0
        self.update_log        = []
        self.total_edges_added = 0

        # Python-side pool: set of node IDs currently active as entry points.
        # Eviction is centroid-distance-based (see _pool_evict_by_centroid).
        self._pool = set()


    # pool management:centroid-distance eviction

    def _pool_add(self, node_id):
        """Add node_id to pool. No-op if already present."""
        self._pool.add(node_id)

    def _pool_evict_by_centroid(self, centroid):
        """Evict the pool node farthest from centroid until len <= max_pool_size.

        Fetch all pool vectors in one batched get_items call (cheap), compute squared-L2 distances to centroid, remove the farthest node.  
        Repeat until the pool is within budget.
        """
        while len(self._pool) > self.max_pool_size:
            node_ids = list(self._pool)
            vecs     = self.index.get_items(node_ids) # (pool_size, dim)
            sq_dists = np.sum((vecs - centroid) ** 2, axis=1)
            farthest = node_ids[int(np.argmax(sq_dists))]
            self._pool.discard(farthest)

    def _pool_best_entry(self, query):
        """Return the pool node with the smallest squared-L2 distance to query.

        Falls back to the global index entry point when the pool is empty.
        """
        if not self._pool:
            return int(self.index.enterpoint_node)
        node_ids = list(self._pool)
        vecs     = self.index.get_items(node_ids)
        sq_dists = np.sum((vecs - query) ** 2, axis=1)
        return node_ids[int(np.argmin(sq_dists))]


    def record(self, query_vec, bl_entry_dist):
        self.query_window.append(query_vec.astype(np.float32))
        self.bl_entry_window.append(float(bl_entry_dist))
        self.queries_since_last_adapt += 1
        if len(self.query_window) > self.window_size:
            self.query_window.pop(0)
            self.bl_entry_window.pop(0)

    def _drift_detected(self):
        if len(self.query_window) < self.window_size:
            return False, "window not full"

        mean_bl = float(np.mean(self.bl_entry_window))
        if mean_bl > self.bl_entry_threshold:
            return True, f"bl_entry {mean_bl:.2f} > {self.bl_entry_threshold:.2f}"

        if self.centroid_threshold is not None:
            disp = float(np.linalg.norm(
                np.mean(self.query_window, axis=0) - self.reference_centroid))
            if disp > self.centroid_threshold:
                return True, f"centroid_disp {disp:.2f} > {self.centroid_threshold:.2f}"

        return False, "no drift"

    def maybe_adapt(self):
        if self.queries_since_last_adapt < self.cooldown:
            return False
        detected, reason = self._drift_detected()
        if not detected:
            return False
        self._adapt(reason)
        return True


    def _adapt(self, reason):
        if len(self.query_window) < self.queries_per_rewire:
            return

        recent = self.query_window[-self.queries_per_rewire:]

        # Component 1: rewire upper layers for each recent query.
        edges_this_step = 0
        for q in recent:
            n = self.index.rewire_for_query(q, max_layer=self.max_layer, alpha=self.alpha)
            edges_this_step += n
        self.total_edges_added += edges_this_step

        # Component 2: add k_pool=5 nodes nearest to the full-window centroid.
        # Querying k=5 instead of k=1 ensures the pool grows on every step even when the single nearest centroid node is already present.
        # Each candidate is promoted to max_level so it can serve as a top-level entry point (same layer as the original global entry point).
        # After adding, evict pool nodes farthest from the centroid.
        centroid = np.mean(self.query_window, axis=0).astype(np.float32)
        self.index.set_ef(50)
        k_pool = min(5, self.max_pool_size)
        cand_labels, _ = self.index.knn_query(centroid.reshape(1, -1), k=k_pool)
        for new_ep in cand_labels[0]:
            new_ep = int(new_ep)
            self.index.promote_node(new_ep, target_layer=self.index.max_level)
            self._pool_add(new_ep)
        self._pool_evict_by_centroid(centroid)

        self.update_count += 1
        self.queries_since_last_adapt = 0
        self.update_log.append({
            "update_n":      self.update_count,
            "reason":        reason,
            "edges_added":   edges_this_step,
            "total_edges":   self.total_edges_added,
            "pool_size":     len(self._pool),
            "mean_bl_entry": float(np.mean(self.bl_entry_window)),
            "bl_threshold":  self.bl_entry_threshold,
        })
        print(f"  [adapt] update #{self.update_count}  reason={reason}  "
              f"edges={edges_this_step}  pool={len(self._pool)}")


    def search_with_pool(self, query, k, ef):
        """Merged search: pool entry point + original global entry point.

        1. _pool_best_entry selects the pool member closest to query.
        2. Run knn_query from that pool node.
        3. Run knn_query from the original global entry point.
        4. Return the k closest candidates from the UNION of both result sets.

        Merging guarantees recall >= no_adaptation: the original search's
        results are always in the candidate set.
        Fast path (single search) when the pool is still empty.
        """
        query = np.asarray(query, dtype=np.float32).ravel()
        original_ep = int(self.index.enterpoint_node)

        if not self._pool:
            self.index.set_ef(ef)
            labels, dists = self.index.knn_query(query.reshape(1, -1), k=k)
            stats = hnswlib.get_last_query_stats()
            return labels, dists, stats

        # Pool search
        best_ep = self._pool_best_entry(query)
        self.index.set_entry_point(best_ep)
        self.index.set_ef(ef)
        labels_pool, dists_pool = self.index.knn_query(query.reshape(1, -1), k=k)
        stats_pool = hnswlib.get_last_query_stats()

        # Original-entry search (restores global entry point first)
        self.index.set_entry_point(original_ep)
        labels_orig, dists_orig = self.index.knn_query(query.reshape(1, -1), k=k)
        stats_orig = hnswlib.get_last_query_stats()

        # Merge: union of both candidate sets, keep k closest by distance.
        merged = {}
        for lab, dist in zip(labels_orig[0], dists_orig[0]):
            merged[int(lab)] = float(dist)
        for lab, dist in zip(labels_pool[0], dists_pool[0]):
            lab_i, dist_f = int(lab), float(dist)
            if lab_i not in merged or dist_f < merged[lab_i]:
                merged[lab_i] = dist_f

        sorted_items = sorted(merged.items(), key=lambda x: x[1])[:k]
        labels_merged = np.array([[item[0] for item in sorted_items]])
        dists_merged  = np.array([[item[1] for item in sorted_items]], dtype=np.float32)
        stats = (stats_pool
                 if stats_pool["base_layer_entry_distance"] <= stats_orig["base_layer_entry_distance"]
                 else stats_orig)
        return labels_merged, dists_merged, stats


def run_query_batch(index, queries, gt, k, ef, controller=None, use_pool=False):
    """
    Run a batch of queries, collecting per-query recall and search stats.

    controller: PoolAndRewireController — if provided, feeds bl_entry_dist
                into the controller and calls maybe_adapt() after each query.
    use_pool:   if True, call controller.search_with_pool;
                if False, use standard knn_query.
    """
    index.set_ef(ef)
    results = []

    for i, q in enumerate(queries):
        if use_pool and controller is not None:
            labels, _, stats = controller.search_with_pool(q, k=k, ef=ef)
        else:
            labels, _ = index.knn_query(q.reshape(1, -1), k=k)
            stats = hnswlib.get_last_query_stats()

        results.append({
            "recall":               len(set(labels[0]) & set(gt[i])) / k,
            "ep_dist":              float(stats["entry_point_distance"]),
            "bl_entry_dist":        float(stats["base_layer_entry_distance"]),
            "ul_dist_comps":        int(stats["upper_layer_distance_computations"]),
            "layer_visits":         list(stats["layer_visit_counts"]),
            "base_visited":         int(stats["base_layer_visited_count"]),
            "base_dist_comps":      int(stats["base_layer_distance_computations"]),
            "candidates_remaining": int(stats["candidates_remaining_at_termination"]),
            "lb_trace":             list(stats["lowerbound_trace"]),
        })

        if controller is not None:
            controller.record(q, float(stats["base_layer_entry_distance"]))
            controller.maybe_adapt()

    return results
