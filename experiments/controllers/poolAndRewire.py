import time
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
                 alpha=1.1,
                 max_layer=3,
                 queries_per_rewire=10,
                 cooldown=100,
                 # entry pool
                 max_pool_size=200):

        self.index              = index
        self.alpha              = alpha
        self.max_layer          = max_layer
        self.queries_per_rewire = queries_per_rewire
        self.cooldown           = cooldown
        self.max_pool_size      = max_pool_size

        self.reference_centroid = reference_queries.mean(axis=0).astype(np.float32)

        baseline = measure_baseline(index, reference_queries)
        self.baseline_bl_entry  = baseline
        self.bl_entry_threshold = bl_entry_threshold or baseline * 1.2
        self.centroid_threshold = centroid_threshold

        self.query_window    = []
        self.bl_entry_window = []
        self.window_size     = window_size

        self.queries_since_last_adapt = 0
        self.update_count        = 0
        self.update_log          = []
        self.total_edges_added   = 0
        self.total_adapt_time_ms = 0.0

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
    
    def _pool_evict_by_redundancy(self):
        while len(self._pool) > self.max_pool_size:
            node_ids = list(self._pool)
            vecs = self.index.get_items(node_ids) #(pool_size,dim)
            #pairwise squared L2 norm
            diff = vecs[:, None, :] - vecs[None, :, :] #(n,n,dim)
            sq_dists = np.sum(diff**2, axis=2) #(n,n)
            np.fill_diagonal(sq_dists, np.inf)
            min_dists = sq_dists.min(axis=1)
            most_redundant = node_ids[int(np.argmin(min_dists))]
            self._pool.discard(most_redundant)


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
    
    def _pool_best_2_entries(self, query):
        """Return the 2 pool nodes with the smallest squared-L2 distance to query.

        Falls back to the global index entry point when the pool is empty.
        """
        if not self._pool:
            return int(self.index.enterpoint_node)
        node_ids = list(self._pool)
        vecs     = self.index.get_items(node_ids)
        sq_dists = np.sum((vecs - query) ** 2, axis=1)
        node_id_1 = node_ids[int(np.argmin(sq_dists))]
        node_ids.remove(node_id_1)
        node_id_2 = node_ids[int(np.argmin(sq_dists))]
        return node_id_1, node_id_2


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
    

    def _build_highway(self):
        """Add directed edge from the global entry point at max_layer towards the current query
        query centroid. After k adapt steps, the entry point has k extra max layer neighbours pointing to the drifted
        region, so upper-layer descent should start routing there for fututre queries."""
        centroid = np.mean(self.query_window, axis=0).astype(np.float32)
        max_layer = self.max_layer
        max_layer_nodes = self.index.get_nodes_at_layer(max_layer)
        if len(max_layer_nodes) ==0:
            return
        
        vecs = self.index.get_items(max_layer_nodes)
        sq_dists = np.sum((vecs-centroid) **2, axis=1)
        nearest = int(max_layer_nodes[np.argmin(sq_dists)])

        ep = int(self.index.enterpoint_node)
        if (nearest==ep):
            return
        
        # add_back_edge adds nearest to ep neighbour list at max_layer- if the list is already full, it removes
        # the currently farthest neighbour only if nearest is closer so it does not disturb connectivity.
        self.index.add_back_edge(ep, nearest,max_layer)


    def _adapt(self, reason):
        if len(self.query_window) < self.queries_per_rewire:
            return

        t_adapt_start = time.perf_counter()
        recent = self.query_window[-self.queries_per_rewire:]

        # Component 1: rewire upper layers for each recent query.
        edges_this_step = 0
        t_rewire_start = time.perf_counter()
        for q in recent:
            n = self.index.rewire_for_query(q, max_layer=self.max_layer, alpha=self.alpha)
            edges_this_step += n
        t_rewire_ms = (time.perf_counter() - t_rewire_start) * 1000
        self.total_edges_added += edges_this_step

        # Component 2: build pool from recent query distribution.
        self.index.set_ef(500)
        k_pool = min(20, self.max_pool_size)
        t_pool_start = time.perf_counter()
        sample_queries = recent[-k_pool:]
        for q in sample_queries:
            cand, _ = self.index.knn_query(q.reshape(1,-1),k=1)
            new_ep = int(cand[0][0])
            self.index.promote_node(new_ep, target_layer=self.max_layer)
            self._pool_add(new_ep)
        self._pool_evict_by_redundancy()
        t_pool_ms = (time.perf_counter() - t_pool_start) * 1000

        # Component 3: max layer highway edge toward the drifted region.
        t_highway_start = time.perf_counter()
        self._build_highway()
        t_highway_ms = (time.perf_counter() - t_highway_start) * 1000

        t_adapt_ms = (time.perf_counter() - t_adapt_start) * 1000
        self.total_adapt_time_ms += t_adapt_ms

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
            "t_adapt_ms":    round(t_adapt_ms, 2),
            "t_rewire_ms":   round(t_rewire_ms, 2),
            "t_pool_ms":     round(t_pool_ms, 2),
            "t_highway_ms":  round(t_highway_ms, 2),
        })
        print(f"  [adapt] update #{self.update_count}  reason={reason}  "
              f"edges={edges_this_step}  pool={len(self._pool)}  "
              f"t_adapt={t_adapt_ms:.0f}ms")


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
        t_total_start = time.perf_counter()

        if not self._pool:
            self.index.set_ef(ef)
            labels, dists = self.index.knn_query(query.reshape(1, -1), k=k)
            stats = hnswlib.get_last_query_stats()
            stats["t_query_ms"]     = (time.perf_counter() - t_total_start) * 1000
            stats["t_pool_scan_ms"] = 0.0
            stats["t_pool_knn_ms"]  = stats["t_query_ms"]
            stats["t_orig_knn_ms"]  = 0.0
            return labels, dists, stats

        # Pool scan: find closest pool node to query.
        t0 = time.perf_counter()
        best_ep = self._pool_best_entry(query)
        t_pool_scan_ms = (time.perf_counter() - t0) * 1000

        # Pool-entry knn search.
        t0 = time.perf_counter()
        self.index.set_entry_point(best_ep)
        self.index.set_ef(ef)
        labels_pool, dists_pool = self.index.knn_query(query.reshape(1, -1), k=k)
        stats_pool = hnswlib.get_last_query_stats()
        t_pool_knn_ms = (time.perf_counter() - t0) * 1000

        # Original-entry knn search.
        t0 = time.perf_counter()
        self.index.set_entry_point(original_ep)
        labels_orig, dists_orig = self.index.knn_query(query.reshape(1, -1), k=k)
        stats_orig = hnswlib.get_last_query_stats()
        t_orig_knn_ms = (time.perf_counter() - t0) * 1000

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
        stats["t_query_ms"]     = (time.perf_counter() - t_total_start) * 1000
        stats["t_pool_scan_ms"] = t_pool_scan_ms
        stats["t_pool_knn_ms"]  = t_pool_knn_ms
        stats["t_orig_knn_ms"]  = t_orig_knn_ms
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
        t0 = time.perf_counter()
        if use_pool and controller is not None:
            labels, _, stats = controller.search_with_pool(q, k=k, ef=ef)
        else:
            labels, _ = index.knn_query(q.reshape(1, -1), k=k)
            stats = hnswlib.get_last_query_stats()
            stats["t_query_ms"]     = (time.perf_counter() - t0) * 1000
            stats["t_pool_scan_ms"] = 0.0
            stats["t_pool_knn_ms"]  = stats["t_query_ms"]
            stats["t_orig_knn_ms"]  = 0.0

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
            "t_query_ms":           float(stats["t_query_ms"]),
            "t_pool_scan_ms":       float(stats["t_pool_scan_ms"]),
            "t_pool_knn_ms":        float(stats["t_pool_knn_ms"]),
            "t_orig_knn_ms":        float(stats["t_orig_knn_ms"]),
        })

        if controller is not None:
            controller.record(q, float(stats["base_layer_entry_distance"]))
            controller.maybe_adapt()

    return results
