import numpy as np
import hnswlib


def measure_baseline(index, queries, ef=200, k=10):
    index.set_ef(ef)
    dists = []
    for q in queries:
        index.knn_query(q.reshape(1, -1), k=k)
        dists.append(float(hnswlib.get_last_query_stats()["base_layer_entry_distance"]))
    return float(np.mean(dists))




    # Two-component online adaptation controller.

    # Component 1: query-driven rewiring (rewire_for_query):
    # Runs after drift is detected.  
    # For each recent query, calls
    #   index.rewire_for_query(q, max_layer, alpha) which identifies every
    #   upper-layer node where greedy descent chose a suboptimal neighbour
    #   relative to q, and adds a corrective edge toward the beam-search
    #   winner at that layer.  This directly repairs the structural failure
    #   (no edge pointing toward the shifted query region) without touching
    #   the base layer.

    # Component 2: entry-point pool:
    #   On each adaptation step the nearest data node to the current query
    #   centroid is promoted to layer >= 1 (so it becomes a navigation hub)
    #   and added to the pool.  At query time search_with_pool selects the
    #   closest pool member and sets it as the HNSW entry point before
    #   calling knn_query, giving the search a geometrically good start in
    #   the correct region.  LRU eviction keeps the pool bounded.

    # Drift is detected via a sliding window on base_layer_entry_distance
    # (the distance from the fixed entry point to the node where upper-layer
    # descent terminates).  This distance grows when the entry point becomes
    # irrelevant for the shifted queries.
class PoolAndRewireController:

    def __init__(self, index, reference_queries,
                 # drift detection
                 window_size=200,
                 bl_entry_threshold=None,   # absolute value; default = 1.5x baseline
                 centroid_threshold=None,   # L2 displacement; None = disabled
                 # rewiring
                 alpha=1.5,
                 max_layer=1,
                 queries_per_rewire=10,
                 cooldown=100,
                 # entry pool
                 max_pool_size=5):

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

        # Component 1: rewire upper layers for each recent query
        edges_this_step = 0
        for q in recent:
            n = self.index.rewire_for_query(q, max_layer=self.max_layer, alpha=self.alpha)
            edges_this_step += n
        self.total_edges_added += edges_this_step

        # Component 2: add nearest node to current centroid to the entry pool.
        # Promote to layer >= 1 so set_entry_point on it enables upper-layer
        # descent rather than skipping all layers.
        centroid = np.mean(recent, axis=0).astype(np.float32)
        self.index.set_ef(50)
        labels, _ = self.index.knn_query(centroid.reshape(1, -1), k=1)
        new_ep = int(labels[0][0])

        # CHANGE FROM PREVIOUS VERSION: PROMOTE TO HIGHER LAYER
        target_layer = max(3, self.index.max_layer()-1)

        self.index.promote_node(new_ep, target_layer=target_layer)

        self.index.add_to_entry_pool(new_ep)
        self.index.prune_entry_pool(self.max_pool_size)

        self.update_count += 1
        self.queries_since_last_adapt = 0
        self.update_log.append({
            "update_n":       self.update_count,
            "reason":         reason,
            "edges_added":    edges_this_step,
            "total_edges":    self.total_edges_added,
            "pool_size":      self.index.entry_pool_size(),
            "new_ep":         new_ep,
            "mean_bl_entry":  float(np.mean(self.bl_entry_window)),
            "bl_threshold":   self.bl_entry_threshold,
        })
        print(f"  [adapt] update #{self.update_count}  reason={reason}  "
              f"edges={edges_this_step}  pool={self.index.entry_pool_size()}")


def search_with_pool(index, query, k, ef):
    """Select best pool entry point, set it, run knn_query, return labels/dists/stats."""
    query = np.asarray(query, dtype=np.float32).ravel()
    best_ep = index.get_best_entry_point(query)
    index.set_entry_point(best_ep)
    index.set_ef(ef)
    labels_pool, dists_pool = index.knn_query(query.reshape(1, -1), k=k)
    stats_pool = hnswlib.get_last_query_stats()

    # also try original entry point

    # restore original entry point
    index.reset_entry_point()
    labels_original, dists_original = index.knn_query(query.reshape(1, -1), k=k)
    stats_original = hnswlib.get_last_query_stats()

    # choose whichever has lower base layer entry distance
    if stats_pool["base_layer_entry_distance"] < stats_original["base_layer_entry_dsitance"]:
        return labels_pool, dists_pool, stats_pool
    else: 
        return labels_original, dists_original, stats_original


def run_query_batch(index, queries, gt, k, ef, controller=None, use_pool=False):
    """
    Run a batch of queries, collecting per-query recall and search stats.

    controller: ThesisAdaptationController — if provided, feeds bl_entry_dist
                into the controller and calls maybe_adapt() after each query.
    use_pool:   if True, use search_with_pool (pool-based entry point routing);
                if False, use standard knn_query.
    """
    index.set_ef(ef)
    results = []

    for i, q in enumerate(queries):
        if use_pool:
            labels, _, stats = search_with_pool(index, q, k=k, ef=ef)
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
