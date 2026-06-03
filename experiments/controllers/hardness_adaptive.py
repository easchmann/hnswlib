import time
import numpy as np
import hnswlib


def _measure_dist_comps(index, queries, ef, k):
    """Run queries at given ef and return array of base_layer_distance_computations."""
    index.set_ef(ef)
    comps = []
    for q in queries:
        index.knn_query(q.reshape(1, -1), k=k)
        comps.append(int(hnswlib.get_last_query_stats()["base_layer_distance_computations"]))
    return np.array(comps, dtype=np.float32)


class HardnessAdaptiveController:
    """
    Adaptation controller for hardness-based distribution drift.

    Compared to PoolAndRewireController (which triggers on aggregate bl_entry_distance),
    this controller acts on per-query difficulty measured by base_layer_distance_computations.
    A query is "hard" when its dist_comps exceeds a threshold set from warmup statistics.

    Three mechanisms (each independently toggleable):

    1. Hard-query pool: pool entries are nearest index nodes of hard queries,
       concentrating entry points near structurally difficult regions.
       Eviction removes the pool node farthest from the current hard query
       so the pool stays focused on the active hard region.

    2. Difficulty-triggered rewiring: rewire immediately after a hard query
       (with a short cooldown counted in hard queries), rather than waiting for
       global drift detection and batch rewiring.

    3. Adaptive ef escalation: when the rolling fraction of recent hard queries
       exceeds escalation_trigger, boost ef by escalation_factor for the next
       query. hardness drift is monotone so the history predicts the next query's difficulty well.

    Search uses a union-merge of pool entry point + global entry point at the
    chosen ef, guaranteeing recall >= no-adaptation baseline.
    """

    def __init__(
        self,
        index,
        reference_queries,
        k=10,
        # hard-query threshold
        hard_percentile=75, # percentile of warmup dist_comps -> hard threshold
        warmup_ef=50, # ef used to measure warmup dist_comps
        # rewiring
        alpha=1.1,
        max_layer=3,
        hard_rewire_cooldown=10, # hard queries between rewires (not total queries)
        # pool
        max_pool_size=50,
        pool_seed_ef=20, # ef used when finding the node to promote into the pool
        # ef escalation
        escalation_window=50, # rolling window size for hard-query fraction
        escalation_trigger=0.3, # fraction of hard queries in window that triggers boost
        escalation_factor=3, # boosted ef = ef * escalation_factor
        # feature flags
        use_pool=True,
        use_rewire=True,
        use_ef_escalation=True,
    ):
        self.index = index
        self.k = k
        self.alpha = alpha
        self.max_layer = max_layer
        self.hard_rewire_cooldown  = hard_rewire_cooldown
        self.max_pool_size = max_pool_size
        self.pool_seed_ef = pool_seed_ef
        self.escalation_window = escalation_window
        self.escalation_trigger = escalation_trigger
        self.escalation_factor = escalation_factor
        self.use_pool = use_pool
        self.use_rewire = use_rewire
        self.use_ef_escalation = use_ef_escalation

        self.warmup_ef = warmup_ef
        warmup_comps = _measure_dist_comps(index, reference_queries, warmup_ef, k)
        self.hard_threshold = float(np.percentile(warmup_comps, hard_percentile))

        self._pool = set()
        self._recent_hardness = []   # rolling bool window for escalation
        self._hard_queries_since_rewire = 0
        self._last_used_ef = warmup_ef

        self.update_count = 0
        self.update_log = []
        self.total_edges_added = 0
        self.escalation_count = 0
        self.total_adapt_time_ms = 0.0

        print(
            f"  [HardnessAdaptive] hard_threshold={self.hard_threshold:.1f} "
            f"(p{hard_percentile} of {len(warmup_comps)} warmup queries at ef={warmup_ef})"
        )


    # pool management

    def _pool_add(self, node_id):
        self._pool.add(int(node_id))

    def _pool_evict(self, anchor):
        """Evict pool nodes farthest from anchor until pool is within budget."""
        while len(self._pool) > self.max_pool_size:
            node_ids = list(self._pool)
            vecs = self.index.get_items(node_ids)
            sq_dists = np.sum((vecs - anchor) ** 2, axis=1)
            self._pool.discard(node_ids[int(np.argmax(sq_dists))])

    def _pool_best_entry(self, query):
        if not self._pool:
            return int(self.index.enterpoint_node)
        node_ids = list(self._pool)
        vecs = self.index.get_items(node_ids)
        sq_dists = np.sum((vecs - query) ** 2, axis=1)
        return node_ids[int(np.argmin(sq_dists))]


    # per-query update (called after each search with the returned stats)

    def record_and_adapt(self, query_vec, stats, used_ef):
        """
        Update internal state with the result of one query.

        """
        t_adapt_start = time.perf_counter()

        dist_comps = int(stats["base_layer_distance_computations"])
        # scale threshold based on used ef to avoid falsely classifying queries as hard when the 
        # last used ef value is higher than the warmup_ef that was used for calibration
        is_hard = dist_comps > self.hard_threshold * (used_ef/self.warmup_ef)

        # rolling hardness window for ef escalation
        self._recent_hardness.append(is_hard)
        if len(self._recent_hardness) > self.escalation_window:
            self._recent_hardness.pop(0)

        if not is_hard:
            t_adapt_ms = (time.perf_counter() - t_adapt_start) * 1000
            self.total_adapt_time_ms += t_adapt_ms
            return t_adapt_ms

        q = np.asarray(query_vec, dtype=np.float32).ravel()

        # pool: find the nearest index node to this hard query and promote it
        t_pool_ms = 0.0
        if self.use_pool:
            t0 = time.perf_counter()
            self.index.set_ef(self.pool_seed_ef)
            cand, _ = self.index.knn_query(q.reshape(1, -1), k=1)
            new_ep = int(cand[0][0])
            self.index.promote_node(new_ep, target_layer=self.max_layer)
            self._pool_add(new_ep)
            self._pool_evict(q)
            t_pool_ms = (time.perf_counter() - t0) * 1000

        # rewiring: after hard_rewire_cooldown hard queries, add corrective edges
        t_rewire_ms = 0.0
        self._hard_queries_since_rewire += 1
        if self.use_rewire and self._hard_queries_since_rewire >= self.hard_rewire_cooldown:
            t0 = time.perf_counter()
            n_edges = self.index.rewire_for_query(q, max_layer=self.max_layer, alpha=self.alpha)
            t_rewire_ms = (time.perf_counter() - t0) * 1000
            self.total_edges_added += n_edges
            self._hard_queries_since_rewire = 0
            self.update_count += 1
            self.update_log.append({
                "update_n":    self.update_count,
                "dist_comps":  dist_comps,
                "threshold":   self.hard_threshold,
                "edges_added": n_edges,
                "total_edges": self.total_edges_added,
                "pool_size":   len(self._pool),
                "t_pool_ms":   round(t_pool_ms, 3),
                "t_rewire_ms": round(t_rewire_ms, 3),
            })

        t_adapt_ms = (time.perf_counter() - t_adapt_start) * 1000
        self.total_adapt_time_ms += t_adapt_ms
        return t_adapt_ms

    # ef selectio

    def _effective_ef(self, ef_base):
        """
        Return the ef to use for the next query.
        Boosts ef when the rolling hard-query fraction exceeds escalation_trigger.
        """
        if not self.use_ef_escalation or len(self._recent_hardness) == 0:
            self._last_used_ef = ef_base
            return ef_base
        # True = 1 False =0
        hard_fraction = sum(self._recent_hardness) / len(self._recent_hardness)
        if hard_fraction >= self.escalation_trigger:
            self.escalation_count += 1
            escalated_ef = int(ef_base * self.escalation_factor)
            self._last_used_ef = escalated_ef
            return escalated_ef
        self._last_used_ef = ef_base
        return ef_base


    # search

    def search(self, query, k, ef):
        """
        Search with optional ef escalation and pool union-merge.
        Returns (labels, dists, stats) where stats is from the most informative search point
        (lowest base-layer entry distance).
        """
        query = np.asarray(query, dtype=np.float32).ravel()
        original_ep = int(self.index.enterpoint_node)
        ef_use = self._effective_ef(ef)
        t_total_start = time.perf_counter()

        # global-EP search
        self.index.set_entry_point(original_ep)
        self.index.set_ef(ef_use)
        t0 = time.perf_counter()
        labels_g, dists_g = self.index.knn_query(query.reshape(1, -1), k=k)
        t_global_knn_ms = (time.perf_counter() - t0) * 1000
        stats_g = hnswlib.get_last_query_stats()

        if not self.use_pool or not self._pool:
            stats_g["t_query_ms"] = (time.perf_counter() - t_total_start) * 1000
            stats_g["t_global_knn_ms"] = t_global_knn_ms
            stats_g["t_pool_scan_ms"] = 0.0
            stats_g["t_pool_knn_ms"] = 0.0
            return labels_g, dists_g, stats_g

        # pool search
        t0 = time.perf_counter()
        best_ep = self._pool_best_entry(query)
        t_pool_scan_ms = (time.perf_counter() - t0) * 1000

        self.index.set_entry_point(best_ep)
        self.index.set_ef(ef_use)
        t0 = time.perf_counter()
        labels_p, dists_p = self.index.knn_query(query.reshape(1, -1), k=k)
        t_pool_knn_ms = (time.perf_counter() - t0) * 1000
        stats_p = hnswlib.get_last_query_stats()

        # restore global EP
        self.index.set_entry_point(original_ep)

        # union-merge: guarantees recall >= baseline
        merged = {}
        for lab, dist in zip(labels_g[0], dists_g[0]):
            merged[int(lab)] = float(dist)
        for lab, dist in zip(labels_p[0], dists_p[0]):
            lab_i, dist_f = int(lab), float(dist)
            if lab_i not in merged or dist_f < merged[lab_i]:
                merged[lab_i] = dist_f

        sorted_items = sorted(merged.items(), key=lambda x: x[1])[:k]
        labels_merged = np.array([[it[0] for it in sorted_items]])
        dists_merged = np.array([[it[1] for it in sorted_items]], dtype=np.float32)

        stats = (stats_p if stats_p["base_layer_entry_distance"] <= stats_g["base_layer_entry_distance"] else stats_g)
        stats["t_query_ms"] = (time.perf_counter() - t_total_start) * 1000
        stats["t_global_knn_ms"] = t_global_knn_ms
        stats["t_pool_scan_ms"] = t_pool_scan_ms
        stats["t_pool_knn_ms"] = t_pool_knn_ms
        return labels_merged, dists_merged, stats


def run_query_batch_hardness(index, queries, gt, k, ef, controller=None):
    """
    Run a batch of queries with optional HardnessAdaptiveController.

    If controller is None, runs plain knn_query at the given ef (baseline).
    Returns list of per-query result dicts.
    """
    results = []

    for i, q in enumerate(queries):
        if controller is not None:
            t0 = time.perf_counter()
            labels, _, stats = controller.search(q, k=k, ef=ef)
            t_query_ms = (time.perf_counter() - t0) * 1000
        else:
            index.set_ef(ef)
            t0 = time.perf_counter()
            labels, _ = index.knn_query(q.reshape(1, -1), k=k)
            t_query_ms = (time.perf_counter() - t0) * 1000
            stats = hnswlib.get_last_query_stats()
            stats["t_query_ms"] = t_query_ms
            stats["t_global_knn_ms"] = t_query_ms
            stats["t_pool_scan_ms"] = 0.0
            stats["t_pool_knn_ms"] = 0.0

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
            "t_query_ms":           t_query_ms,
            "t_global_knn_ms":      float(stats.get("t_global_knn_ms", t_query_ms)),
            "t_pool_scan_ms":       float(stats.get("t_pool_scan_ms", 0.0)),
            "t_pool_knn_ms":        float(stats.get("t_pool_knn_ms", 0.0)),
            "t_adapt_ms":           0.0,
        })

        if controller is not None:
            t_adapt_ms = controller.record_and_adapt(q, stats, used_ef=controller._last_used_ef)
            results[-1]["t_adapt_ms"] = t_adapt_ms

    return results
