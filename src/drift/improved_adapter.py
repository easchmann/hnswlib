"""ImprovedAdaptationManager: four fixes over the current EH conjugate adapter."""

from collections import deque

import numpy as np

import hnswlib
from src.drift.conjugate_graph import ConjugateGraph
from src.drift.detector import (
    DriftDetector,
    assign_cell,
    build_spatial_index,
    compute_eh_batch,
)


class ImprovedAdaptationManager:
    """EH conjugate adapter with per-query repair, visited-expansion, sliding-window, mode dispatch."""

    def __init__(self, index, base, conjugate_graph, detector, cell_labels, centroids, config):
        self.index = index
        self.base = base
        self.conjugate_graph = conjugate_graph
        self.detector = detector
        self.cell_labels = cell_labels
        self._centroids = centroids
        self.config = config

        self.node_eh_accumulator = np.zeros(base.shape[0], dtype=np.float64)
        self.current_epoch = 0

        self.mini_batch_size = config["mini_batch_size"]
        self.min_repair_queries = config["min_repair_queries"]
        _visited_ok = hasattr(hnswlib, "get_last_query_stats")
        if not _visited_ok and config.get("use_visited_expansion", True):
            print("WARNING: get_last_query_stats not available — visited-expansion (fix 2) disabled; "
                  "falling back to top-k expansion only. Rebuild hnswlib with the thesis C++ patch.")
        self.use_visited_expansion = config.get("use_visited_expansion", True) and _visited_ok
        self.ef_repair = config["repair_ef_search"]
        self.M_candidates = config["M_candidates"]
        self.rng_relaxation = config.get("rng_relaxation", 0)
        self.primary_ef = config["primary_ef"]
        self._two_hop = config.get("two_hop", True)

        self._mode_dispatch_sample = config.get("mode_dispatch_sample_size", 50)
        self._mode_dispatch_ef_mult = config.get("mode_dispatch_ef_multiplier", 4)
        self._mode_dispatch_ratio = config.get("mode_dispatch_eh_ratio", 0.5)
        self._escalated_ef = config.get("escalated_ef", 128)

        self._current_mode = "repair"
        self._individual_eh_threshold = None
        self._calibration_eh_buffer = []
        self._drift_detected_ever = False
        self.mmd2_gate = config.get("mmd2_gate", False)
        self.eh_population_gate = config.get("eh_population_gate", False)
        self._eh_gate_window_size = config.get("eh_gate_window_size", 200)
        self._eh_gate_sigma = config.get("eh_gate_sigma", 2.0)
        self._eh_rolling_window = deque(maxlen=self._eh_gate_window_size)
        self._eh_population_gate_open = False
        self._eh_gate_threshold = None

    def run_calibration_epochs(self, dataset, n_calib, k, primary_ef):
        """Run pre-drift calibration epochs; returns (calib_eh, calib_cell_ids) flat arrays."""
        calib_eh, calib_cell_ids = [], []
        self.index.set_ef(primary_ef)
        for i in range(n_calib):
            queries = dataset["epochs"][i]
            labels, _ = self.index.knn_query(queries, k=k, num_threads=1)
            eh_vals = compute_eh_batch([self.base[labels[j]] for j in range(len(queries))])
            cids = assign_cell(queries, self._centroids)
            calib_eh.extend(eh_vals)
            calib_cell_ids.extend(cids.tolist())
        return np.array(calib_eh), np.array(calib_cell_ids, dtype=np.int32)

    def calibrate(self, calib_eh_values, calib_cell_ids):
        """Calibrate detector and set per-query EH threshold from calibration data."""
        self.detector.calibrate(calib_eh_values, calib_cell_ids)
        self._individual_eh_threshold = float(np.percentile(
            calib_eh_values, self.config["eh_individual_threshold_percentile"]
        ))
        print(f"Per-query EH threshold (p{self.config['eh_individual_threshold_percentile']:.0f}): "
              f"{self._individual_eh_threshold:.4f}")
        if self.eh_population_gate:
            self._eh_gate_threshold = (
                float(np.mean(calib_eh_values))
                + self._eh_gate_sigma * float(np.std(calib_eh_values))
            )
            print(f"Population EH gate threshold: {self._eh_gate_threshold:.4f} "
                  f"(calib_mean={float(np.mean(calib_eh_values)):.4f}, "
                  f"calib_std={float(np.std(calib_eh_values)):.4f})")

    def search_single(self, query, k, ef):
        """Search one query with conjugate expansion from all ef-visited nodes."""
        self.index.set_ef(ef)
        labels, distances = self.index.knn_query(query.reshape(1, -1), k=k)

        if self.use_visited_expansion:
            stats = hnswlib.get_last_query_stats()
            expansion_ids = np.array(stats["base_layer_visited_node_ids"], dtype=np.int64)
        else:
            expansion_ids = labels[0]

        candidate_ids = list(labels[0])
        candidate_dists = list(distances[0])
        seen = set(int(x) for x in candidate_ids)

        # hop 1: expand from all expansion_ids (visited or top-k), increment traversal counts
        hop1_new = []
        for node_id in expansion_ids:
            for edge in self.conjugate_graph.lookup(int(node_id), self.current_epoch):
                n = edge.neighbor_id
                if n not in seen:
                    seen.add(n)
                    diff = query - self.base[n]
                    dist = float(np.dot(diff, diff))
                    candidate_ids.append(n)
                    candidate_dists.append(dist)
                    hop1_new.append(n)

        # hop 2: expand from hop-1 discoveries without incrementing traversal counts
        if self._two_hop:
            for node_id in hop1_new:
                for edge in self.conjugate_graph._edges.get(node_id, []):
                    n = edge.neighbor_id
                    if n not in seen:
                        seen.add(n)
                        diff = query - self.base[n]
                        dist = float(np.dot(diff, diff))
                        candidate_ids.append(n)
                        candidate_dists.append(dist)

        if len(candidate_ids) <= k:
            dists_arr = np.array(candidate_dists, dtype=np.float32)
            ids_arr = np.array(candidate_ids, dtype=np.int64)
            order = np.argsort(dists_arr)
            return ids_arr[order], dists_arr[order]

        dists_arr = np.array(candidate_dists, dtype=np.float32)
        ids_arr = np.array(candidate_ids, dtype=np.int64)
        top_k = np.argpartition(dists_arr, k)[:k]
        top_k = top_k[np.argsort(dists_arr[top_k])]
        return ids_arr[top_k], dists_arr[top_k]

    def _run_mode_dispatch(self, hard_queries):
        """Diagnose whether drift is beam-width or routing failure; sets self._current_mode."""
        rng = np.random.default_rng(self.current_epoch)
        n = min(self._mode_dispatch_sample, len(hard_queries))
        idx = rng.choice(len(hard_queries), size=n, replace=False)
        sample = [hard_queries[i] for i in idx]

        high_ef = self.primary_ef * self._mode_dispatch_ef_mult

        eh_low, eh_high = [], []
        for q in sample:
            self.index.set_ef(self.primary_ef)
            lbl_low, _ = self.index.knn_query(q.reshape(1, -1), k=self.config["k"])
            eh_low.append(compute_eh_batch([self.base[lbl_low[0]]])[0])

            self.index.set_ef(high_ef)
            lbl_high, _ = self.index.knn_query(q.reshape(1, -1), k=self.config["k"])
            eh_high.append(compute_eh_batch([self.base[lbl_high[0]]])[0])

        mean_low = float(np.mean(eh_low))
        mean_high = float(np.mean(eh_high))

        if mean_high < self._mode_dispatch_ratio * mean_low:
            self._current_mode = "escalate"
        else:
            self._current_mode = "repair"

        print(f"Mode dispatch (epoch {self.current_epoch}): {self._current_mode} "
              f"(eh_low={mean_low:.4f}, eh_high={mean_high:.4f})")

    def _perquery_repair(self, hard_query_buffer):
        """Fix 1: repair from each hard query's exact vector instead of cell mean.

        Also measures, per repair event, the mean distance from the repair source nodes (``low_ef_ids``) to the query they were retrieved for. 
        (intended as test of the "positive feedback loop" hypothesis)
        """
        old_ef = self.index.ef
        self.index.set_ef(self.ef_repair)

        t_added = self.current_epoch / max(1, self.config.get("n_epochs", 25))
        edges_added = 0
        source_query_dists = []

        for q, low_ef_ids in hard_query_buffer:
            high_ef_labels, high_ef_dists = self.index.knn_query(
                q.reshape(1, -1), k=self.M_candidates
            )
            low_ef_set = set(int(x) for x in low_ef_ids)
            new_candidates = [
                (int(nid), float(d))
                for nid, d in zip(high_ef_labels[0], high_ef_dists[0])
                if int(nid) not in low_ef_set
            ]
            source_query_dists.append(
                float(np.mean(np.linalg.norm(self.base[low_ef_ids] - q, axis=1)))
            )
            for src in low_ef_ids:
                for dst_nid, dst_dist in new_candidates:
                    added = self.conjugate_graph.add_edge(
                        int(src), dst_nid, dst_dist,
                        t_added=t_added,
                        epoch_added=self.current_epoch,
                        current_epoch=self.current_epoch,
                    )
                    if added:
                        edges_added += 1

        self.index.set_ef(old_ef)
        return edges_added, source_query_dists

    def process_epoch_online(self, queries, k, ef_values):
        """Mini-batch sliding-window repair with per-query direction and mode dispatch."""
        n_queries = len(queries)
        primary_ef = self.primary_ef

        # Pre-allocate result arrays
        all_ids_by_ef = {ef: np.empty((n_queries, k), dtype=np.int64) for ef in ef_values}
        all_dists_by_ef = {ef: np.empty((n_queries, k), dtype=np.float32) for ef in ef_values}

        hard_query_buffer = []
        total_edges = 0
        repair_count = 0
        all_eh = []
        epoch_source_dists = []
        epoch_hardonly_static_dists = []
        epoch_hard_query_indices = []
        mode_dispatched_this_epoch = False

        for batch_start in range(0, n_queries, self.mini_batch_size):
            batch_end = min(batch_start + self.mini_batch_size, n_queries)
            mb_queries = queries[batch_start:batch_end]

            # search each query at all ef values
            for i, q in enumerate(mb_queries):
                gi = batch_start + i
                for ef in ef_values:
                    ids, dists = self.search_single(q, k, ef)
                    all_ids_by_ef[ef][gi] = ids
                    all_dists_by_ef[ef][gi] = dists

            # compute EH from primary ef results
            mb_result_vecs = [self.base[all_ids_by_ef[primary_ef][batch_start + i]]
                               for i in range(batch_end - batch_start)]
            mb_eh = compute_eh_batch(mb_result_vecs)
            all_eh.extend(mb_eh.tolist())

            if self.eh_population_gate and not self._eh_population_gate_open:
                self._eh_rolling_window.extend(mb_eh.tolist())
                if (len(self._eh_rolling_window) >= self._eh_gate_window_size // 2
                        and float(np.mean(self._eh_rolling_window)) > self._eh_gate_threshold):
                    self._eh_population_gate_open = True
                    print(f"  [eh_population_gate] opened at epoch {self.current_epoch} "
                          f"(mean_eh={float(np.mean(self._eh_rolling_window)):.4f}, "
                          f"threshold={self._eh_gate_threshold:.4f})")

            # update node EH accumulator (EMA)
            for i, eh in enumerate(mb_eh):
                gi = batch_start + i
                for node_id in all_ids_by_ef[primary_ef][gi]:
                    self.node_eh_accumulator[node_id] = (
                        0.9 * self.node_eh_accumulator[node_id] + 0.1 * float(eh)
                    )

            # update detector buffer
            mb_cell_ids = assign_cell(mb_queries, self._centroids)
            self.detector.update_batch(mb_eh, mb_cell_ids)

            # identify hard queries in mini-batch
            for i, eh in enumerate(mb_eh):
                if self._individual_eh_threshold is not None and eh > self._individual_eh_threshold:
                    if ((not self.mmd2_gate or self._drift_detected_ever)
                            and (not self.eh_population_gate or self._eh_population_gate_open)):
                        gi = batch_start + i
                        hard_query_buffer.append((queries[gi], all_ids_by_ef[primary_ef][gi].copy()))

                        # Hard-query-only static reference: raw, non-adaptive top-k
                        # search on this exact same query, restricted to the same
                        # hard-query population used for mean_source_query_dist. 
                        self.index.set_ef(primary_ef)
                        raw_labels, _ = self.index.knn_query(
                            queries[gi].reshape(1, -1), k=k, num_threads=1
                        )
                        epoch_hardonly_static_dists.append(float(np.mean(
                            np.linalg.norm(self.base[raw_labels[0]] - queries[gi], axis=1)
                        )))

                        # Record which query (by index within this epoch's batch) was
                        # flagged hard, to compute a matched recall comparison (adaptive vs. raw static, same queries) against ground truth. 
                        epoch_hard_query_indices.append(gi)

            # fire repair when buffer is large enough
            if len(hard_query_buffer) >= self.min_repair_queries:
                if self._drift_detected_ever and not mode_dispatched_this_epoch:
                    self._run_mode_dispatch([q for q, _ in hard_query_buffer])
                    mode_dispatched_this_epoch = True

                if self._current_mode == "repair":
                    edges, dists = self._perquery_repair(hard_query_buffer)
                    total_edges += edges
                    epoch_source_dists.extend(dists)
                    repair_count += 1
                elif self._current_mode == "escalate":
                    # re-run hard queries at escalated ef
                    for q, _ in hard_query_buffer:
                        pass  # escalation updates are for future queries; no structural change
                    repair_count += 1

                hard_query_buffer = []

        # drain remaining buffer
        if hard_query_buffer:
            if ((not self.mmd2_gate or self._drift_detected_ever)
                    and (not self.eh_population_gate or self._eh_population_gate_open)):
                if self._drift_detected_ever and not mode_dispatched_this_epoch:
                    self._run_mode_dispatch([q for q, _ in hard_query_buffer])
                    mode_dispatched_this_epoch = True

                if self._current_mode == "repair":
                    edges, dists = self._perquery_repair(hard_query_buffer)
                    total_edges += edges
                    epoch_source_dists.extend(dists)
                    repair_count += 1
                elif self._current_mode == "escalate":
                    repair_count += 1

            hard_query_buffer = []

        # global drift check
        drift_result = self.detector.check_drift()
        drift_detected = drift_result is not None and drift_result["drift_detected"]
        mmd_sq = drift_result["mmd_squared"] if drift_result is not None else None

        if drift_detected and not self._drift_detected_ever:
            self._drift_detected_ever = True
            # run mode dispatch on epoch's hard queries (buffer already drained; use accumulated)
            if all_eh and not mode_dispatched_this_epoch:
                eh_arr = np.array(all_eh)
                if self._individual_eh_threshold is not None:
                    hard_mask = eh_arr > self._individual_eh_threshold
                    hard_indices = np.where(hard_mask)[0]
                    if len(hard_indices) > 0:
                        sample_q = [queries[i] for i in hard_indices]
                        self._run_mode_dispatch(sample_q)

        # Reference distance: mean distance from the raw static HNSW top-k results to their queries, over every query in the epoch.
        # "target region" baseline for mean_source_query_dist 
        self.index.set_ef(primary_ef)
        static_labels, _ = self.index.knn_query(queries, k=k, num_threads=1)
        mean_static_topk_dist = float(np.mean(
            np.linalg.norm(self.base[static_labels] - queries[:, None, :], axis=2)
        ))

        self.current_epoch += 1

        return all_ids_by_ef, all_dists_by_ef, {
            "drift_detected": drift_detected,
            "mmd_squared": mmd_sq,
            "total_edges_added": total_edges,
            "repair_count": repair_count,
            "mean_eh": float(np.mean(all_eh)) if all_eh else 0.0,
            "mode": self._current_mode,
            "eh_population_gate_open": self._eh_population_gate_open,
            "mean_eh_window": float(np.mean(self._eh_rolling_window)) if self._eh_rolling_window else 0.0,
            "mean_source_query_dist": float(np.mean(epoch_source_dists)) if epoch_source_dists else None,
            "mean_static_topk_dist": mean_static_topk_dist,
            "mean_hardonly_static_dist": (
                float(np.mean(epoch_hardonly_static_dists)) if epoch_hardonly_static_dists else None
            ),
            "hard_query_indices": epoch_hard_query_indices,
        }
