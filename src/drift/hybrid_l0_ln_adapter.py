"""Combined layer-0 and upper-layer EH-guided edge injection adapter."""

import numpy as np

from src.drift.adapter import compute_candidate_edges, find_repair_candidates
from src.drift.detector import assign_cell, compute_eh_batch


class HybridL0LNAdapter:
    """EH-guided combined layer-0 eviction and upper-layer injection, one shared signal."""

    def __init__(self, index, base, detector, cell_labels, centroids, config):
        self.index = index
        self.base = base
        self.detector = detector
        self.cell_labels = cell_labels
        self._centroids = centroids
        self.config = config
        self._inject_level = config.get("inject_level", 1)
        self._layer1_nodes = set(int(x) for x in index.get_nodes_at_layer(self._inject_level))
        self.node_eh_accumulator = np.zeros(base.shape[0], dtype=np.float64)
        self._node_repair_epoch = np.full(base.shape[0], -1, dtype=np.int32)
        self._l0_count = 0
        self._lN_count = 0
        self._last_repair_epoch = -999
        self.current_epoch = 0

    def search_enhanced(self, queries, k, ef_search):
        """Standard HNSW search; both layer-0 and upper-layer edges in graph."""
        self.index.set_ef(ef_search)
        ids, dists = self.index.knn_query(queries, k=k, num_threads=1)
        return ids, dists

    def process_epoch(self, queries, result_ids, result_distances, k):
        """Run EH detection once, inject into layer-0 and upper layer if drift detected."""
        result_vectors = self.base[result_ids]
        eh_values = compute_eh_batch(list(result_vectors))
        cell_ids = assign_cell(queries, self._centroids)

        for i, row in enumerate(result_ids):
            eh = float(eh_values[i])
            for nid in row:
                self.node_eh_accumulator[nid] = 0.9 * self.node_eh_accumulator[nid] + 0.1 * eh

        self.detector.update_batch(eh_values, cell_ids)
        drift_result = self.detector.check_drift()
        drift_detected = drift_result is not None and drift_result["drift_detected"]
        hot_cells = (drift_result or {}).get("hot_cells", [])
        l0_added = 0
        lN_added = 0
        lN_attempts = 0
        cooldown_ok = (self.current_epoch - self._last_repair_epoch) > self.config.get("repair_cooldown", 0)

        if drift_detected and hot_cells and cooldown_ok:
            repair_nodes = find_repair_candidates(
                hot_cells,
                self.cell_labels,
                self.node_eh_accumulator,
                eh_threshold_percentile=self.config.get("eh_threshold_percentile", 75.0),
                max_nodes=self.config.get("max_repair_nodes", 500),
                last_repaired=self._node_repair_epoch,
                use_diversity=False,
            )
            if len(repair_nodes) > 0:
                self._last_repair_epoch = self.current_epoch
                self._node_repair_epoch[repair_nodes] = self.current_epoch
                candidates = compute_candidate_edges(
                    repair_nodes,
                    self.index,
                    self.base,
                    M_candidates=self.config.get("M_candidates", 32),
                    ef_search=self.config.get("repair_ef_search", 200),
                    queries=queries,
                    result_ids=result_ids,
                )
                for src, cand_list in candidates.items():
                    for dst, _ in cand_list:
                        l0_added += self.index.add_layer0_edge_evict(int(src), int(dst))
                        if int(src) in self._layer1_nodes and int(dst) in self._layer1_nodes:
                            lN_attempts += 1
                            if self.index.add_back_edge(int(src), int(dst), self._inject_level):
                                lN_added += 1
                self._l0_count += l0_added
                self._lN_count += lN_added

        self.current_epoch += 1
        return {"drift_detected": drift_detected, "l0_edges_added": l0_added,
                "lN_edges_added": lN_added, "lN_attempts": lN_attempts}

    def l0_edge_count(self):
        return self._l0_count

    def lN_edge_count(self):
        return self._lN_count
