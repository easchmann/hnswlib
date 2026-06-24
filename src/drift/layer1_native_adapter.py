"""Upper-layer edge injection seeded from a FAISS sub-index over layer-1 nodes."""

import faiss
import numpy as np

from src.drift.detector import assign_cell, compute_eh_batch


class Layer1NativeAdapter:
    """Upper-layer injection seeded from a FAISS sub-index over layer-1 nodes."""

    def __init__(self, index, base, detector, cell_labels, centroids, config):
        self.index = index
        self.base = base
        self.detector = detector
        self.cell_labels = cell_labels
        self._centroids = centroids
        self.config = config
        self._inject_level = config.get("inject_level", 1)
        self._k_src = config.get("k_src", 50)
        self._k_dst = config.get("k_dst", 32)
        self.node_eh_accumulator = np.zeros(base.shape[0], dtype=np.float64)
        self._total_edges = 0
        self._total_attempts = 0
        self._last_repair_epoch = -999
        self.current_epoch = 0

        self._layer1_ids = np.array(index.get_nodes_at_layer(self._inject_level), dtype=np.int64)
        print(f"  Building FAISS sub-index over {len(self._layer1_ids)} layer-1 nodes...", flush=True)
        self._faiss_index = faiss.IndexFlatL2(base.shape[1])
        self._faiss_index.add(base[self._layer1_ids].astype(np.float32))
        print(f"  Layer-1 sub-index ready.", flush=True)

    def search_enhanced(self, queries, k, ef_search):
        """Standard HNSW search; injected upper-layer edges already in graph."""
        self.index.set_ef(ef_search)
        ids, dists = self.index.knn_query(queries, k=k, num_threads=1)
        return ids, dists

    def process_epoch(self, queries, result_ids, result_distances, k, drift_result=None):
        """Update EH accumulator, detect drift, inject layer-1 edges if drift detected."""
        result_vectors = self.base[result_ids]
        eh_values = compute_eh_batch(list(result_vectors))
        cell_ids = assign_cell(queries, self._centroids)

        for i, row in enumerate(result_ids):
            eh = float(eh_values[i])
            for nid in row:
                self.node_eh_accumulator[nid] = 0.9 * self.node_eh_accumulator[nid] + 0.1 * eh

        if drift_result is None:
            self.detector.update_batch(eh_values, cell_ids)
            drift_result = self.detector.check_drift()

        drift_detected = drift_result is not None and drift_result["drift_detected"]
        hot_cells = (drift_result or {}).get("hot_cells", [])
        edges_added = 0
        attempts = 0
        cooldown_ok = (self.current_epoch - self._last_repair_epoch) > self.config.get("repair_cooldown", 0)

        if drift_detected and hot_cells and cooldown_ok:
            all_src = set()
            for cell in hot_cells:
                q = self._centroids[cell].reshape(1, -1).astype(np.float32)
                _, rows = self._faiss_index.search(q, self._k_src)
                for r in rows[0]:
                    if r >= 0:
                        all_src.add(int(self._layer1_ids[r]))
            self._last_repair_epoch = self.current_epoch
            for src in all_src:
                q = self.base[src].reshape(1, -1).astype(np.float32)
                _, rows = self._faiss_index.search(q, self._k_dst + 1)
                for r in rows[0]:
                    if r < 0:
                        continue
                    dst = int(self._layer1_ids[r])
                    if dst == src:
                        continue
                    attempts += 1
                    if self.index.add_back_edge(src, dst, self._inject_level):
                        edges_added += 1
            self._total_edges += edges_added
            self._total_attempts += attempts

        self.current_epoch += 1
        return {"drift_detected": drift_detected, "edges_added": edges_added, "attempts": attempts}

    def edge_count(self): return self._total_edges
    def attempt_count(self): return self._total_attempts
