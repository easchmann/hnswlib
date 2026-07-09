"""EfEscalationAdapter: per-query ef escalation triggered by individual EH threshold.

Purely inference-time: does not modify graph topology. Hard queries (EH > calibration p75 threshold) are re-run at ef_base * escalation_factor; 
the better result (lower mean distance) is returned. Best fix for cluster drift since the graph structure
is adequate, but the beam is too narrow.
"""

import numpy as np

from src.drift.detector import assign_cell, compute_eh_batch


class EfEscalationAdapter:
    """Wraps an HNSW index; re-runs hard queries at a higher ef."""

    def __init__(self, index, base, detector, centroids, config):
        self.index = index
        self.base = base
        self.detector = detector
        self._centroids = centroids
        self.config = config
        self._individual_eh_threshold = None

    def calibrate(self, calib_eh, calib_cell_ids):
        """Calibrate detector and set per-query EH threshold from calibration data."""
        self.detector.calibrate(calib_eh, calib_cell_ids)
        self._individual_eh_threshold = float(
            np.percentile(calib_eh, self.config["eh_threshold_percentile"])
        )
        print(
            f"EfEscalation EH threshold "
            f"(p{self.config['eh_threshold_percentile']:.0f}): "
            f"{self._individual_eh_threshold:.4f}"
        )

    def search(self, queries, k, ef_base):
        """Search all queries; escalate hard ones to ef_base * escalation_factor.

        Returns (all_ids, all_dists, mean_eh, n_escalated).
        """
        factor = self.config["escalation_factor"]
        ef_escalated = ef_base * factor
        n_queries = len(queries)
        all_ids = np.empty((n_queries, k), dtype=np.int64)
        all_dists = np.empty((n_queries, k), dtype=np.float32)
        n_escalated = 0

        # batch search at ef_base for all queries
        self.index.set_ef(ef_base)
        ids_base, dists_base = self.index.knn_query(queries, k=k, num_threads=1)

        # compute EH from ef_base results to decide which queries are hard
        all_eh = compute_eh_batch([self.base[ids_base[i]] for i in range(n_queries)])

        for i in range(n_queries):
            if (
                self._individual_eh_threshold is not None
                and all_eh[i] > self._individual_eh_threshold
            ):
                # re-run at escalated ef
                self.index.set_ef(ef_escalated)
                ids_esc, dists_esc = self.index.knn_query(queries[i].reshape(1, -1), k=k)
                n_escalated += 1
                # take the result with lower mean distance (closer = better)
                if np.mean(dists_esc[0]) < np.mean(dists_base[i]):
                    all_ids[i] = ids_esc[0]
                    all_dists[i] = dists_esc[0]
                else:
                    all_ids[i] = ids_base[i]
                    all_dists[i] = dists_base[i]
            else:
                all_ids[i] = ids_base[i]
                all_dists[i] = dists_base[i]

        return all_ids, all_dists, float(np.mean(all_eh)), n_escalated

    def process_epoch(self, queries, all_ids_primary, mean_eh, cell_ids):
        """Update detector state; no structural changes. Returns drift_result."""
        eh_vals = compute_eh_batch(
            [self.base[all_ids_primary[i]] for i in range(len(queries))]
        )
        self.detector.update_batch(eh_vals, cell_ids)
        return self.detector.check_drift()
