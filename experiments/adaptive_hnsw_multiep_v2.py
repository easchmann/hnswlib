# MultiEPAdaptiveHNSW v2 -- improvements over v1.

# Changes from v1
# 1. Pool rebuilt from scratch on each adaptation event 
# v1 accumulated EPs across events
# -> for continuous drift the pool fills with stale EPs from earlier drift levels that mislead per-query routing later.
# v2 rebuilds the pool from only the current buffer
# -> pool reflects the current query distribution.

# 2. Original EP is restored before each seed search in _adapt().
# v1 searched from whatever EP happened to be current. 
# -> if a previous adaptation had set a bad EP the seed searches would start from a wrong position.
# v2 always searches from the known-good original EP with high ef
# -> should reach the nearest data node even for drifted seeds

# 3. Threshold adapts after each adaptation event.
# After rebuilding the pool, the per-query routing lowers bl_entry_dist.
# Re-measuring the threshold from post-adaptation queries prevents the detector from immediately re-triggering on the (now hopefully improved) bl_entry_dist values.
# A fresh re-calibration is done on the next n_recalibrate queries after each adaptation.

# Usage:
#     raw = hnswlib.Index(space="l2", dim=128)
#     raw.init_index(max_elements=N, M=16, ef_construction=200)
#     raw.add_items(data)
#     index = MultiEPAdaptiveHNSW_v2(raw, ef_seed=2000)
#     index.calibrate(baseline_queries)
#     for q in query_stream:
#         labels, dists = index.knn_query(q, k=10, ef=ef, adapt=True)


from __future__ import annotations

import numpy as np
import hnswlib
from collections import deque
from typing import Optional


class MultiEPAdaptiveHNSW_v2:
    def __init__(self, index, *, window_size=200, threshold_factor=1.5, ef_seed=2000, query_buffer_size=500, n_recalibrate=100,):

        self.index = index
        self.window_size = window_size
        self.threshold_factor = threshold_factor
        self.ef_seed = ef_seed
        self.query_buffer_size = query_buffer_size
        self.n_recalibrate = n_recalibrate

        self.baseline = None
        self.threshold =None
        self.window = deque(maxlen=window_size)
        self.query_buffer = deque(maxlen=query_buffer_size)

        self.original_ep = index.enterpoint_node

        # Pool: None means routing not yet active.
        self.ep_vecs = None
        self.ep_ids = None

        # Post adaptation recalibration countdown.
        self.recal_remaining = 0
        self.recal_accum = []

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

    # query the index with optional per query EP routing and drift adaptation
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
            # Per-query routing to nearest pool member
            if self.ep_vecs is not None:
                sq_dists = ((self.ep_vecs - q) ** 2).sum(axis=1)
                best_ep = self.ep_ids[int(np.argmin(sq_dists))]
                if best_ep != self.index.enterpoint_node:
                    self.index.set_entry_point(best_ep)

            labels, dists = self.index.knn_query(q.reshape(1, -1), k=k)
            all_labels.append(labels[0])
            all_dists.append(dists[0])

            if adapt and self.threshold is not None:
                stats = hnswlib.get_last_query_stats()
                bl_entry = float(stats["base_layer_entry_distance"])

                # Post-adaptation recalibration: collect new baseline.
                if self.recal_remaining > 0:
                    self.recal_accum.append(bl_entry)
                    self.recal_remaining -= 1
                    if self.recal_remaining == 0:
                        new_baseline = float(np.mean(self.recal_accum))
                        self.baseline = new_baseline
                        self.threshold = new_baseline * self.threshold_factor
                        self.recal_accum = []
                        print(
                            f"[MultiEPAdaptiveHNSW_v2] recalibrated  "
                            f"new_baseline={new_baseline:.1f}  "
                            f"new_threshold={self.threshold:.1f}"
                        )
                    continue  # skip drift detection during recalibration

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
        

    @property
    def pool_size(self):
        if self.ep_ids is not None:
            return len(self.ep_ids)
        else:
            return 0


 

    def drift_detected(self):
        if len(self.window) < self.window_size:
            return False
        return self.window_mean() > self.threshold

   
    def adapt(self):
        buffer = np.array(self.query_buffer, dtype=np.float32)
        max_level = self.index.max_level

        # save and set high ef for reliable seed searches
        saved_ef = self.index.ef
        self.index.set_ef(self.ef_seed)

        # always search from the original EP so seed searches are consistent regardless of what set_entry_point calls have done in between.
        self.index.set_entry_point(self.original_ep)

        # find the nearest existing data node to each buffered query
        # each result is supposed to be a cluster/boundary representative:
        # individual queries lie near real data points, so the found node is in the correct neighbourhood for future similar queries.
        seen = {self.original_ep}
        new_eps = []
        for seed in buffer:
            # reset for each seed
            self.index.set_entry_point(self.original_ep)
            labels, _ = self.index.knn_query(seed.reshape(1, -1), k=1)
            node = int(labels[0][0])
            if node not in seen:
                seen.add(node)
                new_eps.append(node)

        self.index.set_ef(saved_ef)

        # promote each representative to max_level so upper-layer descent starts from it properly
        for node in new_eps:
            self.index.promote_node(node, max_level)

        # Rebuild pool from scratch (not cumulative). For continuous drift stale EPs from earlier phases might mislabel queries with the old routing.
        pool_eps = [self.original_ep] + new_eps
        self.ep_vecs = np.array(self.index.get_items(pool_eps), dtype=np.float32)
        self.ep_ids = pool_eps

        entry = {
            "adaptation_n": len(self.adaptation_log) + 1,
            "n_new_eps": len(new_eps),
            "total_eps": len(pool_eps),
            "max_level": max_level,
            "window_mean_before": self.window_mean(),
            "threshold": self.threshold,
        }
        self.adaptation_log.append(entry)
        print(
            f"[MultiEPAdaptiveHNSW_v2] adaptation #{entry['adaptation_n']}  "
            f"new_eps={len(new_eps)}  pool_size={len(pool_eps)}  "
            f"window_mean={entry['window_mean_before']:.1f}  "
            f"threshold={self.threshold:.1f}"
        )

        # Clear drift window and start post-adaptation recalibration.
        self.window.clear()
        self.recal_remaining = self.n_recalibrate
        self.recal_accum = []
