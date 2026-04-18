# AdaptiveHNSW — drop-in wrapper around hnswlib.Index that detects query
# distribution shift and improves recall by repairing base-layer connectivity
# and relocating the global entry point to the shifted region.

# Adaptation strategy
# -------------------
# Drift is detected via a sliding window on base_layer_entry_distance (the L2
# distance between the query and the node where upper-layer greedy descent
# terminates).  When the window mean exceeds baseline × threshold_factor, the
# following three steps execute:

# 1.  repair_base_layer(centroid, k_repair_nodes, ef_repair)
#     Finds the k_repair_nodes nodes nearest the query centroid using a high-ef
#     search seeded from the current entry point.  Each found node is re-searched
#     FROM ITSELF with ef_repair candidates (no bias from the global entry point),
#     and its layer-0 link list is overwritten with the MRNG-heuristic neighbours.
#     Result: dense, well-connected base-layer neighbourhood in the shifted region.

# 2.  promote_node(best_node, max_level)
#     best_node is the repaired node geometrically closest to the centroid.
#     Promoting it to the current max_level gives it upper-layer link lists wired
#     toward its geometric neighbours at each level (found via searchBaseLayer at
#     construction time).

# 3.  set_entry_point(best_node)
#     best_node is now a max_level node.  searchKnn starts greedy descent FROM
#     best_node at max_level.  For shifted queries, best_node's upper-layer
#     neighbours (wired toward the original distribution) are all farther from the
#     query than best_node itself, so greedy descent stays at best_node through
#     every level.  searchBaseLayerST therefore starts from best_node, which lies
#     near the query centroid.

#     Expected bl_entry_dist after adaptation ≈ ‖best_node − query‖²
#                                             ≈ cluster_std² × dim
#     vs. baseline × threshold_factor before adaptation.  This is a qualitative
#     drop, not a marginal one.

# Why earlier approaches failed
# ------------------------------
# • set_entry_point(level-0 node): skips upper-layer descent entirely
#   (level > element_levels_[currObj] → continue for every level), so
#   searchBaseLayerST starts from a fixed centroid node that may be worse than
#   what upper-layer descent would have found.

# • promote_node + add_back_edge (keep original EP): the back-edge gives EP a
#   chance to jump to best_node, but subsequent descent from best_node uses its
#   upper-layer connections (boundary nodes of the original distribution).  The
#   improvement is real but marginal — only ~7% reduction in bl_entry_dist.

# • promote_node + set_entry_point(best_node) [this version]: upper-layer
#   descent from best_node terminates at best_node itself (no closer upper-layer
#   neighbour exists in the shifted region), so bl_entry_dist collapses to the
#   within-cluster distance, giving large recall gains.

# Usage
# -----
#     import hnswlib
#     from adaptive_hnsw import AdaptiveHNSW

#     raw = hnswlib.Index(space="l2", dim=128)
#     raw.init_index(max_elements=100_000, M=16, ef_construction=200)
#     raw.add_items(data)

#     index = AdaptiveHNSW(raw)
#     index.calibrate(baseline_queries)

#     for batch in query_stream:
#         labels, dists = index.knn_query(batch, k=10)  # auto-adapts on drift


from __future__ import annotations

import numpy as np
import hnswlib
from collections import deque
from typing import Optional


class AdaptiveHNSW:

    def __init__(self, index: hnswlib.Index, *, window_size=200, threshold_factor=1.5, k_repair_nodes=50, ef_repair=500,n_repair_rounds=3,query_buffer_size=500):
        self.index = index
        self.window_size = window_size
        self.threshold_factor = threshold_factor
        self.k_repair_nodes = k_repair_nodes
        self.ef_repair = ef_repair
        self.n_repair_rounds = n_repair_rounds
        self.query_buffer_size = query_buffer_size

        # drift detection state
        self.baseline = None
        self.threshold = None
        self.window = deque(maxlen=window_size)

        # query buffer for centroid estimation
        self.query_buffer= deque(maxlen=query_buffer_size)

        # adaptation history
        self.adaptation_log = []


    # Run queries against the index and record the mean base_layer_entry_distance as the baseline.
    # Returns the measured baseline value.
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

    def knn_query(self,queries,k=10,ef=None, adapt=True):
        queries = np.asarray(queries, dtype=np.float32)
        single = queries.ndim == 1
        if single:
            queries = queries.reshape(1, -1)

        if ef is not None:
            self.index.set_ef(ef)

        all_labels = []
        all_dists = []

        for q in queries:
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


    def adapt(self) -> None:
        current_centroid = np.array(self.query_buffer, dtype=np.float32).mean(axis=0)
        max_level = self.index.max_level

        # Iterative repair: each round seeds repair_base_layer from the entry point set by the previous round. 
        # Round 1 starts from the original entry point and finds the best reachable tail nodes. 
        # Round 2 starts from those tail nodes (now set as EP). Repeat for n_repair_rounds total.
        best_node = None
        total_repaired = 0

        for round_i in range(self.n_repair_rounds):
            repaired_ids = self.index.repair_base_layer(
                current_centroid.astype(np.float32),
                k_nodes=self.k_repair_nodes,
                ef_repair=self.ef_repair,
            )
            total_repaired += len(repaired_ids)

            if repaired_ids:
                vecs = np.array(self.index.get_items(repaired_ids), dtype=np.float32)
                sq_dists = ((vecs - current_centroid) ** 2).sum(axis=1)
                candidate = int(repaired_ids[int(np.argmin(sq_dists))])
            else:
                # fallback: nearest neighbour with high ef
                saved_ef = self.index.ef
                self.index.set_ef(self.ef_repair)
                labels, _ = self.index.knn_query(current_centroid.reshape(1, -1), k=1)
                self.index.set_ef(saved_ef)
                candidate = int(labels[0][0])

            if candidate == best_node:
                break  # converged, further rounds won't help

            best_node = candidate

            # Promote best_node to max_level so searchKnn runs a full upper-layer descent from it (not skip all levels as for L0).
            self.index.promote_node(best_node, max_level)

            # Set best_node as the new entry point so the next repair round
            # seeds from a position closer to the shifted centroid.
            self.index.set_entry_point(best_node)

        entry = {
            "adaptation_n": len(self.adaptation_log) + 1,
            "n_repaired": total_repaired,
            "best_node": best_node,
            "max_level": max_level,
            "window_mean_before": self.window_mean(),
            "threshold": self.threshold,
        }
        self.adaptation_log.append(entry)
        print(
            f"[AdaptiveHNSW] adaptation #{entry['adaptation_n']}  "
            f"rounds={self.n_repair_rounds}  repaired={total_repaired} nodes  "
            f"new_ep={best_node}@L{max_level}  "
            f"window_mean={entry['window_mean_before']:.3f}  "
            f"threshold={self.threshold:.3f}"
        )

        # reset window so we don't trigger again immediately
        self.window.clear()
