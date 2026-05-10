import numpy as np
import hnswlib


class EntryPointController:

    def __init__(self, index, baseline_bl_entry_dist, window_size=200, threshold_factor=1.5):
        self.index = index
        self.baseline  = baseline_bl_entry_dist
        self.threshold = baseline_bl_entry_dist * threshold_factor
        self.window_size = window_size
        self.window = []
        self.update_count = 0
        self.update_log = []

    def record(self, bl_entry_dist):
        self.window.append(float(bl_entry_dist))
        if len(self.window) > self.window_size:
            self.window.pop(0)

    def drift_detected(self):
        if len(self.window) < self.window_size:
            return False
        return np.mean(self.window) > self.threshold

    def window_mean(self):
        return float(np.mean(self.window)) if self.window else float("nan")

    def reset_window(self):
        self.window = []

    def log_update(self, sigma, new_ep):
        self.update_count += 1
        self.update_log.append({
            "update_n":    self.update_count,
            "sigma":       sigma,
            "new_ep":      new_ep,
            "window_mean": self.window_mean(),
            "threshold":   self.threshold,
        })
        print(f"  [adapt] update #{self.update_count}  sigma={sigma}  "
              f"new_ep={new_ep}  window_mean={self.window_mean():.1f}  "
              f"threshold={self.threshold:.1f}")

    def update(self, recent_queries, sigma=None):
        raise NotImplementedError






# utilities

def measure_baseline(index, queries, ef=200, k=10):
    # run queries at sigma=0 and return mean bl_entry_dist
    # used to set the detection threshold before any shift is applied
    index.set_ef(ef)
    dists = []
    for q in queries:
        index.knn_query(q.reshape(1, -1), k=k)
        dists.append(float(hnswlib.get_last_query_stats()["base_layer_entry_distance"]))
    return float(np.mean(dists))


def run_query_batch(index, queries, gt, k, ef, controller=None):
    # run a batch of queries and collect per-query stats.
    # if a controller is passed, feeds bl_entry_dist into it for drift detection.
    index.set_ef(ef)
    results = []

    for i, q in enumerate(queries):
        pred, _ = index.knn_query(q.reshape(1, -1), k=k)
        s = hnswlib.get_last_query_stats()

        results.append({
            "recall":               len(set(pred[0]) & set(gt[i])) / k,
            "ep_dist":              float(s["entry_point_distance"]),
            "bl_entry_dist":        float(s["base_layer_entry_distance"]),
            "ul_dist_comps":        int(s["upper_layer_distance_computations"]),
            "layer_visits":         list(s["layer_visit_counts"]),
            "base_visited":         int(s["base_layer_visited_count"]),
            "base_dist_comps":      int(s["base_layer_distance_computations"]),
            "candidates_remaining": int(s["candidates_remaining_at_termination"]),
            "lb_trace":             list(s["lowerbound_trace"]),
        })

        if controller is not None:
            controller.record(float(s["base_layer_entry_distance"]))

    return results


# adaptation4: progressive neighbourhood expansion 
#
# The core problem with options 1-3 is that they make a single structural change based on a single centroid estimate. 
# For continuous drift this is insufficient: the query distribution moves gradually, so the graph needs
# to be updated gradually too.
#
# This strategy maintains a running estimate of the query distribution mean and promotes k_anchors nodes spread across the drift trajectory,
# not just the current centroid, but intermediate points between the baseline and the current centroid. 
# Each anchor node is promoted to layer 1 and wired in, so the navigational skeleton has coverage across the entire drift path.
#
# Additionally, instead of replacing the entry point, we keep the original entry point and add the new anchors alongside it.
# The entry point is only updated if the new anchor is strictly closer to the current query centroid than the existing entry point.

class ProgressiveExpansionController(EntryPointController):

    def __init__(self, index, baseline_bl_entry_dist,
                 window_size=200, threshold_factor=1.5,
                 k_anchors=4,        # number of intermediate anchor nodes
                 k_nodes_per_anchor=8,  # edges added per anchor
                 layer=1):
        super().__init__(index, baseline_bl_entry_dist, window_size, threshold_factor)
        self.k_anchors           = k_anchors
        self.k_nodes_per_anchor  = k_nodes_per_anchor
        self.layer               = layer
        self.baseline_centroid   = None  # set on first update
        self.promoted_anchors    = []    # internal ids of all promoted nodes

    def update(self, recent_queries, sigma=None):
        current_centroid = recent_queries.mean(axis=0).astype(np.float32)

        # on the first update, record the current query centroid as the
        # baseline so subsequent updates know the full drift trajectory
        if self.baseline_centroid is None:
            self.baseline_centroid = current_centroid.copy()

        # measure ep distance before adaptation so we can report improvement
        self.index.set_ef(200)
        self.index.knn_query(current_centroid.reshape(1, -1), k=10)
        import hnswlib as _hnswlib
        ep_dist_before = float(_hnswlib.get_last_query_stats()["entry_point_distance"])
        bl_dist_before = float(_hnswlib.get_last_query_stats()["base_layer_entry_distance"])

        # generate k_anchors intermediate points along the drift path.
        # anchor 1 is 1/k of the way from baseline to current centroid, anchor k is at the current centroid itself.
        # placing anchors along the full path rather than just at the tip creates a navigable chain
        anchors = []
        for i in range(1, self.k_anchors + 1):
            t = i / self.k_anchors
            anchor_point = (self.baseline_centroid * (1 - t) + current_centroid * t).astype(np.float32)
            anchors.append(anchor_point)

        self.index.set_ef(50)

        for anchor_point in anchors:
            labels, _ = self.index.knn_query(anchor_point.reshape(1, -1), k=1)
            anchor_node = int(labels[0][0])

            # promote to layer 1 so it becomes part of the navigational skeleton
            self.index.promote_node(anchor_node, target_layer=self.layer)

            # add directed edges from the surrounding neighbourhood toward this anchor so greedy descent can reach it from the existing graph
            self.index.add_directed_edges(anchor_point, layer=self.layer, k_nodes=self.k_nodes_per_anchor)
            self.promoted_anchors.append(anchor_node)

        # set entry point to the node closest to the current query centroid
        self.index.set_ef(50)
        labels_ep, _ = self.index.knn_query(current_centroid.reshape(1, -1), k=1)
        new_ep = int(labels_ep[0][0])
        self.index.set_entry_point(new_ep)

        # measure ep distance after adaptation
        self.index.set_ef(200)
        self.index.knn_query(current_centroid.reshape(1, -1), k=10)
        ep_dist_after = float(_hnswlib.get_last_query_stats()["entry_point_distance"])
        bl_dist_after = float(_hnswlib.get_last_query_stats()["base_layer_entry_distance"])

        print(f"ep_dist: {ep_dist_before:.1f} -> {ep_dist_after:.1f} "
              f"({'improved' if ep_dist_after < ep_dist_before else 'worse'})")
        print(f"bl_entry:{bl_dist_before:.1f} -> {bl_dist_after:.1f}  "
              f"({'improved' if bl_dist_after < bl_dist_before else 'worse'})")

        # update baseline centroid to current position so the next update places anchors from here rather than from the original baseline.
        # this makes the chain grow incrementally with each adaptation step.
        self.baseline_centroid = current_centroid.copy()

        self.log_update(sigma, new_ep)
        # do NOT reset the window. we want to keep # accumulating the bl_entry signal across sigma levels
        return new_ep