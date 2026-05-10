# controller and utilities for drift adaptation mechanisms
# adaptations using entry point distance as drift indicator can use EntryPointController as
# parent class and override update()

# EntryPointController monitors base layer entry distance over a sliding window and triggers an update
# when drift is detected.

import numpy as np
import hnswlib

class EntryPointController:
    def __init__(self, index, baseline_entry_dist, window_size=200, threshold_factor=1.5):
        self.index = index
        self.baseline = baseline_entry_dist
        self.threshold = threshold_factor * baseline_entry_dist
        self.window_size = window_size
        self.window = []
        self.update_count = 0
        self.update_log = []
        self.updated_sigmas = set()

    def record(self, entry_dist):
        self.window.append(float(entry_dist))
        if len(self.window) > self.window_size:
            # slide window by one if already filled
            self.window.pop(0)
    
    def drift_detected(self):
        if len(self.window) < self.window_size:
            return False
        else:
            return float(np.mean(self.window)) > self.threshold
        
    def window_mean(self):
        if self.window:
            return float(np.mean(self.window))
        else: 
            return float("nan")
    
    def reset_window(self):
        self.window = []
    
    def log_update(self, sigma, new_entry_point):
        self.update_log.append({
            "update_number": self.update_count,
            "sigma": sigma,
            "new_entry_point": new_entry_point,
            "window_mean": self.window_mean(),
            "threshold": self.threshold,
        })
    
    def update(self, queries, sigma=None):
        raise NotImplementedError("update() must be implemented by subclass")
    




# utilites

# run queries at index build time distribution and return mean base layer entry distance
# needed to set detection threshold before shift is applied
def measure_baseline(index, queries, ef=200, k=10):
    index.set_ef(ef)
    dists= []
    for query in queries:
        index.knn_query(query.reshape(1,-1), k=k)
        stats = hnswlib.get_last_query_stats()
        dists.append(float(stats["base_layer_entry_distance"]))
    return float(np.mean(dists))

def run_query_batch(index, queries, ground_truth, k, ef, controller=None):
    index.set_ef(ef)
    results = []

    for i, query in enumerate(queries):
        pred, _= index.knn_query(query.reshape(1,-1), k=k)
        stats = hnswlib.get_last_query_stats()

        results.append({
            "recall": len(set(pred[0]) & set(ground_truth[i])) / k,
            "ep_dist": float(stats["entry_point_distance"]),
            "bl_entry_dist": float(stats["base_layer_entry_distance"]),
            "ul_dist_comps": int(stats["upper_layer_distance_computations"]),
            "layer_visits": list(stats["layer_visit_counts"]),
            "base_visited": int(stats["base_layer_visited_count"]),
            "base_dist_comps": int(stats["base_layer_distance_computations"]),
            "candidates_remaining": int(stats["candidates_remaining_at_termination"]),
            "lb_trace": list(stats["lowerbound_trace"]),

        })

        if controller is not None:
            controller.record(float(stats["base_layer_entry_distance"]))
    
    return results



# adaptation 4: progressive neighbourhood expansion
# maintains a running estimate of the query distribution mean and promotes k_anchors nodes spread across the drift trajectory 
# Each anchor node is promoted to layer 1 and wired in, so the navigational skeleton has coverage across the entire drift path.
#
# instead of replacing the entry point, we keep the original entry point and add the new anchors alongside it.
# This should preserve recall for in-distribution queries while improving recall for shifted queries.
# The entry point is only updated if the new anchor is strictly closer to the current query centroid than the existing entry point.

class ProgressiveExpansionController(EntryPointController):

    def __init__(self, index, baseline_bl_entry_dist, window_size=200, threshold_factor=1.5,
                 k_anchors=4, # number of intermediate anchor nodes
                 k_nodes_per_anchor=8, # edges added per anchor
                 layer=1):
        super().__init__(index, baseline_bl_entry_dist, window_size, threshold_factor)
        self.k_anchors = k_anchors
        self.k_nodes_per_anchor = k_nodes_per_anchor
        self.layer = layer
        # set on first update
        self.baseline_centroid = None 
        # internal ids of all promoted nodes
        self.promoted_anchors = []    

    def update(self, recent_queries, sigma=None):
        current_centroid = recent_queries.mean(axis=0).astype(np.float32)

        # on the first update, record the baseline centroid so we know the direction of drift relative to the construction distribution
        if self.baseline_centroid is None:
            self.index.set_ef(50)
            # approximate the construction centroid by querying from a zero vector and taking the result's location
            labels, _ = self.index.knn_query(current_centroid.reshape(1, -1), k=1)
            ep_vec = np.array(self.index.get_items([int(labels[0][0])]), dtype=np.float32).flatten()
            self.baseline_centroid = ep_vec

        # generate k_anchors intermediate points along the drift path from baseline toward the current centroid
        anchors = []
        for i in range(1, self.k_anchors + 1):
            t = i / self.k_anchors 
            anchor_point = (self.baseline_centroid * (1 - t) + current_centroid * t).astype(np.float32)
            anchors.append(anchor_point)

        self.index.set_ef(50)
        new_ep = None

        for anchor_point in anchors:
            # find the existing node closest to this anchor point
            labels, dists = self.index.knn_query(anchor_point.reshape(1, -1), k=1)
            anchor_node   = int(labels[0][0])

            # promote it to layer 1 if not already there
            self.index.promote_node(anchor_node, target_layer=self.layer)

            # add directed edges toward this anchor from its neighbourhood so greedy descent can navigate toward it
            self.index.add_directed_edges(
                anchor_point, layer=self.layer, k_nodes=self.k_nodes_per_anchor
            )

            self.promoted_anchors.append(anchor_node)

            # track the anchor closest to the current centroid as the new ep
            if anchor_point is anchors[-1]:
                new_ep = anchor_node

        # only update the entry point if the new anchor is closer to the current centroid than the existing entry point
        labels_ep, dist_ep = self.index.knn_query(current_centroid.reshape(1, -1), k=1)
        best_node = int(labels_ep[0][0])
        self.index.set_entry_point(best_node)
        new_ep = best_node

        self.log_update(sigma, new_ep)
        self.reset_window()
        return new_ep