#controller and utilities for drift adaptation mechanisms
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


