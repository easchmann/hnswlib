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
    




# adaptation 1: edge insertion
# adds edges from the k_nodes layer 1 (1 by default but can be set to any layer) nodes closest to the query centroid toward the centroid region
class DirectedEdgeController(EntryPointController):
    def __init__(self, index, baseline_entry_dist, window_size=200, threshold_factor=1.5, k_nodes=16,layer=1):
        super().__init__(index, baseline_entry_dist, window_size, threshold_factor)
        self.k_nodes = k_nodes
        self.layer = layer
    
    def update(self, queries, sigma=None):
        centroid = queries.mean(axis=0).astype(np.float32)
        self.index.add_directed_edges(centroid, layer=self.layer, k_nodes=self.k_nodes)

        # set entry point to the node closest to the centroid so queries benefit from the new edges
        self.index.set_ef(50)
        labels, _ = self.index.knn_query(centroid.reshape(1,-1), k=1)
        new_entry_point = int(labels[0][0])
        self.index.set_entry_point(new_entry_point)

        self.log_update(sigma, new_entry_point)
        self.reset_window()
        return new_entry_point
    

# adaptation 2: node promotion to upper layer
# finds the node closest to the query centroid and promotes it to layer 1, wiring it in using the standard HNSW neighbour-selection heuristic.
# promoted node becomes the new entry point
class PromotionController(EntryPointController):
    def __init__(self, index, baseline_entry_dist, window_size=200, threshold_factor=1.5, target_layer=1):
        super().__init__(index, baseline_entry_dist, window_size, threshold_factor)
        self.target_layer=target_layer

    def update(self, queries, sigma=None):
        centroid = queries.mean(axis=0).astype(np.float32)

        #find existing node closest to centroid
        self.index.set_ef(50)
        labels, _ = self.index.knn_query(centroid.reshape(1, -1), k=1)
        new_entry_point = int(labels[0][0])

        #lift it to layer 1 and make it entry point
        self.index.promote_node(new_entry_point, target_layer=self.target_layer)

        self.index.set_entry_point(new_entry_point)

        self.log_update(sigma, new_entry_point)
        self.reset_window()
        return new_entry_point
    

#adaptation 3: 
# recomputes neighbours for the k_nodes layer 1 nodes closest to the query centroid, using each node itself as the search centre
class RewireController(EntryPointController):
    def __init__(self, index, baseline_bl_entry_dist,
                    window_size=200, threshold_factor=1.5,
                    k_nodes=32, layer=1):
            super().__init__(index, baseline_bl_entry_dist, window_size, threshold_factor)
            self.k_nodes = k_nodes
            self.layer   = layer
    
    def update(self, queries, sigma=None):
        centroid = queries.mean(axis=0).astype(np.float32)
        self.index.rewire_local_neighbourhood(centroid, layer=self.layer, k_nodes=self.k_nodes)

        #reposition the entry point
        self.index.set_ef(50)
        labels, _ = self.index.knn_query(centroid.reshape(1, -1), k=1)
        new_entry_point = int(labels[0][0])
        self.index.set_entry_point(new_entry_point)

        self.log_update(sigma, new_entry_point)
        self.reset_window()
        return new_entry_point







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


