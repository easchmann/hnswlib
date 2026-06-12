"""Conjugate graph: secondary edge store augmenting hnswlib without modifying it."""

import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ConjugateEdge:
    neighbor_id: int
    distance: float
    t_added: float
    epoch_added: int
    traversal_count: int = 0

    def staleness(self, current_epoch, alpha=1.0, beta=0.5):
        """Higher = more stale = eviction candidate."""
        return alpha * (current_epoch - self.epoch_added) - beta * self.traversal_count


class ConjugateGraph:

    def __init__(self, M_conj=8, max_total_edges=500_000):
        self.M_conj = M_conj
        self.max_total_edges = max_total_edges
        self._edges = {}  # node_id -> list[ConjugateEdge]
        self._total_edges = 0

    def add_edge(self, src, dst, distance, t_added, epoch_added, current_epoch):
        """Add directed edge src -> dst. Returns True if edge was added."""
        neighbors = self._edges.get(src)

        if neighbors is None:
            # Check capacity
            if self._total_edges >= self.max_total_edges:
                evicted = self.evict(current_epoch)
                if evicted == 0 and self._total_edges >= self.max_total_edges:
                    return False
            self._edges[src] = [ConjugateEdge(dst, distance, t_added, epoch_added)]
            self._total_edges += 1
            return True

        if len(neighbors) < self.M_conj:
            if self._total_edges >= self.max_total_edges:
                evicted = self.evict(current_epoch)
                if evicted == 0 and self._total_edges >= self.max_total_edges:
                    return False
            neighbors.append(ConjugateEdge(dst, distance, t_added, epoch_added))
            self._total_edges += 1
            return True

        # Node is full — evict most stale edge if new edge is fresher
        new_staleness = ConjugateEdge(dst, distance, t_added, epoch_added).staleness(current_epoch)
        most_stale_idx = max(range(len(neighbors)), key=lambda i: neighbors[i].staleness(current_epoch))
        most_stale = neighbors[most_stale_idx]

        if new_staleness < most_stale.staleness(current_epoch):
            neighbors[most_stale_idx] = ConjugateEdge(dst, distance, t_added, epoch_added)
            return True

        return False

    def lookup(self, node_id, current_epoch):
        """Return edges from node_id, incrementing traversal counts."""
        edges = self._edges.get(node_id, [])
        for e in edges:
            e.traversal_count += 1
        return edges

    def enhanced_search(self, primary_result_ids, primary_result_distances, base, query_vector, k, current_epoch, two_hop=True):
        """Augment HNSW top-k with conjugate graph expansion."""
        # Start with primary results
        candidate_ids = list(primary_result_ids)
        candidate_dists = list(primary_result_distances)
        seen = set(candidate_ids)

        # Hop 1: expand from primary HNSW results; increment traversal counts
        hop1_new = []
        for node_id in primary_result_ids:
            for edge in self.lookup(int(node_id), current_epoch):
                n = edge.neighbor_id
                if n not in seen:
                    seen.add(n)
                    diff = query_vector - base[n]
                    dist = float(np.dot(diff, diff))  # squared L2, matches hnswlib space="l2"
                    candidate_ids.append(n)
                    candidate_dists.append(dist)
                    hop1_new.append(n)

        # Hop 2: expand from hop-1 discoveries; do NOT increment traversal counts
        if two_hop:
            for node_id in hop1_new:
                for edge in self._edges.get(node_id, []):
                    n = edge.neighbor_id
                    if n not in seen:
                        seen.add(n)
                        diff = query_vector - base[n]
                        dist = float(np.dot(diff, diff))
                        candidate_ids.append(n)
                        candidate_dists.append(dist)

        if len(candidate_ids) <= k:
            # Pad if not enough candidates
            ids_arr = np.array(candidate_ids, dtype=np.int64)
            dists_arr = np.array(candidate_dists, dtype=np.float32)
            order = np.argsort(dists_arr)
            return ids_arr[order], dists_arr[order]

        dists_arr = np.array(candidate_dists, dtype=np.float32)
        ids_arr = np.array(candidate_ids, dtype=np.int64)
        top_k_idx = np.argpartition(dists_arr, k)[:k]
        top_k_idx = top_k_idx[np.argsort(dists_arr[top_k_idx])]
        return ids_arr[top_k_idx], dists_arr[top_k_idx]

    def evict(self, current_epoch, n_evict=None):
        """Remove most stale edges globally. Returns count evicted."""
        if n_evict is None:
            target = int(self.max_total_edges * 0.9)
            n_evict = max(0, self._total_edges - target)

        if n_evict <= 0:
            return 0

        # Collect (staleness, node_id, list_index)
        all_edges = []
        for node_id, neighbors in self._edges.items():
            for i, e in enumerate(neighbors):
                all_edges.append((e.staleness(current_epoch), node_id, i))

        if not all_edges:
            return 0

        all_edges.sort(key=lambda x: -x[0])  # most stale first
        to_remove = all_edges[:n_evict]

        # Group by node to remove indices in reverse order
        by_node = {}
        for _, node_id, idx in to_remove:
            by_node.setdefault(node_id, []).append(idx)

        evicted = 0
        for node_id, indices in by_node.items():
            for i in sorted(indices, reverse=True):
                self._edges[node_id].pop(i)
                evicted += 1
            if not self._edges[node_id]:
                del self._edges[node_id]

        self._total_edges -= evicted
        return evicted

    def stats(self):
        if not self._edges:
            return {
                "n_nodes_with_edges": 0,
                "total_edges": 0,
                "mean_edges_per_node": 0.0,
                "max_edges_per_node": 0,
                "mean_traversal_count": 0.0,
                "mean_staleness": 0.0,
            }
        counts = [len(v) for v in self._edges.values()]
        all_e = [e for neighbors in self._edges.values() for e in neighbors]
        return {
            "n_nodes_with_edges": len(self._edges),
            "total_edges": self._total_edges,
            "mean_edges_per_node": float(np.mean(counts)),
            "max_edges_per_node": int(np.max(counts)),
            "mean_traversal_count": float(np.mean([e.traversal_count for e in all_e])),
            "mean_staleness": float(np.mean([e.staleness(0) for e in all_e])),
        }

    def save(self, path):
        data = {
            "M_conj": self.M_conj,
            "max_total_edges": self.max_total_edges,
            "edges": {
                str(node_id): [
                    {
                        "neighbor_id": e.neighbor_id,
                        "distance": e.distance,
                        "t_added": e.t_added,
                        "epoch_added": e.epoch_added,
                        "traversal_count": e.traversal_count,
                    }
                    for e in neighbors
                ]
                for node_id, neighbors in self._edges.items()
            },
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        cg = cls(M_conj=data["M_conj"], max_total_edges=data["max_total_edges"])
        for node_id_str, neighbors in data["edges"].items():
            node_id = int(node_id_str)
            cg._edges[node_id] = [
                ConjugateEdge(
                    neighbor_id=e["neighbor_id"],
                    distance=e["distance"],
                    t_added=e["t_added"],
                    epoch_added=e["epoch_added"],
                    traversal_count=e["traversal_count"],
                )
                for e in neighbors
            ]
            cg._total_edges += len(neighbors)
        return cg
