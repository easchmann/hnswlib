"""Layer-0 edge injection adapter: injects edges directly into HNSW layer-0."""

from collections import Counter
import numpy as np


class Layer0Adapter:
    """NAF-guided layer-0 edge injection."""

    def __init__(self, index, base, hot_k, ef_repair, n_neighbors):
        self.index = index
        self.base = base
        self.hot_k = hot_k
        self.ef_repair = ef_repair
        self.n_neighbors = n_neighbors
        self.visit_counts = Counter()
        self._edges_added = 0

    def record_visits(self, labels):
        for node in labels:
            self.visit_counts[int(node)] += 1

    def repair(self):
        top_nodes = [nid for nid, _ in self.visit_counts.most_common(self.hot_k)]
        old_ef = self.index.ef
        self.index.set_ef(self.ef_repair)
        edges_added = 0
        for v in top_nodes:
            query = self.base[v:v + 1]
            labels, _ = self.index.knn_query(query, k=self.ef_repair)
            pool = [int(x) for x in labels[0] if int(x) != v]
            if not pool:
                continue
            dists = np.sum(
                (self.base[pool].astype(np.float64) - query.astype(np.float64)) ** 2,
                axis=1,
            )
            added = 0
            for idx in np.argsort(dists):
                if added >= self.n_neighbors:
                    break
                nb = pool[idx]
                self.index.add_layer0_edge(v, nb)
                edges_added += 1
                added += 1
        self.index.set_ef(old_ef)
        self._edges_added += edges_added
        self.visit_counts = Counter()
        return edges_added

    def search_enhanced(self, query, k, ef):
        self.index.set_ef(ef)
        labels, _ = self.index.knn_query(query, k=ef)
        self.record_visits(labels.flatten())
        return labels[:, :k]

    def edge_count(self):
        return self._edges_added
