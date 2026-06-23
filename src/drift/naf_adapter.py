"""Conjugate graph guided by node access frequency."""

from collections import Counter
import numpy as np


class NAFAdapter:
    """Conjugate graph guided by node access frequency."""

    def __init__(self, index, base, hot_k, ef_repair, n_neighbors, M_conj, faiss_index=None):
        self.index = index
        self.base = base
        self.hot_k = hot_k
        self.ef_repair = ef_repair
        self.n_neighbors = n_neighbors
        self.M_conj = M_conj
        self.faiss_index = faiss_index  # when set, used instead of HNSW for pool retrieval
        self.visit_counts = Counter()
        self.conjugate_edges = {}

    def record_visits(self, labels):
        """Increment visit count for each visited node ID."""
        for nid in labels:
            self.visit_counts[int(nid)] += 1

    def repair(self):
        """Exact-rerank local pools for top visited nodes; add conjugate edges."""
        top_nodes = [nid for nid, _ in self.visit_counts.most_common(self.hot_k)]
        old_ef = self.index.ef
        self.index.set_ef(self.ef_repair)
        edges_added = 0
        for v in top_nodes:
            query = self.base[v:v + 1]
            if self.faiss_index is not None:
                _, faiss_ids = self.faiss_index.search(query.astype(np.float32), self.ef_repair + 1)
                pool = [int(x) for x in faiss_ids[0] if int(x) != v and int(x) != -1]
            else:
                labels, _ = self.index.knn_query(query, k=self.ef_repair)
                pool = [int(x) for x in labels[0] if int(x) != v]
            if not pool:
                continue
            dists = np.sum(
                (self.base[pool].astype(np.float64) - query.astype(np.float64)) ** 2,
                axis=1,
            )
            existing = set(self.conjugate_edges.get(v, []))
            added = 0
            for idx in np.argsort(dists):
                if added >= self.n_neighbors or len(existing) >= self.M_conj:
                    break
                nbr = pool[idx]
                if nbr in existing:
                    continue
                self.conjugate_edges.setdefault(v, []).append(nbr)
                existing.add(nbr)
                edges_added += 1
                added += 1
        self.index.set_ef(old_ef)
        self.visit_counts = Counter()
        return edges_added

    def search_enhanced(self, query, k, ef):
        """HNSW search expanded with conjugate neighbours, exact-rescored."""
        self.index.set_ef(ef)
        labels, _ = self.index.knn_query(query, k=ef)
        flat = labels.flatten()
        self.record_visits(flat)
        candidates = set(int(x) for x in flat)
        for nid in flat:
            for nbr in self.conjugate_edges.get(int(nid), []):
                candidates.add(nbr)
        q = query[0].astype(np.float64)
        cand_list = list(candidates)
        dists = np.sum((self.base[cand_list].astype(np.float64) - q) ** 2, axis=1)
        order = np.argsort(dists)[:k]
        return np.array([[cand_list[i] for i in order]], dtype=np.int64)

    def edge_count(self):
        """Total conjugate edges stored."""
        return sum(len(v) for v in self.conjugate_edges.values())
