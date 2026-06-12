"""NGFix-style local graph repair via conjugate graph edge additions."""

import numpy as np

from src.drift.conjugate_graph import ConjugateGraph
from src.drift.detector import compute_eh_batch, assign_cell, DriftDetector


def find_repair_candidates(hot_cells, cell_labels, eh_per_node=None,
                           eh_threshold_percentile=75.0, max_nodes=1000,
                           last_repaired=None, use_diversity=True):
    """Return node IDs to repair: those in hot_cells with highest EH.

    last_repaired: int array of shape (n_base,), value = last epoch repaired,
      -1 = never repaired. When provided, never-repaired nodes are prioritised
      over recently-repaired ones to maximise unique coverage across events.
    """
    hot_set = set(hot_cells)
    in_hot = np.where(np.isin(cell_labels, list(hot_set)))[0]

    if len(in_hot) == 0:
        return np.empty(0, dtype=np.int64)

    if eh_per_node is None:
        candidates = in_hot
    else:
        threshold = np.percentile(eh_per_node[in_hot], eh_threshold_percentile)
        candidates = in_hot[eh_per_node[in_hot] >= threshold]
        if len(candidates) == 0:
            candidates = in_hot

    effective_lr = last_repaired if use_diversity else None

    if len(candidates) <= max_nodes:
        if effective_lr is None or eh_per_node is None:
            order = (np.argsort(eh_per_node[candidates])[::-1]
                     if eh_per_node is not None else np.arange(len(candidates)))
        else:
            order = _diversity_order(candidates, eh_per_node, effective_lr)
        return candidates[order].astype(np.int64)

    if effective_lr is None or eh_per_node is None:
        order = np.argsort(eh_per_node[candidates])[::-1] if eh_per_node is not None else np.arange(len(candidates))
    else:
        order = _diversity_order(candidates, eh_per_node, effective_lr)
    return candidates[order[:max_nodes]].astype(np.int64)


def _diversity_order(candidates, eh_per_node, last_repaired):
    """Sort candidates: never-repaired first, then oldest-repaired, then by EH desc."""
    lr = last_repaired[candidates]
    # Never repaired (lr==-1) gets priority bucket 0; repaired at epoch e gets bucket e+1
    priority = np.where(lr < 0, 0, lr + 1)
    # lexsort: secondary key first, primary last
    # We want ascending priority (0=best), descending EH within same priority
    return np.lexsort((-eh_per_node[candidates], priority))


def compute_candidate_edges(repair_nodes, index, base, M_candidates=32, ef_search=200,
                             queries=None, result_ids=None):
    """Find candidate edges for each repair node.

    When queries and result_ids are provided, searches from the mean drifted query
    that visited each node — finding neighbors relevant to the current query
    distribution rather than duplicating existing base-space structure.
    Falls back to base[v] when no visiting queries are available.
    """
    # Build node -> visiting query index mapping
    node_to_query_idx = {}
    if queries is not None and result_ids is not None:
        for q_idx, node_ids in enumerate(result_ids):
            for nid in node_ids:
                nid = int(nid)
                node_to_query_idx.setdefault(nid, []).append(q_idx)

    old_ef = index.ef
    index.set_ef(ef_search)

    result = {}
    for i, v in enumerate(repair_nodes):
        if i > 0 and i % 100 == 0:
            print(f"  compute_candidate_edges: {i}/{len(repair_nodes)}")

        q_indices = node_to_query_idx.get(int(v), [])
        if q_indices:
            search_vec = queries[q_indices].mean(axis=0).astype(np.float32)
        else:
            search_vec = base[v]

        labels, distances = index.knn_query(search_vec, k=M_candidates)
        neighbors = [
            (int(nid), float(d))
            for nid, d in zip(labels[0], distances[0])
            if int(nid) != int(v)
        ]
        result[int(v)] = neighbors

    index.set_ef(old_ef)
    return result


def apply_repairs(candidate_edges, conjugate_graph, t_added, epoch_added,
                  current_epoch, base=None, rng_relaxation=1.5):
    """Add candidate edges to conjugate graph with relaxed RNG pruning.

    rng_relaxation <= 0 disables RNG, letting M_conj cap capacity instead.
    With query-guided candidates (all pointing toward the same drifted-query
    cluster), RNG would shadow nearly every edge after the first, so disable
    it by setting rng_relaxation=0 in config.
    """
    edges_attempted = 0
    edges_added = 0
    edges_rejected_rng = 0
    edges_rejected_full = 0

    for src, candidates in candidate_edges.items():
        existing = conjugate_graph._edges.get(src, [])

        for dst, dist in candidates:
            edges_attempted += 1

            shadowed = False
            if rng_relaxation > 0 and existing:
                if base is not None:
                    # Correct RNG: dist_L2(dst, w) < dist_L2(src, dst) * rng_relaxation
                    # dist is squared L2 (from hnswlib), convert for the comparison
                    dist_l2 = dist ** 0.5
                    for e in existing:
                        diff = (base[int(dst)].astype(np.float64)
                                - base[int(e.neighbor_id)].astype(np.float64))
                        if np.dot(diff, diff) ** 0.5 < dist_l2 * rng_relaxation:
                            shadowed = True
                            break
                else:
                    # Proxy fallback (no base available)
                    for e in existing:
                        if e.distance < dist * rng_relaxation:
                            shadowed = True
                            break

            if shadowed:
                edges_rejected_rng += 1
                continue

            added = conjugate_graph.add_edge(
                src, dst, dist,
                t_added=t_added,
                epoch_added=epoch_added,
                current_epoch=current_epoch,
            )
            if added:
                edges_added += 1
                existing = conjugate_graph._edges.get(src, [])
            else:
                edges_rejected_full += 1

    return {
        "edges_attempted": edges_attempted,
        "edges_added": edges_added,
        "edges_rejected_rng": edges_rejected_rng,
        "edges_rejected_full": edges_rejected_full,
    }


def update_node_eh_accumulator(result_ids_batch, eh_values, node_eh_accumulator):
    """EMA update: each node accumulates EH of queries it appeared in."""
    for i, result_ids in enumerate(result_ids_batch):
        eh = float(eh_values[i])
        for j in result_ids:
            node_eh_accumulator[j] = 0.9 * node_eh_accumulator[j] + 0.1 * eh


class AdaptationManager:
    """Orchestrates drift detection and conjugate graph repair per epoch."""

    def __init__(self, index, base, conjugate_graph, detector, cell_labels,
                 centroids, config):
        self.index = index
        self.base = base
        self.conjugate_graph = conjugate_graph
        self.detector = detector
        self.cell_labels = cell_labels
        self._centroids = centroids
        self.config = config
        self.node_eh_accumulator = np.zeros(base.shape[0], dtype=np.float64)
        self._node_repair_epoch = np.full(base.shape[0], -1, dtype=np.int32)
        self.current_epoch = 0
        self._use_diversity = config.get("use_diversity", True)
        self._two_hop = config.get("two_hop", True)

    def process_epoch(self, queries, result_ids, result_distances, k):
        """Run one epoch: compute EH, update detector, optionally repair."""
        n_queries = queries.shape[0]

        # 1. result vectors shape (n_queries, k, dim)
        result_vectors = self.base[result_ids]

        # 2. EH per query
        eh_values = compute_eh_batch(list(result_vectors))

        # 3. Assign queries to cells
        cell_ids = assign_cell(queries, self._centroids)

        # 4. Update detector buffer
        self.detector.update_batch(eh_values, cell_ids)

        # 5. Update node EH accumulator
        update_node_eh_accumulator(result_ids, eh_values, self.node_eh_accumulator)

        # 6. Check for drift
        drift_result = self.detector.check_drift()
        drift_detected = drift_result is not None and drift_result["drift_detected"]
        mmd_sq = drift_result["mmd_squared"] if drift_result is not None else None
        hot_cells = drift_result["hot_cells"] if drift_result is not None else []

        # 7. Repair if drift detected
        repair_stats = None
        if drift_detected and hot_cells:
            repair_nodes = find_repair_candidates(
                hot_cells,
                self.cell_labels,
                self.node_eh_accumulator,
                max_nodes=self.config.get("max_repair_nodes", 1000),
                last_repaired=self._node_repair_epoch,
                use_diversity=self._use_diversity,
            )
            if len(repair_nodes) > 0:
                self._node_repair_epoch[repair_nodes] = self.current_epoch
                candidates = compute_candidate_edges(
                    repair_nodes,
                    self.index,
                    self.base,
                    M_candidates=self.config.get("M_candidates", 32),
                    ef_search=self.config.get("repair_ef_search", 200),
                    queries=queries,
                    result_ids=result_ids,
                )
                repair_stats = apply_repairs(
                    candidates,
                    self.conjugate_graph,
                    t_added=self.current_epoch / max(1, self.config.get("n_epochs", 25)),
                    epoch_added=self.current_epoch,
                    current_epoch=self.current_epoch,
                    base=self.base,
                    rng_relaxation=self.config.get("rng_relaxation", 1.5),
                )
                print(
                    f"Epoch {self.current_epoch}: repair {repair_stats['edges_added']} edges "
                    f"({repair_stats['edges_attempted']} attempted, "
                    f"{len(repair_nodes)} nodes)"
                )

        self.current_epoch += 1

        return {
            "drift_detected": drift_detected,
            "mmd_squared": mmd_sq,
            "hot_cells": hot_cells,
            "repair_stats": repair_stats,
            "mean_eh_this_epoch": float(np.mean(eh_values)),
            "n_queries": n_queries,
        }

    def search_enhanced(self, query_vectors, k, ef_search):
        """Primary HNSW search augmented with conjugate graph one-hop expansion."""
        self.index.set_ef(ef_search)
        n = query_vectors.shape[0]
        all_ids = np.empty((n, k), dtype=np.int64)
        all_dists = np.empty((n, k), dtype=np.float32)

        for i, q in enumerate(query_vectors):
            labels, distances = self.index.knn_query(q, k=k)
            ids_aug, dists_aug = self.conjugate_graph.enhanced_search(
                labels[0], distances[0], self.base, q, k, self.current_epoch,
                two_hop=self._two_hop
            )
            all_ids[i] = ids_aug
            all_dists[i] = dists_aug

        return all_ids, all_dists
