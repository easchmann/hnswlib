"""Proactive node promotion: elevate representative hot-cell nodes into HNSW upper layers."""

import numpy as np


class NodePromoter:
    """Elevates base-layer nodes into HNSW upper layers to fix navigational failures.

    Selects nodes that are most central to hot-cell regions and promotes them
    via index.promote_node so greedy descent encounters them during upper-layer traversal.
    """

    def __init__(self, index, base, centroids, cell_labels, config):
        self.index = index
        self.base = base
        self.centroids = centroids
        self.cell_labels = cell_labels
        self.n_promote = config.get("n_promote", 200)
        self.promote_layer = config.get("promote_layer", 1)
        self.promote_refresh = config.get("promote_refresh", False)
        self._promoted = set()

    def select_promotion_candidates(self, hot_cells, k=None):
        """Return top-k node IDs most central to hot cells, ranked by centroid proximity."""
        if k is None:
            k = self.n_promote

        hot_set = set(hot_cells)
        candidates = []

        for cell_id in hot_set:
            cell_mask = np.where(self.cell_labels == cell_id)[0]

            # Unless refreshing, skip already-promoted nodes
            if not self.promote_refresh and self._promoted:
                promoted_arr = np.array(list(self._promoted), dtype=np.int64)
                cell_mask = cell_mask[~np.isin(cell_mask, promoted_arr)]

            if len(cell_mask) == 0:
                continue

            centroid = self.centroids[cell_id].astype(np.float64)
            vecs = self.base[cell_mask].astype(np.float64)
            diffs = vecs - centroid[np.newaxis, :]
            dists = np.sum(diffs ** 2, axis=1)  # squared L2; relative order is the same
            order = np.argsort(dists)
            candidates.extend(cell_mask[order].tolist())

        if len(candidates) == 0:
            return np.empty(0, dtype=np.int64)

        return np.array(candidates[:k], dtype=np.int64)

    def promote(self, candidates, target_layer=None):
        """Call index.promote_node for each candidate; return count of promotions."""
        if target_layer is None:
            target_layer = self.promote_layer

        count = 0
        for node_id in candidates:
            nid = int(node_id)
            self.index.promote_node(nid, target_layer)
            self._promoted.add(nid)
            count += 1

        return count

    @property
    def n_promoted(self):
        return len(self._promoted)
