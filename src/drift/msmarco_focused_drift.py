"""Select the tightest semantic cluster from the OOD query pool."""

import numpy as np


def select_focused_ood_cluster(ood_vecs, n_clusters, seed):
    """Cluster OOD query embeddings and return the tightest cluster.

    Returns (cluster_vecs, cluster_idx, diagnostics_dict).
    """
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(ood_vecs)
    centroids = km.cluster_centers_

    mean_dists = []
    for k in range(n_clusters):
        members = ood_vecs[labels == k]
        if len(members) == 0:
            mean_dists.append(float('inf'))
            continue
        dists = np.linalg.norm(members - centroids[k], axis=1)
        mean_dists.append(float(dists.mean()))

    chosen = int(np.argmin(mean_dists))
    cluster_vecs = ood_vecs[labels == chosen].astype(np.float32)

    sizes = [int((labels == k).sum()) for k in range(n_clusters)]
    diagnostics = {
        "n_clusters": n_clusters,
        "cluster_sizes": sizes,
        "chosen_cluster_idx": chosen,
        "chosen_cluster_size": len(cluster_vecs),
        "chosen_cluster_mean_dist_to_centroid": mean_dists[chosen],
        "all_cluster_mean_dists": mean_dists,
        "centroid": centroids[chosen].tolist(),
    }
    return cluster_vecs, chosen, diagnostics


def select_top_n_ood_queries(ood_vecs, centroid, n):
    """Return the n OOD queries closest to centroid by L2 distance.

    Returns (selected_vecs, diagnostics_dict).
    """
    centroid = np.asarray(centroid, dtype=np.float32)
    dists = np.linalg.norm(ood_vecs - centroid, axis=1)
    idx = np.argsort(dists)[:n]
    selected = ood_vecs[idx].astype(np.float32)
    diagnostics = {
        "n_requested": n,
        "n_returned": len(idx),
        "mean_dist_to_centroid": float(dists[idx].mean()),
        "max_dist_to_centroid": float(dists[idx].max()),
        "min_dist_to_centroid": float(dists[idx].min()),
    }
    return selected, diagnostics
