"""SIFT cluster-concentrated query drift: clustering and routing failure diagnostics."""

import numpy as np
from sklearn.cluster import KMeans


def cluster_query_pool(query_vecs, n_clusters, seed):
    """K-means cluster query vectors. Returns (labels, centroids, diagnostics_dict)."""
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(query_vecs)
    cluster_sizes = [int((labels == c).sum()) for c in range(n_clusters)]
    diag = {"n_clusters": n_clusters, "cluster_sizes": cluster_sizes, "seed": seed}
    return labels, km.cluster_centers_.astype(np.float32), diag


def get_entry_point_vec(index, base_vecs):
    """Return (vector, method_str) of the HNSW entry point node."""
    try:
        ep_id = index.appr_alg.enterpoint_node_
        ep_vec = base_vecs[ep_id].astype(np.float32)
        print(f"  Entry point: node {ep_id} (from index.appr_alg.enterpoint_node_)")
        return ep_vec, "enterpoint_node"
    except AttributeError:
        rng = np.random.default_rng(0)
        sample = base_vecs[rng.integers(0, len(base_vecs), 50000)]
        centroid = sample.mean(axis=0).astype(np.float32)
        print("  Entry point: base centroid (fallback, enterpoint_node_ unavailable)")
        return centroid, "base_centroid_sample"


def select_far_cluster(centroids, reference_vec):
    """Return (chosen_idx, diagnostics_dict) for cluster farthest from reference_vec."""
    ref = np.asarray(reference_vec, dtype=np.float32)
    diffs = centroids - ref
    dists = np.sqrt((diffs ** 2).sum(axis=1)).tolist()
    chosen_idx = int(np.argmax(dists))
    diag = {
        "chosen_cluster_idx": chosen_idx,
        "all_distances_to_reference": dists,
        "chosen_distance_to_reference": dists[chosen_idx],
    }
    return chosen_idx, diag


def routing_failure_diagnostic(index, base_vecs, uniform_queries, far_cluster_queries,
                                ef_values, k, groundtruth_func):
    """Check whether routing failure exists at low ef; returns list of dicts."""
    import faiss

    gt_uniform = groundtruth_func(base_vecs, uniform_queries, k=k)
    gt_far = groundtruth_func(base_vecs, far_cluster_queries, k=k)

    rows = []
    print(f"\n{'ef':>6}  {'recall_uniform':>14}  {'recall_far':>12}  {'gap':>8}")
    for ef in ef_values:
        index.set_ef(ef)

        ids_u, _ = index.knn_query(uniform_queries, k=k, num_threads=1)
        hits_u = [
            len(set(ids_u[i].tolist()) & set(gt_uniform[i, :k].tolist())) / k
            for i in range(len(uniform_queries))
        ]
        recall_u = float(np.mean(hits_u))

        ids_f, _ = index.knn_query(far_cluster_queries, k=k, num_threads=1)
        hits_f = [
            len(set(ids_f[i].tolist()) & set(gt_far[i, :k].tolist())) / k
            for i in range(len(far_cluster_queries))
        ]
        recall_f = float(np.mean(hits_f))

        gap = recall_u - recall_f
        print(f"{ef:>6}  {recall_u:>14.4f}  {recall_f:>12.4f}  {gap:>8.4f}")
        rows.append({
            "ef": ef,
            "recall_uniform": recall_u,
            "recall_far_cluster": recall_f,
            "gap": gap,
        })

    print()
    return rows
