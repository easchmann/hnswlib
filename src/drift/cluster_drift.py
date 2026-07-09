"""Cluster-reweighted query drift construction for GloVe-100 experiment."""

import json
import os

import faiss
import h5py
import numpy as np
from sklearn.cluster import KMeans


def load_glove_hdf5(path):
    """Returns (base, query_pool) as float32 arrays, L2-normalised."""
    with h5py.File(path, "r") as f:
        base = f["train"][:].astype(np.float32)
        queries = f["test"][:].astype(np.float32)
    base = base / np.linalg.norm(base, axis=1, keepdims=True)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    return base, queries


def build_query_clusters(query_pool, n_clusters, seed):
    """KMeans cluster labels and centroids for the query pool."""
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(query_pool)
    return labels, km.cluster_centers_


def select_hot_clusters(centroids, n_hot):
    """Return indices of n_hot clusters most distant from the remaining cluster centroid."""
    global_mean = centroids.mean(axis=0)
    dists = np.linalg.norm(centroids - global_mean, axis=1)
    return np.argsort(dists)[-n_hot:].tolist()


def select_sparse_clusters(base, centroids, n_hot, radius=None, k_density=50):
    """Return n_hot cluster indices with lowest base-vector density near centroid.

    Uses mean distance of k_density nearest base vectors as proxy for local density.
    Higher mean distance = sparser region = selected as hot.
    """
    nn_index = faiss.IndexFlatL2(base.shape[1])
    nn_index.add(base.astype(np.float32))
    dists, _ = nn_index.search(centroids.astype(np.float32), k_density)
    mean_dists = dists.mean(axis=1)  # higher mean dist = sparser coverage
    return np.argsort(mean_dists)[-n_hot:].tolist()


def sample_epoch_queries(query_pool, cluster_labels, hot_cluster_ids, t, epoch_size, rng):
    """Sample epoch_size queries with weights interpolated between uniform and hot-only."""
    n_clusters = len(np.unique(cluster_labels))
    hot_set = set(hot_cluster_ids)

    weights = np.zeros(len(query_pool))
    for i, lbl in enumerate(cluster_labels):
        if lbl in hot_set:
            w_uniform = 1.0 / n_clusters
            w_hot = 1.0 / len(hot_cluster_ids)
            weights[i] = (1.0 - t) * w_uniform + t * w_hot
        else:
            weights[i] = (1.0 - t) * (1.0 / n_clusters)

    total = weights.sum()
    if total <= 0:
        raise ValueError(f"All weights zero at t={t}")
    weights /= total

    idx = rng.choice(len(query_pool), size=epoch_size, replace=True, p=weights)
    return query_pool[idx]


def build_cluster_drift_sequence(query_pool, cluster_labels, hot_cluster_ids,
                                  schedule, epoch_size, base_seed):
    """Build list of query arrays, one per epoch, following schedule of t values."""
    epochs = []
    for epoch_idx, t in enumerate(schedule):
        rng = np.random.default_rng(base_seed + epoch_idx)
        q = sample_epoch_queries(
            query_pool, cluster_labels, hot_cluster_ids, t, epoch_size, rng
        )
        epochs.append(q)
    return epochs


def compute_groundtruth(base, queries, k=100, batch_size=1000):
    """Exact k-NN ground truth using FAISS IndexFlatL2."""
    assert base.shape[1] == queries.shape[1]
    index = faiss.IndexFlatL2(base.shape[1])
    index.add(base.astype(np.float32))

    all_ids = []
    n_batches = (len(queries) + batch_size - 1) // batch_size
    for batch_idx, start in enumerate(range(0, len(queries), batch_size)):
        batch = queries[start:start + batch_size].astype(np.float32)
        _, ids = index.search(batch, k)
        all_ids.append(ids)
        if batch_idx % 10 == 0:
            print(f"  groundtruth batch {batch_idx}/{n_batches}")
    return np.concatenate(all_ids, axis=0).astype(np.int32)


def characterise_cluster_drift(base, query_epochs, schedule=None, n_rff=256, rff_seed=0):
    """Compute per-epoch drift diagnostics: OOD distance, MMD², mean NN distance."""
    base_centroid = base.mean(axis=0).astype(np.float64)

    q0 = query_epochs[0].astype(np.float64)
    rng = np.random.default_rng(rff_seed)
    n = len(q0)
    i1 = rng.choice(n, 500, replace=True)
    i2 = rng.choice(n, 500, replace=True)
    sigma = float(np.median(np.sqrt(((q0[i1] - q0[i2]) ** 2).sum(axis=1)))) or 1.0

    W = np.random.default_rng(rff_seed).standard_normal((n_rff, q0.shape[1])) / sigma

    def rff(X):
        proj = X.astype(np.float64) @ W.T
        return np.concatenate([np.cos(proj), np.sin(proj)], axis=1) / np.sqrt(n_rff)

    mu0 = rff(q0).mean(axis=0)

    nn_index = faiss.IndexFlatL2(base.shape[1])
    nn_index.add(base.astype(np.float32))

    results = []
    for i, eq in enumerate(query_epochs):
        ood = float(np.linalg.norm(eq.astype(np.float64).mean(axis=0) - base_centroid))

        mu_i = rff(eq).mean(axis=0)
        mmd2 = float(np.dot(mu_i - mu0, mu_i - mu0))

        n_sample = min(500, len(eq))
        sample = eq[np.random.default_rng(rff_seed + i).choice(len(eq), n_sample, replace=False)]
        dists, _ = nn_index.search(sample.astype(np.float32), 1)
        mean_nn = float(np.sqrt(dists[:, 0]).mean())

        t_val = schedule[i] if schedule is not None else None
        results.append(dict(epoch_idx=i, t_value=t_val, ood_distance=ood,
                            mmd_squared=mmd2, mean_nn_distance=mean_nn))

    print(f"\n{'Epoch':>6}  {'t':>6}  {'OOD':>10}  {'MMD²':>12}  {'MeanNN':>10}")
    for d in results:
        t_str = f"{d['t_value']:.3f}" if d["t_value"] is not None else "N/A"
        print(f"{d['epoch_idx']:>6}  {t_str:>6}  {d['ood_distance']:>10.4f}"
              f"  {d['mmd_squared']:>12.6f}  {d['mean_nn_distance']:>10.4f}")
    print()

    # warn (not error) if any diagnostic is non-monotone — stochastic sampling can cause minor violations
    for key in ("ood_distance", "mmd_squared", "mean_nn_distance"):
        vals = [r[key] for r in results]
        if schedule is not None:
            t_vals = [r["t_value"] for r in results]
            for j in range(1, len(vals)):
                if t_vals[j] > t_vals[j - 1] and vals[j] < vals[j - 1]:
                    print(f"  WARNING: {key} non-monotone at epoch {j} "
                          f"(t={t_vals[j-1]:.3f}→{t_vals[j]:.3f}: "
                          f"{vals[j-1]:.4f}→{vals[j]:.4f})")

    return results


def save_cluster_drift_dataset(path, base, epochs, groundtruth, config, diagnostics):
    """Save dataset arrays and metadata to a directory."""
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "base.npy"), base)
    for i, ep in enumerate(epochs):
        np.save(os.path.join(path, f"epoch_{i:03d}.npy"), ep)
    for i, gt in enumerate(groundtruth):
        np.save(os.path.join(path, f"groundtruth_{i:03d}.npy"), gt)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(path, "diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)

    total_mb = sum(os.path.getsize(os.path.join(path, fn)) for fn in os.listdir(path)) / 1e6
    print(f"Dataset saved to {path}/ — {total_mb:.1f} MB")


def load_cluster_drift_dataset(path):
    """Load dataset saved by save_cluster_drift_dataset; returns dict with base/epochs/groundtruth/config/diagnostics."""
    base = np.load(os.path.join(path, "base.npy"))

    epochs, i = [], 0
    while os.path.exists(os.path.join(path, f"epoch_{i:03d}.npy")):
        epochs.append(np.load(os.path.join(path, f"epoch_{i:03d}.npy")))
        i += 1

    groundtruth, i = [], 0
    while os.path.exists(os.path.join(path, f"groundtruth_{i:03d}.npy")):
        groundtruth.append(np.load(os.path.join(path, f"groundtruth_{i:03d}.npy")))
        i += 1

    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(path, "diagnostics.json")) as f:
        diagnostics = json.load(f)

    return dict(base=base, epochs=epochs, groundtruth=groundtruth,
                config=config, diagnostics=diagnostics)
