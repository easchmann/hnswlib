import json
import os

import faiss
import numpy as np
import yaml
from scipy.linalg import expm

_rotation_cache = {}


# numpy subclass so each epoch array can carry its drift parameter t
class _DriftArray(np.ndarray):
    def __new__(cls, array, t=None):
        obj = np.asarray(array).view(cls)
        obj.t = t
        return obj

    def __array_finalize__(self, obj):
        self.t = getattr(obj, "t", None)


def load_fvecs(path, n_vectors=None):
    vectors = []
    with open(path, "rb") as f:
        while True:
            dim_buf = f.read(4)
            if not dim_buf:
                break
            dim = int(np.frombuffer(dim_buf, dtype=np.int32)[0])
            vec = np.frombuffer(f.read(dim * 4), dtype=np.float32)
            vectors.append(vec)
            if n_vectors is not None and len(vectors) >= n_vectors:
                break
    return np.stack(vectors)


def make_rotation_generator(dim, seed):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((dim, dim))
    A = M - M.T  # skew-symmetric: A = -A^T
    # Normalise by spectral norm and scale to pi/2 so that expm(1*A) is a
    # 90-degree rotation in the principal plane — enough to meaningfully
    # degrade recall. Frobenius normalisation would give only ~5 degrees in
    # 128D, which leaves recall virtually unchanged.
    sigma_max = np.linalg.norm(A, ord=2)
    A = A * np.pi / sigma_max
    return A.astype(np.float64)


def rotate_queries(queries, A, t):
    key = float(t)
    if key not in _rotation_cache:
        _rotation_cache[key] = expm(t * A).astype(np.float64)
    R = _rotation_cache[key]
    return (queries.astype(np.float64) @ R.T).astype(np.float32)


def build_rotation_drift_sequence(queries, A, schedule, epoch_size, seed=42):
    total = len(schedule)
    epochs = []
    for i, t in enumerate(schedule):
        rng = np.random.default_rng(seed + i)
        idx = rng.choice(len(queries), size=epoch_size, replace=True)
        sampled = queries[idx]
        rotated = rotate_queries(sampled, A, t)
        shift = float(np.linalg.norm(rotated.mean(axis=0) - sampled.mean(axis=0)))
        print(f"Epoch {i}/{total}: t={t:.3f}, centroid shift={shift:.4f}")
        epochs.append(_DriftArray(rotated, t=t))
    return epochs


def compute_groundtruth(base, queries, k=100, batch_size=1000):
    assert base.shape[1] == queries.shape[1], (
        f"Dimension mismatch: base {base.shape[1]} vs queries {queries.shape[1]}"
    )
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


def characterise_drift(base, query_epochs, schedule=None, n_rff=256, rff_seed=0):
    base_centroid = base.mean(axis=0).astype(np.float64)

    # estimate RBF bandwidth as median pairwise distance in epoch 0 (500 random pairs)
    q0 = query_epochs[0].astype(np.float64)
    rng = np.random.default_rng(rff_seed)
    n = len(q0)
    i1, i2 = rng.choice(n, 500, replace=True), rng.choice(n, 500, replace=True)
    sigma = float(np.median(np.sqrt(((q0[i1] - q0[i2]) ** 2).sum(axis=1)))) or 1.0

    # random Fourier features: omega ~ N(0, 1/sigma^2)
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

        # sample for NN distance to keep this cheap
        n_sample = min(500, len(eq))
        sample = eq[np.random.default_rng(rff_seed + i).choice(len(eq), n_sample, replace=False)]
        dists, _ = nn_index.search(sample.astype(np.float32), 1)
        mean_nn = float(np.sqrt(dists[:, 0]).mean())

        t_val = schedule[i] if schedule is not None else getattr(eq, "t", None)
        results.append(dict(epoch_idx=i, t_value=t_val, ood_distance=ood,
                            mmd_squared=mmd2, mean_nn_distance=mean_nn))

    print(f"\n{'Epoch':>6}  {'t':>6}  {'OOD':>10}  {'MMD²':>12}  {'MeanNN':>10}")
    for d in results:
        t_str = f"{d['t_value']:.3f}" if d["t_value"] is not None else "N/A"
        print(f"{d['epoch_idx']:>6}  {t_str:>6}  {d['ood_distance']:>10.4f}"
              f"  {d['mmd_squared']:>12.6f}  {d['mean_nn_distance']:>10.4f}")
    print()
    return results


def save_drift_dataset(path, base, epochs, groundtruth_per_epoch, config, diagnostics):
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "base.npy"), base)
    for i, ep in enumerate(epochs):
        np.save(os.path.join(path, f"epoch_{i:03d}.npy"), ep)
    for i, gt in enumerate(groundtruth_per_epoch):
        np.save(os.path.join(path, f"gt_{i:03d}.npy"), gt)
    with open(os.path.join(path, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    with open(os.path.join(path, "diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)

    total_mb = sum(os.path.getsize(os.path.join(path, fn)) for fn in os.listdir(path)) / 1e6
    print(f"Dataset saved to {path}/ — {total_mb:.1f} MB")


def load_drift_dataset(path):
    base = np.load(os.path.join(path, "base.npy"))

    epochs, i = [], 0
    while os.path.exists(os.path.join(path, f"epoch_{i:03d}.npy")):
        epochs.append(np.load(os.path.join(path, f"epoch_{i:03d}.npy")))
        i += 1

    groundtruth, i = [], 0
    while os.path.exists(os.path.join(path, f"gt_{i:03d}.npy")):
        groundtruth.append(np.load(os.path.join(path, f"gt_{i:03d}.npy")))
        i += 1

    with open(os.path.join(path, "config.yaml")) as f:
        config = yaml.safe_load(f)
    with open(os.path.join(path, "diagnostics.json")) as f:
        diagnostics = json.load(f)

    if "n_epochs" in config and not (len(epochs) == len(groundtruth) == config["n_epochs"]):
        raise ValueError(
            f"Length mismatch: {len(epochs)} epochs, {len(groundtruth)} GT arrays, "
            f"config says {config['n_epochs']}")
    return dict(base=base, epochs=epochs, groundtruth=groundtruth, config=config, diagnostics=diagnostics)
