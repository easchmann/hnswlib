"""Target-coherence analysis for Exp52 (clustered hard pool)

This measures spatial coherence directly, using the actual base vector coordinates (not just indices):
  - For each (deduplicated) query, compute its NN centroid: the mean of its k=10 true nearest-neighbour vectors.
  - within_pool_dispersion: mean Euclidean distance between every pair of NN centroids within a pool.
  - centroid_spread: mean distance from each query's NN centroid to the pool-wide mean of all NN centroids.

Computed for TWO pools already present in the same Exp52 dataset build, so no separate dataset (Exp37/38) needs to be fetched:
  - HARD pool: the plateau epochs (20-24), where the schedule mixture is
    100% hard (cluster 35, the target-coherence condition under test).
  - EASY pool: the calibration epochs (0-4), where the schedule mixture is
    100% easy (drawn broadly, not from any single cluster -- the natural
    "dispersed" comparison baseline already available in this same build).
  - RANDOM reference: mean distance between random pairs of base vectors,
    for scale (how large is a "typical" distance in this embedding space).
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml

EXP_DIR = Path(__file__).parent
PLATEAU_EPOCHS = [20, 21, 22, 23, 24]
CALIBRATION_EPOCHS = [0, 1, 2, 3, 4]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_pool(dataset_dir, epochs):
    queries, gts = [], []
    for ep in epochs:
        queries.append(np.load(dataset_dir / f"epoch_{ep:03d}.npy"))
        gts.append(np.load(dataset_dir / f"groundtruth_{ep:03d}.npy"))
    q_all = np.concatenate(queries, axis=0)
    gt_all = np.concatenate(gts, axis=0)
    _, first_idx = np.unique(q_all, axis=0, return_index=True)
    first_idx = np.sort(first_idx)
    return q_all[first_idx], gt_all[first_idx]


def nn_centroids(gt, base):
    """One 128-d centroid per query: mean of its k true-NN vectors."""
    return np.stack([base[row].mean(axis=0) for row in gt], axis=0)


def within_pool_dispersion(centroids, max_pairs=200_000, seed=0):
    """Mean pairwise Euclidean distance between centroids (sampled if large)."""
    n = centroids.shape[0]
    total_pairs = n * (n - 1) // 2
    rng = np.random.default_rng(seed)
    if total_pairs > max_pairs:
        i = rng.integers(0, n, size=max_pairs)
        j = rng.integers(0, n, size=max_pairs)
        mask = i != j
        i, j = i[mask], j[mask]
    else:
        i, j = np.triu_indices(n, k=1)
    dists = np.linalg.norm(centroids[i] - centroids[j], axis=1)
    return float(dists.mean()), int(len(dists))


def centroid_spread(centroids):
    """Mean distance from each centroid to the pool-wide mean centroid."""
    mean_centroid = centroids.mean(axis=0)
    dists = np.linalg.norm(centroids - mean_centroid[None, :], axis=1)
    return float(dists.mean())


def random_base_pair_distance(base, n_pairs=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = base.shape[0]
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    dists = np.linalg.norm(np.asarray(base[i]) - np.asarray(base[j]), axis=1)
    return float(dists.mean())


def analyze_pool(name, dataset_dir, epochs, base):
    q, gt = load_pool(dataset_dir, epochs)
    centroids = nn_centroids(gt, base)
    dispersion, n_pairs_used = within_pool_dispersion(centroids)
    spread = centroid_spread(centroids)
    print(f"    {name}: {q.shape[0]} unique queries; "
          f"within_pool_dispersion={dispersion:.4f} (over {n_pairs_used} pairs)  "
          f"centroid_spread={spread:.4f}")
    return {
        "n_queries_dedup": int(q.shape[0]),
        "within_pool_dispersion": dispersion,
        "n_pairs_used": n_pairs_used,
        "centroid_spread": spread,
    }


def analyze_schedule(schedule_name, cfg, base):
    dataset_dir = EXP_DIR / f"dataset_{schedule_name}"
    print(f"\n=== Exp52 {schedule_name} schedule ===")
    print("  HARD pool (plateau epochs 20-24, cluster 35):")
    hard = analyze_pool("hard", dataset_dir, PLATEAU_EPOCHS, base)
    print("  EASY pool (calibration epochs 0-4, dispersed baseline):")
    easy = analyze_pool("easy", dataset_dir, CALIBRATION_EPOCHS, base)
    ratio_dispersion = hard["within_pool_dispersion"] / easy["within_pool_dispersion"]
    ratio_spread = hard["centroid_spread"] / easy["centroid_spread"]
    print(f"  ratio (hard/easy): dispersion={ratio_dispersion:.4f}  spread={ratio_spread:.4f}")
    return {"hard": hard, "easy": easy,
            "ratio_dispersion_hard_over_easy": ratio_dispersion,
            "ratio_spread_hard_over_easy": ratio_spread}


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else EXP_DIR / "config.yaml"
    cfg = load_config(config_path)

    print(f"Loading base vectors from {cfg['data_dir']}/index_vectors.npy (mmap)...")
    base = np.load(Path(cfg["data_dir"]) / "index_vectors.npy", mmap_mode="r")
    print(f"  base shape: {base.shape}")

    random_ref = random_base_pair_distance(base)
    print(f"\nRandom base-vector pair distance (scale reference): {random_ref:.4f}")

    results = {"random_base_pair_distance": random_ref}
    for schedule_name in ["gradual", "sudden"]:
        results[schedule_name] = analyze_schedule(schedule_name, cfg, base)

    out_path = EXP_DIR / "target_coherence_spatial.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
