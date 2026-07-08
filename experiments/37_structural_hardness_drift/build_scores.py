"""Score all 5M queries for structural hardness (exp37).

Produces structural_scores.npy with shape (5M, 3):
  col 0: EH at ef_score_low  (spread of k returned neighbor vectors)
  col 1: recall@k at ef_score_low
  col 2: recall@k at ef_score_high
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hnswlib
from src.drift.detector import compute_eh_batch

BATCH_SIZE = 2000
CHECKPOINT_EVERY = 500_000


def main():
    config_path = sys.argv[1]
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg["data_dir"]
    index_path = cfg["index_path"]
    scores_path = cfg["scores_path"]
    ef_low = cfg["ef_score_low"]
    ef_high = cfg["ef_score_high"]
    k = cfg["k"]

    print(f"Loading data from {data_dir}...")
    base = np.load(os.path.join(data_dir, "index_vectors.npy"), mmap_mode='r')
    queries = np.load(os.path.join(data_dir, "queries_by_hardness.npy"), mmap_mode='r')
    gt = np.load(os.path.join(data_dir, "ground_truth.npy"))
    print(f"  base: {base.shape}  queries: {queries.shape}  gt: {gt.shape}")

    print(f"Loading HNSW index from {index_path}...")
    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.load_index(index_path, max_elements=base.shape[0])
    print(f"  index loaded ({base.shape[0]:,} elements)")

    n_queries = len(queries)
    scores = np.zeros((n_queries, 3), dtype=np.float32)

    print(f"\nScoring {n_queries:,} queries (batch={BATCH_SIZE}, "
          f"ef_low={ef_low}, ef_high={ef_high}, k={k})...")
    t_start = time.time()

    for start in range(0, n_queries, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_queries)
        batch = np.ascontiguousarray(queries[start:end], dtype=np.float32)
        batch_size = end - start

        index.set_ef(ef_low)
        ids_low, _ = index.knn_query(batch, k=k)
        neighbor_vecs = [base[ids_low[j]] for j in range(batch_size)]
        eh_vals = compute_eh_batch(neighbor_vecs)
        recall_low = np.array([
            len(set(ids_low[j].tolist()) & set(gt[start + j].tolist())) / k
            for j in range(batch_size)
        ], dtype=np.float32)

        index.set_ef(ef_high)
        ids_high, _ = index.knn_query(batch, k=k)
        recall_high = np.array([
            len(set(ids_high[j].tolist()) & set(gt[start + j].tolist())) / k
            for j in range(batch_size)
        ], dtype=np.float32)

        scores[start:end, 0] = eh_vals
        scores[start:end, 1] = recall_low
        scores[start:end, 2] = recall_high

        if end % CHECKPOINT_EVERY == 0 or end == n_queries:
            # Save only processed rows so shape[0] < 5M signals incompleteness
            np.save(scores_path, scores[:end])
            elapsed = time.time() - t_start
            rate = end / elapsed if elapsed > 0 else 1.0
            remaining = (n_queries - end) / rate
            print(f"  {end:,}/{n_queries:,} ({100.0 * end / n_queries:.1f}%)  "
                  f"elapsed={elapsed / 60:.1f}m  eta={remaining / 60:.1f}m")

    print(f"\nDone. structural_scores.npy saved to {scores_path}")
    print(f"  shape={scores.shape}  "
          f"eh mean={scores[:, 0].mean():.4f}  "
          f"recall32 mean={scores[:, 1].mean():.4f}  "
          f"recall256 mean={scores[:, 2].mean():.4f}")


if __name__ == "__main__":
    main()
