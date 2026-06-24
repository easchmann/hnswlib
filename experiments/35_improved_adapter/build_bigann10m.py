"""Build HNSW index and rotation-drift datasets for BIGANN-10M (uint8 bvecs, ~10.6M vectors).

No separate BIGANN query file is available on the cluster. A query pool of 10K vectors is
sampled from the base (fixed seed). Because these queries ARE in the index, pre-drift recall
is essentially 1.0 by construction; recall dynamics under rotation drift are unaffected.
"""

import sys
import time
from pathlib import Path

import faiss
import numpy as np
import hnswlib
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.drift.dataset import (
    build_rotation_drift_sequence,
    characterise_drift,
    load_bvecs,
    make_rotation_generator,
    save_drift_dataset,
)

OUT_DIR = Path(__file__).parent / "bigann10m"

DATA_BASE   = "/scratch/easchman/thesis/data/sift/bigann_base_10M.bvecs"
N_BASE      = 10_000_000   # use first 10M of the 10.6M available
N_QUERIES   = 10_000
QUERY_SEED  = 42            # seed for sampling queries from base
M           = 16
EF_CONSTRUCTION = 200
INDEX_SEED  = 1234
DRIFT_SEED  = 42
EPOCH_SIZE  = 500
N_EPOCHS    = 25
K_GT        = 100

SCHEDULE_GRADUAL = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.125, 0.25, 0.375, 0.5,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    0.625, 0.75, 0.875, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0,
]
SCHEDULE_SUDDEN = [0.0] * 13 + [1.0] * 12


def build_index(base, out_path):
    if out_path.exists():
        print(f"Index already exists at {out_path} — skipping.")
        return
    print(f"Building HNSW index ({base.shape[0]} vectors, M={M}, ef_construction={EF_CONSTRUCTION})...")
    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.init_index(max_elements=base.shape[0], ef_construction=EF_CONSTRUCTION,
                     M=M, random_seed=INDEX_SEED)
    t0 = time.perf_counter()
    index.add_items(base, num_threads=-1)
    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.1f}s")
    index.save_index(str(out_path))
    print(f"  saved to {out_path}")


def build_dataset(base, queries, schedule, schedule_name, out_path):
    if (out_path / "base.npy").exists():
        print(f"Dataset {schedule_name} already exists at {out_path} — skipping.")
        return
    A = make_rotation_generator(base.shape[1], DRIFT_SEED)
    print(f"Building drift sequence ({schedule_name}, {N_EPOCHS} epochs)...")
    epochs = build_rotation_drift_sequence(queries, A, schedule, EPOCH_SIZE, DRIFT_SEED)

    print("Characterising drift...")
    diagnostics = characterise_drift(base, epochs, schedule=schedule)

    # Build FAISS index once and reuse for all epochs — critical for 10M base
    print("Building FAISS exact index for ground truth (built once, reused across epochs)...")
    flat = faiss.IndexFlatL2(base.shape[1])
    flat.add(base.astype(np.float32))
    print(f"  FAISS index ready ({base.shape[0]:,} vectors)")

    groundtruth_per_epoch = []
    for i, epoch_queries in enumerate(epochs):
        print(f"  GT epoch {i+1}/{N_EPOCHS}")
        _, ids = flat.search(epoch_queries.astype(np.float32), K_GT)
        groundtruth_per_epoch.append(ids.astype(np.int32))

    config = {
        "data": {"base": "bigann_base_10M.bvecs", "n_base": N_BASE,
                 "query": "sampled_from_base", "n_queries": N_QUERIES,
                 "query_seed": QUERY_SEED},
        "index": {"M": M, "ef_construction": EF_CONSTRUCTION},
        "drift": {"seed": DRIFT_SEED, "epoch_size": EPOCH_SIZE,
                  "n_epochs": N_EPOCHS, "schedule": schedule_name},
        "n_epochs": N_EPOCHS,
    }
    save_drift_dataset(str(out_path), base, epochs, groundtruth_per_epoch, config, diagnostics)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading base: {DATA_BASE}  (n_vectors={N_BASE:,})")
    t0 = time.perf_counter()
    base = load_bvecs(DATA_BASE, n_vectors=N_BASE)
    print(f"  shape: {base.shape}  loaded in {time.perf_counter()-t0:.1f}s")

    # Sample query pool from base with fixed seed
    rng = np.random.default_rng(QUERY_SEED)
    query_indices = rng.choice(N_BASE, size=N_QUERIES, replace=False)
    queries = base[query_indices].copy()
    print(f"Sampled {N_QUERIES} queries from base (seed={QUERY_SEED})")
    print(f"  queries shape: {queries.shape}")

    build_index(base, OUT_DIR / "index.bin")

    build_dataset(base, queries, SCHEDULE_GRADUAL, "gradual",
                  OUT_DIR / "dataset_gradual")
    build_dataset(base, queries, SCHEDULE_SUDDEN, "sudden",
                  OUT_DIR / "dataset_sudden")

    print("Done.")


if __name__ == "__main__":
    main()
