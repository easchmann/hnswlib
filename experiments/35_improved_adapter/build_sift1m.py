"""Build HNSW index and rotation-drift datasets for SIFT1M (sift_base.fvecs, 1M vectors)."""

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
    load_fvecs,
    make_rotation_generator,
    save_drift_dataset,
)

OUT_DIR = Path(__file__).parent / "sift1m"

DATA_BASE   = "/scratch/easchman/thesis/data/sift/sift_base.fvecs"
DATA_QUERY  = "/scratch/easchman/thesis/data/sift/sift_query.fvecs"
N_BASE      = 1_000_000
N_QUERIES   = 10_000
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
SCHEDULE_SUDDEN = (
    [0.0] * 13 + [1.0] * 12
)


def build_index(base, out_path):
    if out_path.exists():
        print(f"Index already exists at {out_path} — skipping.")
        return
    print(f"Building HNSW index ({N_BASE} vectors, M={M}, ef_construction={EF_CONSTRUCTION})...")
    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.init_index(max_elements=N_BASE, ef_construction=EF_CONSTRUCTION,
                     M=M, random_seed=INDEX_SEED)
    t0 = time.perf_counter()
    index.add_items(base, num_threads=-1)
    print(f"  done in {time.perf_counter()-t0:.1f}s")
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

    print("Building FAISS index for ground truth (reused across all epochs)...")
    flat = faiss.IndexFlatL2(base.shape[1])
    flat.add(base.astype(np.float32))

    groundtruth_per_epoch = []
    for i, epoch_queries in enumerate(epochs):
        print(f"  GT epoch {i+1}/{N_EPOCHS}")
        _, ids = flat.search(epoch_queries.astype(np.float32), K_GT)
        groundtruth_per_epoch.append(ids.astype(np.int32))

    config = {
        "data": {"base": "sift_base.fvecs", "n_base": N_BASE,
                 "query": "sift_query.fvecs", "n_queries": N_QUERIES},
        "index": {"M": M, "ef_construction": EF_CONSTRUCTION},
        "drift": {"seed": DRIFT_SEED, "epoch_size": EPOCH_SIZE,
                  "n_epochs": N_EPOCHS, "schedule": schedule_name},
        "n_epochs": N_EPOCHS,
    }
    save_drift_dataset(str(out_path), base, epochs, groundtruth_per_epoch, config, diagnostics)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading base: {DATA_BASE}")
    base = load_fvecs(DATA_BASE, n_vectors=N_BASE)
    print(f"  shape: {base.shape}")

    print(f"Loading queries: {DATA_QUERY}")
    queries = load_fvecs(DATA_QUERY, n_vectors=N_QUERIES)
    print(f"  shape: {queries.shape}")

    build_index(base, OUT_DIR / "index.bin")

    build_dataset(base, queries, SCHEDULE_GRADUAL, "gradual",
                  OUT_DIR / "dataset_gradual")
    build_dataset(base, queries, SCHEDULE_SUDDEN, "sudden",
                  OUT_DIR / "dataset_sudden")

    print("Done.")


if __name__ == "__main__":
    main()
