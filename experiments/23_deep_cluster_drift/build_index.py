"""Build HNSW index for experiment 23: DEEP-1B cluster drift."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hnswlib

from src.drift.deep_drift import load_deep_hdf5


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    args = parser.parse_args()

    cfg = _load_config(args.config)
    index_cfg = cfg["index"]
    index_path = str(ROOT / index_cfg["path"])
    n_base = cfg["data"].get("n_base", None)
    num_threads = index_cfg.get("num_threads", 1)

    print(f"Loading base vectors (n_base={n_base})...")
    t0 = time.perf_counter()
    train, _ = load_deep_hdf5(cfg["data"]["hdf5_path"], n_base=n_base)
    print(f"  loaded: {train.shape}  ({(time.perf_counter() - t0):.1f}s)")

    dim = train.shape[1]
    n = len(train)
    print(f"\nBuilding HNSW index: {n:,} vectors, dim={dim}, M={index_cfg['M']}, "
          f"ef_construction={index_cfg['ef_construction']}, threads={num_threads}")

    index = hnswlib.Index(space=index_cfg["space"], dim=dim)
    index.init_index(
        max_elements=n,
        ef_construction=index_cfg["ef_construction"],
        M=index_cfg["M"],
        random_seed=index_cfg["seed"],
    )
    index.set_num_threads(num_threads)

    batch_size = 100_000
    t0 = time.perf_counter()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        index.add_items(train[start:end], ids=np.arange(start, end))
        elapsed = time.perf_counter() - t0
        rate = end / elapsed
        eta = (n - end) / rate if rate > 0 else 0
        print(f"  {end:>8,}/{n:,}  ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    total = time.perf_counter() - t0
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    index.save_index(index_path)
    print(f"\nDone. {n:,} vectors indexed in {total:.1f}s")
    print(f"Index saved to {index_path}")


if __name__ == "__main__":
    main()
