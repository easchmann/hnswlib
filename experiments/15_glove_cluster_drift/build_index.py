"""Build HNSW index on GloVe-100 base vectors."""

import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hnswlib

from src.drift.cluster_drift import load_glove_hdf5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    hdf5_path = cfg["data"]["hdf5_path"]
    idx_cfg = cfg["index"]
    out_path = str(ROOT / cfg["output"]["index_path"])

    print(f"Loading base vectors from {hdf5_path}...")
    base, _ = load_glove_hdf5(hdf5_path)
    print(f"  base: {base.shape}")

    print(f"\nBuilding HNSW index (M={idx_cfg['M']}, ef_construction={idx_cfg['ef_construction']})...")
    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.init_index(
        max_elements=base.shape[0],
        ef_construction=idx_cfg["ef_construction"],
        M=idx_cfg["M"],
        random_seed=idx_cfg["seed"],
    )
    index.add_items(base, num_threads=-1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    index.save_index(out_path)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nIndex saved to {out_path}")
    print(f"  elements:       {index.get_current_count():,}")
    print(f"  M:              {idx_cfg['M']}")
    print(f"  ef_construction:{idx_cfg['ef_construction']}")
    print(f"  file size:      {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
