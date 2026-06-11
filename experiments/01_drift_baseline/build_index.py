import argparse
import sys
import time
from pathlib import Path

import hnswlib
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.drift.dataset import load_fvecs


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build(config_path, force=False):
    cfg = load_config(config_path)
    index_path = Path(ROOT / cfg["output"]["index_path"])

    if index_path.exists() and not force:
        print(f"Index already exists at {index_path} — skipping. Pass --force to rebuild.")
        return

    n_base = cfg["data"]["n_base"]
    M = cfg["index"]["M"]
    ef_construction = cfg["index"]["ef_construction"]

    print(f"Loading {n_base} base vectors...")
    base = load_fvecs(str(ROOT / cfg["data"]["base_path"]), n_vectors=n_base)
    dim = base.shape[1]
    print(f"  shape: {base.shape}")

    print(f"Building index (M={M}, ef_construction={ef_construction})...")
    index = hnswlib.Index(space="l2", dim=dim)
    index.init_index(max_elements=n_base, ef_construction=ef_construction,
                     M=M, random_seed=cfg["index"]["seed"])

    t0 = time.perf_counter()
    index.add_items(base, num_threads=-1)
    elapsed = time.perf_counter() - t0

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index.save_index(str(index_path))
    print(f"Saved to {index_path}  ({elapsed:.1f}s, M={M}, ef_construction={ef_construction})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if index already exists.")
    args = parser.parse_args()
    build(args.config, force=args.force)


if __name__ == "__main__":
    main()
