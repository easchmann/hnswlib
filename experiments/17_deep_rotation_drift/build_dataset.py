"""Experiment 17: build rotation-drift dataset on Deep-image-96."""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.drift.dataset import (
    build_rotation_drift_sequence,
    characterise_drift,
    compute_groundtruth,
    make_rotation_generator,
    save_drift_dataset,
)
from src.drift.deep_cluster_drift import load_deep_hdf5


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def validate_monotone(diagnostics, label):
    """Crash if any drift diagnostic is non-monotone as t increases."""
    for key in ("ood_distance", "mmd_squared", "mean_nn_distance"):
        violations = [
            i for i in range(1, len(diagnostics))
            if diagnostics[i]["t_value"] is not None
            and diagnostics[i - 1]["t_value"] is not None
            and diagnostics[i]["t_value"] > diagnostics[i - 1]["t_value"]
            and diagnostics[i][key] < diagnostics[i - 1][key] - 1e-6
        ]
        if violations:
            raise RuntimeError(
                f"[{label}] {key} is non-monotone at epochs {violations} — "
                "rotation should always produce monotone drift diagnostics."
            )
        print(f"  [{label}] {key}: OK monotone")


def build(config_path, schedule_key):
    cfg = load_config(config_path)
    drift_cfg = cfg["drift"]

    print(f"Loading HDF5: {cfg['data']['hdf5_path']}")
    base, query_pool = load_deep_hdf5(cfg["data"]["hdf5_path"], n_vectors=None)
    print(f"  base:       {base.shape}")
    print(f"  query_pool: {query_pool.shape}")

    base_norms = np.linalg.norm(base[:1000], axis=1)
    query_norms = np.linalg.norm(query_pool[:1000], axis=1)
    print(f"  base norms  mean={base_norms.mean():.4f}  std={base_norms.std():.4f}")
    print(f"  query norms mean={query_norms.mean():.4f}  std={query_norms.std():.4f}")

    A = make_rotation_generator(dim=96, seed=drift_cfg["rotation_seed"])
    schedule = drift_cfg[schedule_key]

    print(f"Building drift sequence ({schedule_key}, {len(schedule)} epochs)...")
    epochs = build_rotation_drift_sequence(
        query_pool, A, schedule, drift_cfg["epoch_size"], drift_cfg["base_seed"]
    )

    print("Characterising drift...")
    diagnostics = characterise_drift(base, epochs, schedule=schedule)
    for i, d in enumerate(diagnostics):
        print(
            f"  epoch {i:2d}  t={d['t_value']}  ood={d['ood_distance']:.4f}  "
            f"mmd2={d['mmd_squared']:.6f}  nn_dist={d['mean_nn_distance']:.4f}"
        )
    validate_monotone(diagnostics, schedule_key)

    print("Computing per-epoch ground truth (faiss exact, k=100)...")
    groundtruth_per_epoch = []
    for i, epoch_queries in enumerate(epochs):
        print(f"  epoch {i + 1}/{len(epochs)}")
        groundtruth_per_epoch.append(compute_groundtruth(base, epoch_queries, k=100))

    out_key = "dataset_gradual_path" if schedule_key == "schedule_gradual" else "dataset_sudden_path"
    out_path = ROOT / cfg["output"][out_key]
    out_path.mkdir(parents=True, exist_ok=True)

    save_drift_dataset(
        str(out_path),
        base,
        epochs,
        groundtruth_per_epoch,
        {**cfg, "schedule_used": schedule_key},
        diagnostics,
    )
    print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument(
        "--schedule",
        choices=["schedule_gradual", "schedule_sudden"],
        default="schedule_gradual",
    )
    args = parser.parse_args()
    build(args.config, args.schedule)


if __name__ == "__main__":
    main()
