"""Build drift datasets for experiment 10 plateau stability analysis."""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.drift.dataset import (
    build_rotation_drift_sequence,
    characterise_drift,
    compute_groundtruth,
    load_fvecs,
    make_rotation_generator,
    save_drift_dataset,
)

SCHEDULE_TO_OUTPUT_KEY = {
    "schedule_sudden_plateau": "dataset_sudden_plateau_path",
    "schedule_gradual_plateau": "dataset_gradual_plateau_path",
}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def validate_monotone(diagnostics, label):
    for key in ("ood_distance", "mmd_squared", "mean_nn_distance"):
        violations = [
            i for i in range(1, len(diagnostics))
            if diagnostics[i]["t_value"] is not None
            and diagnostics[i - 1]["t_value"] is not None
            and diagnostics[i]["t_value"] > diagnostics[i - 1]["t_value"]
            and diagnostics[i][key] < diagnostics[i - 1][key] - 1e-6
        ]
        status = f"WARNING non-monotone at epochs {violations}" if violations else "OK monotone"
        print(f"  [{label}] {key}: {status}")


def build(config_path, schedule_key):
    cfg = load_config(config_path)
    drift = cfg["drift"]

    print(f"Loading base: {cfg['data']['base_path']}")
    base = load_fvecs(str(ROOT / cfg["data"]["base_path"]), n_vectors=cfg["data"]["n_base"])
    print(f"  {base.shape}")

    print(f"Loading queries: {cfg['data']['query_path']}")
    queries = load_fvecs(str(ROOT / cfg["data"]["query_path"]), n_vectors=cfg["data"]["n_queries_pool"])
    print(f"  {queries.shape}")

    A = make_rotation_generator(base.shape[1], drift["seed"])
    schedule = drift[schedule_key]
    n_epochs = len(schedule)

    print(f"Building drift sequence ({schedule_key}, {n_epochs} epochs)...")
    epochs = build_rotation_drift_sequence(queries, A, schedule, drift["epoch_size"], drift["seed"])

    print("Characterising drift...")
    diagnostics = characterise_drift(base, epochs, schedule=schedule)
    validate_monotone(diagnostics, schedule_key)

    print("Computing per-epoch ground truth...")
    k_gt = cfg["eval"]["k_groundtruth"]
    groundtruth_per_epoch = []
    for i, epoch in enumerate(epochs):
        print(f"  epoch {i + 1}/{n_epochs}")
        groundtruth_per_epoch.append(compute_groundtruth(base, epoch, k=k_gt))

    out_key = SCHEDULE_TO_OUTPUT_KEY[schedule_key]
    out_path = ROOT / cfg["output"][out_key]
    out_path.mkdir(parents=True, exist_ok=True)

    save_drift_dataset(
        str(out_path), base, epochs, groundtruth_per_epoch,
        {**cfg, "schedule_used": schedule_key},
        diagnostics,
    )
    print(f"Saved dataset to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument(
        "--schedule",
        choices=list(SCHEDULE_TO_OUTPUT_KEY.keys()),
        required=True,
    )
    args = parser.parse_args()
    build(args.config, args.schedule)


if __name__ == "__main__":
    main()
