import csv
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.baselines.periodic_rebuild import PeriodicRebuildBaseline
from src.drift.dataset import load_drift_dataset


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def run(config_path):
    cfg = load_config(config_path)
    dataset_path = str(ROOT / cfg["output"]["dataset_gradual_path"])
    out_csv = str(ROOT / "experiments/01_drift_baseline/results/periodic_rebuild.csv")

    print(f"Loading dataset from {dataset_path} ...")
    dataset = load_drift_dataset(dataset_path)
    epochs = dataset["epochs"]
    groundtruth = dataset["groundtruth"]
    base = dataset["base"]

    rebuild_interval = 5
    k = cfg["eval"]["k"]
    ef_search = cfg["eval"]["ef_search"][2]  # mid-range ef

    print(f"Building initial index ({len(base)} vectors, M={cfg['index']['M']}) ...")
    baseline = PeriodicRebuildBaseline(
        base=base,
        M=cfg["index"]["M"],
        ef_construction=cfg["index"]["ef_construction"],
        rebuild_interval=rebuild_interval,
        seed=cfg["index"]["seed"],
    )

    results = []
    for epoch_idx, (queries, gt) in enumerate(zip(epochs, groundtruth)):
        row = baseline.process_epoch(epoch_idx, queries, gt, k=k, ef_search=ef_search)
        results.append(row)
        tag = " [REBUILT]" if row["rebuilt_this_epoch"] else ""
        print(
            f"  epoch {epoch_idx:2d}  recall@{k}={row['recall_at_k']:.4f}"
            f"  latency={row['mean_latency_ms']:.2f}ms{tag}"
        )

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    n_rebuilds = sum(1 for r in results if r["rebuilt_this_epoch"])
    total = baseline.total_rebuild_time()
    mean = total / n_rebuilds if n_rebuilds else 0.0
    print(f"\nResults saved to {out_csv}")
    print(f"Total rebuild time: {total:.2f}s over {n_rebuilds} events "
          f"(mean {mean:.2f}s per rebuild)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    args = parser.parse_args()
    run(args.config)
