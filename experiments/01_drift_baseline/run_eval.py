import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.drift_eval import run_drift_evaluation, save_results


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def print_summary(results, label, ef_values):
    n_epochs = len(set(r["epoch"] for r in results))
    print(f"\nSchedule: {label}  |  {n_epochs} epochs  |  ef_search={ef_values}")
    print(f"{'ef':>6}  {'epoch 0':>10}  {'epoch 24':>10}")
    for ef in ef_values:
        rows = [r for r in results if r["ef_search"] == ef]
        r0  = next((r["recall"] for r in rows if r["epoch"] == 0), float("nan"))
        r24 = next((r["recall"] for r in rows if r["epoch"] == n_epochs - 1), float("nan"))
        print(f"{ef:>6}  {r0:>10.4f}  {r24:>10.4f}")


def run(config_path):
    cfg = load_config(config_path)
    index_path = str(ROOT / cfg["output"]["index_path"])
    ef_values = cfg["eval"]["ef_search"]
    k = cfg["eval"]["k"]

    schedules = [
        ("gradual", str(ROOT / cfg["output"]["dataset_gradual_path"]),
                    str(ROOT / cfg["output"]["results_gradual"])),
        ("sudden",  str(ROOT / cfg["output"]["dataset_sudden_path"]),
                    str(ROOT / cfg["output"]["results_sudden"])),
    ]

    for label, dataset_path, out_csv in schedules:
        print(f"\nEvaluating {label} schedule...")
        results = run_drift_evaluation(index_path, dataset_path, ef_values, k)
        save_results(results, out_csv)
        print(f"Results saved to {out_csv}")
        print_summary(results, label, ef_values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
