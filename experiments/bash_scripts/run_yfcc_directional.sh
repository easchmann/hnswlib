#!/bin/bash
#SBATCH --job-name=yfcc_dir_drift
#SBATCH --output=logs/yfcc_dir_drift_%j.out
#SBATCH --error=logs/yfcc_dir_drift_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=defq
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --ntasks=1

mkdir -p logs

echo "started: $(date) on $(hostname)"

cd /scratch/easchman/thesis

/usr/bin/python3 hnswlib/experiments/experiment_yfcc_directional_drift.py \
    --embeddings_path data/yfcc_sampled/embeddings/embeddings_float32.npy \
    --metadata_path   data/yfcc_sampled/embeddings/metadata.csv \
    --n_queries       1000 \
    --shift_sigmas    0 1 2 3 4 6 8 10 12 16 \
    --ef_sweep        10 20 50 100 200 500 \
    --M               16 \
    --ef_construction 200 \
    --out_dir         results_yfcc_directional_drift_entireDataset

echo "done: $(date)"