#!/bin/bash
#SBATCH --job-name=yfcc_temporal
#SBATCH --output=logs/yfcc_temporal_%j.out
#SBATCH --error=logs/yfcc_temporal_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --ntasks=1

mkdir -p logs

echo "started: $(date) on $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"

module load python
pip install pandas numpy hnswlib faiss-gpu --quiet --user

cd /scratch/easchman/thesis

python hnswlib/experiments/experiment_yfcc_temporal_drift.py \
    --embeddings_path data/yfcc_sampled/embeddings/embeddings_float32.npy \
    --metadata_path   data/yfcc_sampled/embeddings/metadata.csv \
    --n_queries       1000 \
    --ef_sweep        10 20 50 100 200 500 \
    --M               16 \
    --ef_construction 200 \
    --out_dir         results_yfcc_temporal_drift

echo "done: $(date)"