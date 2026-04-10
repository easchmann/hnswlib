#!/bin/bash
#SBATCH --job-name=hnsw_continuous_drift
#SBATCH --output=logs/continuous_drift_%j.out
#SBATCH --error=logs/continuous_drift_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1

echo "started: $(date)"
echo "node:    $(hostname)"
echo "gpu:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo ""

module load cuda/12.4/toolkit/12.4.1
module load python3

pip install "numpy<2" pandas --quiet --user
pip install faiss-gpu --quiet --user 2>/dev/null || pip install faiss-cpu --quiet --user

echo "building hnswlib..."
cd /scratch/easchman/thesis/hnswlib
python -m pip install -e . --quiet --user

python -c "import faiss"

echo ""
echo "running continuous drift experiment..."
cd /scratch/easchman/thesis

python hnswlib/experiments/experiment_continuous_drift.py \
    --mode    large \
    --out_dir results_continuous_drift_large

echo ""
echo "finished: $(date)"