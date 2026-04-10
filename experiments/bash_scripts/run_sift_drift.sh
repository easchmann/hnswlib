#!/bin/bash
#SBATCH --job-name=hnsw_sift_drift
#SBATCH --output=logs/sift_drift_%j.out
#SBATCH --error=logs/sift_drift_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --ntasks=1

# 128G because 10M* 128 float32 = 5GB for the vectors alone,
# plus index memory (~8GB for HNSW at M=16) and faiss GT overhead

echo "started: $(date)"
echo "node:    $(hostname)"
echo "gpu:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo ""

module load cuda/12.4/toolkit/12.4.1
module load python3

pip install "numpy<2" pandas scikit-learn --quiet --user
pip install faiss-gpu --quiet --user 2>/dev/null || pip install faiss-cpu --quiet --user
pip install h5py --quiet --user

echo "building hnswlib..."
cd /scratch/easchman/thesis/hnswlib
python3 -m pip install -e . --quiet --user

python3 -c "import faiss; print(f'faiss sees {faiss.get_num_gpus()} GPU(s)')"

echo "using SIFT-1B (bigann), n_index=$N_INDEX"


# --- run experiment ----------------------------------------------------------

echo ""
echo "running SIFT drift experiment (n_index=$N_INDEX)..."


cd /scratch/easchman/thesis
python3 hnswlib/experiments/experiment_sift_drift.py \
    --sift_path data/sift/bigann_base_10M.bvecs \
    --n_index 10000000 \
    --n_queries 1000 \
    --n_shift_levels 10 \
    --ef_sweep 10 20 50 100 200 500 \
    --out_dir results_sift_drift_10M

echo ""
echo "finished: $(date)"