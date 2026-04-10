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

# 128G: 10M x 128 float32 = 5GB vectors, ~8GB HNSW index, faiss GT overhead

echo "started: $(date)"
echo "node:    $(hostname)"
echo "gpu:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo ""

module load cuda/12.4/toolkit/12.4.1
module load python3

pip install "numpy<2" pandas scikit-learn --quiet --user
pip install faiss-gpu --quiet --user 2>/dev/null || pip install faiss-cpu --quiet --user

echo "building hnswlib..."
cd /scratch/easchman/thesis/hnswlib
python3 -m pip install -e . --quiet --user

python3 -c "import faiss; print(f'faiss sees {faiss.get_num_gpus()} GPU(s)')"

SIFT_DIR="/scratch/easchman/thesis/data/sift"

if [ -f "$SIFT_DIR/bigann_base_10M.bvecs" ]; then
    SIFT_PATH="$SIFT_DIR/bigann_base_10M.bvecs"
    N_TOTAL=10000000
    echo "using bigann_base_10M.bvecs  (n_total=$N_TOTAL)"
elif [ -f "$SIFT_DIR/bigann_base.bvecs" ]; then
    SIFT_PATH="$SIFT_DIR/bigann_base.bvecs"
    N_TOTAL=10000000
    echo "using bigann_base.bvecs  (n_total=$N_TOTAL)"
else
    echo "no SIFT bvecs file found in $SIFT_DIR"
    echo "run: pv bigann_base.bvecs.gz | gunzip | head -c 1400000000 > bigann_base_10M.bvecs"
    exit 1
fi

# --- run ---------------------------------------------------------------------

echo ""
echo "running SIFT drift experiment..."
cd /scratch/easchman/thesis

python3 hnswlib/experiments/experiment_sift_drift_pc1split.py \
    --sift_path      "$SIFT_PATH" \
    --n_total        $N_TOTAL \
    --n_queries      1000 \
    --n_shift_levels 10 \
    --ef_sweep       10 20 50 100 200 500 \
    --M              16 \
    --ef_construction 200 \
    --k              10 \
    --out_dir        results_sift_drift_pc1split_${N_TOTAL}

echo ""
echo "finished: $(date)"