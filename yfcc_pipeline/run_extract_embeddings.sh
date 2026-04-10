#!/bin/bash
#SBATCH --job-name=yfcc_embed
#SBATCH --output=logs/embed_%j.out
#SBATCH --error=logs/embed_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --ntasks=1

mkdir -p logs

echo "started: $(date) on $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"

module load cuda/12.4/toolkit/12.4.1
module load python
pip install torch torchvision tqdm pandas pillow numpy --quiet --user

# pick batch/shard size based on available VRAM
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')

if   [ "$GPU_MEM_MB" -ge 79000 ] 2>/dev/null; then BATCH=1024; SHARD=500000  # A100 80GB
elif [ "$GPU_MEM_MB" -ge 39000 ] 2>/dev/null; then BATCH=512;  SHARD=300000  # A100 40GB
elif [ "$GPU_MEM_MB" -ge 30000 ] 2>/dev/null; then BATCH=256;  SHARD=200000  # V100 32GB
else                                                BATCH=128;  SHARD=100000
fi
echo "batch=$BATCH shard=$SHARD (VRAM: ${GPU_MEM_MB}MB)"

cd /scratch/easchman/thesis

python yfcc_pipeline/step3_extract_embeddings.py \
    --batch_size  $BATCH \
    --num_workers 8 \
    --shard_size  $SHARD \
    --image_dir   data/yfcc_sampled/images \
    --manifest    data/yfcc_sampled/downloaded_manifest.csv \
    --output_dir  data/yfcc_sampled/embeddings

echo "done: $(date)"
echo "disk usage after cleanup:"
du -sh /scratch/easchman/thesis/data/yfcc_sampled/
