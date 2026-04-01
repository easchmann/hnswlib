#!/bin/bash
#SBATCH --job-name=yfcc_dino_embed
#SBATCH --output=logs/embed_%j.out
#SBATCH --error=logs/embed_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --ntasks=1


echo "Job started:  $(date)"
echo "Node:         $(hostname)"
echo ""
 
# load modules
module load cuda/12.4/toolkit/12.4.1
module load python
 
echo "Python:  $(python --version)"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
 
# install dependencies
echo ""
echo "Installing dependencies..."
pip install torch torchvision tqdm pandas pillow numpy --quiet --user
 
# auto-detect batch size from GPU VRAM
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
             2>/dev/null | head -1 | tr -d ' ')
 
if   [ "$GPU_MEM_MB" -ge 79000 ] 2>/dev/null; then BATCH_SIZE=1024  # A100 80GB
elif [ "$GPU_MEM_MB" -ge 39000 ] 2>/dev/null; then BATCH_SIZE=512   # A100 40GB
elif [ "$GPU_MEM_MB" -ge 30000 ] 2>/dev/null; then BATCH_SIZE=256   # V100 32GB
else                                                BATCH_SIZE=128   # anything else
fi
echo "Batch size: $BATCH_SIZE (GPU VRAM: ${GPU_MEM_MB} MB)"
 
# run step 3 
cd /scratch/easchman/thesis
 
echo ""
echo "Images available: $(find yfcc_sampled/images -name '*.jpg' 2>/dev/null | wc -l)"
echo ""
echo "Starting embedding extraction..."
 
python yfcc_pipeline/step3_extract_embeddings.py \
    --batch_size $BATCH_SIZE \
    --num_workers 8 \
    --image_dir  yfcc_sampled/images \
    --manifest   yfcc_sampled/downloaded_manifest.csv \
    --output_dir yfcc_sampled/embeddings
 
echo ""
echo "Job finished: $(date)"
 