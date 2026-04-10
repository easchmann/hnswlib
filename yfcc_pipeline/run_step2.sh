#!/bin/bash
#SBATCH --job-name=yfcc_download
#SBATCH --output=logs/step2_%A_%a.out
#SBATCH --error=logs/step2_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=defq
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --array=0-19

NUM_TASKS=20

mkdir -p logs

echo "array job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}/${NUM_TASKS}"
echo "started: $(date) on $(hostname)"

module load python
pip install requests tqdm pandas pillow --quiet --user

cd /scratch/easchman/thesis

python yfcc_pipeline/step2_download_images.py \
    --sampled_csv  data/yfcc_sampled/sampled_urls.csv \
    --image_dir    data/yfcc_sampled/images \
    --manifest_out data/yfcc_sampled/downloaded_manifest.csv \
    --n_workers    64 \
    --chunk_size   50000 \
    --task_id      ${SLURM_ARRAY_TASK_ID} \
    --num_tasks    ${NUM_TASKS}

echo "task ${SLURM_ARRAY_TASK_ID} done: $(date)"
