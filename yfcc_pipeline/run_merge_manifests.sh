#!/bin/bash
#SBATCH --job-name=yfcc_merge
#SBATCH --output=logs/merge_%j.out
#SBATCH --error=logs/merge_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --ntasks=1
# submit with --dependency=afterok:<step2_array_job_id>

mkdir -p logs

echo "started: $(date)"

module load python
cd /scratch/easchman/thesis

python yfcc_pipeline/merge_manifests.py \
    --manifest_dir data/yfcc_sampled \
    --output       data/yfcc_sampled/downloaded_manifest.csv

echo "done: $(date)"
