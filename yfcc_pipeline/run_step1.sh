#!/bin/bash
#SBATCH --job-name=yfcc_sample
#SBATCH --output=logs/step1_%j.out
#SBATCH --error=logs/step1_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --ntasks=1

mkdir -p logs

echo "started: $(date) on $(hostname)"

module load python
pip install datasets pandas tqdm --quiet --user

cd /scratch/easchman/thesis

python yfcc_pipeline/step1_download_index_and_sample.py

echo "done: $(date)"
echo "rows: $(wc -l < data/yfcc_sampled/sampled_urls.csv)"
