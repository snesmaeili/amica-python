#!/bin/bash
#SBATCH --job-name=bench-gpu
#SBATCH --account=def-kjerbi_gpu
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/bench-gpu-%j.out
#SBATCH --error=logs/bench-gpu-%j.err
#SBATCH --chdir=/home/sesma/amica-python

source /home/sesma/envs/amica/bin/activate
module load cuda/12.6 2>/dev/null || true
export JAX_PLATFORMS=gpu

for SEED in 0 1 2; do
  python -u validation/benchmark_report.py --dataset mne  --device gpu --seed $SEED --chunk_size 0 --max_iter 20
  python -u validation/benchmark_report.py --dataset mobi --device gpu --seed $SEED --chunk_size 0 --max_iter 20
done
