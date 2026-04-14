#!/bin/bash
#SBATCH --job-name=bench-cpu
#SBATCH --account=def-kjerbi_gpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/bench-cpu-%j.out
#SBATCH --error=logs/bench-cpu-%j.err
#SBATCH --chdir=/home/sesma/amica-python

source /home/sesma/envs/amica/bin/activate
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4

for SEED in 0 1 2; do
  # MNE sample, full-batch (small enough)
  python -u validation/benchmark_report.py --dataset mne --device cpu --seed $SEED --chunk_size 0 --max_iter 20
  # MoBI sub-01, chunked (full-batch is 14GB+ RAM)
  python -u validation/benchmark_report.py --dataset mobi --device cpu --seed $SEED --chunk_size 1024 --max_iter 20
done
