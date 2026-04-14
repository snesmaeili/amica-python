#!/bin/bash
#SBATCH --job-name=bench-chunking
#SBATCH --account=def-kjerbi_gpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/bench-chunking-%j.out
#SBATCH --error=logs/bench-chunking-%j.err
#SBATCH --chdir=/home/sesma/amica-python

source /home/sesma/envs/amica/bin/activate
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4

python -u validation/benchmark_chunking.py
