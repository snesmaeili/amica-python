#!/bin/bash
#SBATCH --job-name=compare-20iter-cpu
#SBATCH --account=def-kjerbi_gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/compare-20iter-cpu-%j.out
#SBATCH --error=logs/compare-20iter-cpu-%j.err
#SBATCH --chdir=/home/sesma/amica-python
source /home/sesma/envs/amica/bin/activate
export JAX_PLATFORMS=cpu
python -u validation/compare_20iter.py
