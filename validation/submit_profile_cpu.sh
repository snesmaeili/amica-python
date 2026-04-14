#!/bin/bash
#SBATCH --job-name=profile-cpu
#SBATCH --account=def-kjerbi_gpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/profile-cpu-%j.out
#SBATCH --error=logs/profile-cpu-%j.err
#SBATCH --chdir=/home/sesma/amica-python

source /home/sesma/envs/amica/bin/activate
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4

python -u validation/profile_cpu.py
