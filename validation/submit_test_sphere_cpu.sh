#!/bin/bash
#SBATCH --job-name=test-sphere-cpu
#SBATCH --account=def-kjerbi_gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/test-sphere-cpu-%j.out
#SBATCH --error=logs/test-sphere-cpu-%j.err
#SBATCH --chdir=/home/sesma/amica-python
source /home/sesma/envs/amica/bin/activate
export JAX_PLATFORMS=cpu
python -u validation/test_fortran_sphere.py
